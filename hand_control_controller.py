"""High-rate MANUS-to-Inspire hand control."""

import threading
import time
from collections import deque

import numpy as np


class JointSmoother:
    """Second-order low-pass smoother for fixed-size joint commands."""

    def __init__(
        self,
        joint_count=6,
        natural_frequency=25.0,
        damping_ratio=0.8,
        input_alpha=0.6,
        minimum=0.0,
        maximum=1000.0,
    ):
        self.joint_count = int(joint_count)
        self.natural_frequency = float(natural_frequency)
        self.damping_ratio = float(damping_ratio)
        self.input_alpha = float(input_alpha)
        self.minimum = float(minimum)
        self.maximum = float(maximum)

        if self.joint_count <= 0:
            raise ValueError("joint_count must be positive")
        if self.natural_frequency <= 0:
            raise ValueError("natural_frequency must be positive")
        if self.damping_ratio <= 0:
            raise ValueError("damping_ratio must be positive")
        if not 0 < self.input_alpha <= 1:
            raise ValueError("input_alpha must be in the range (0, 1]")
        if self.minimum >= self.maximum:
            raise ValueError("minimum must be less than maximum")

        self.k_p = self.natural_frequency * self.natural_frequency
        self.k_d = 2.0 * self.damping_ratio * self.natural_frequency
        self.reset()

    def reset(self):
        self.current_angles = None
        self.current_velocity = np.zeros(self.joint_count, dtype=np.float32)
        self.target_angles = None
        self.smooth_target = None

    def update_target(self, new_target):
        target = np.asarray(new_target, dtype=np.float32).reshape(-1)
        if target.size != self.joint_count:
            raise ValueError(
                f"expected {self.joint_count} joint values, got {target.size}"
            )
        target = np.clip(target, self.minimum, self.maximum)
        self.target_angles = target.copy()

        if self.current_angles is None:
            self.current_angles = target.copy()
            self.current_velocity.fill(0.0)
            self.smooth_target = target.copy()

    def step(self, dt):
        if self.target_angles is None:
            return None
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.smooth_target = (
            (1.0 - self.input_alpha) * self.smooth_target
            + self.input_alpha * self.target_angles
        )
        error = self.smooth_target - self.current_angles
        acceleration = self.k_p * error - self.k_d * self.current_velocity
        self.current_velocity += acceleration * dt
        self.current_angles += self.current_velocity * dt

        clipped_angles = np.clip(self.current_angles, self.minimum, self.maximum)
        outward_velocity = (
            ((clipped_angles <= self.minimum) & (self.current_velocity < 0))
            | ((clipped_angles >= self.maximum) & (self.current_velocity > 0))
        )
        self.current_velocity[outward_velocity] = 0.0
        self.current_angles = clipped_angles
        return self.current_angles.copy()


class HighRateHandController:
    """Sends the latest MANUS command independently from the recording loop."""

    def __init__(
        self,
        manus_source,
        hand_controller,
        control_frequency=120.0,
        manus_timeout=0.25,
        smoothing_natural_frequency=25.0,
        smoothing_damping_ratio=0.8,
        smoothing_input_alpha=0.6,
        command_history_size=128,
    ):
        self.manus_source = manus_source
        self.hand_controller = hand_controller
        self.control_frequency = float(control_frequency)
        self.manus_timeout = float(manus_timeout)
        self.command_history_size = int(command_history_size)

        if self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive")
        if self.manus_timeout <= 0:
            raise ValueError("manus_timeout must be positive")
        if self.command_history_size <= 0:
            raise ValueError("command_history_size must be positive")

        self.smoother = JointSmoother(
            joint_count=6,
            natural_frequency=smoothing_natural_frequency,
            damping_ratio=smoothing_damping_ratio,
            input_alpha=smoothing_input_alpha,
            minimum=0.0,
            maximum=1000.0,
        )
        normalized_frequency = (
            self.smoother.natural_frequency / self.control_frequency
        )
        stability_margin = (
            normalized_frequency * normalized_frequency
            + 4.0 * self.smoother.damping_ratio * normalized_frequency
        )
        if stability_margin >= 4.0:
            raise ValueError(
                "unstable smoother configuration; increase control_frequency "
                "or reduce smoothing_natural_frequency"
            )
        self._stop_event = threading.Event()
        self._command_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._latest_command = None
        self._latest_command_time_ns = None
        self._latest_manus_sample_time_ns = None
        self._latest_manus_seq = None
        self._command_history = deque(maxlen=self.command_history_size)
        self._error = None
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("HighRateHandController is already running")

        self._stop_event.clear()
        self._latest_command = None
        self._latest_command_time_ns = None
        self._latest_manus_sample_time_ns = None
        self._latest_manus_seq = None
        self._command_history.clear()
        self._error = None
        self.smoother.reset()
        self._thread = threading.Thread(
            target=self._run_guarded,
            name="inspire-hand-control-loop",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def latest_command(self):
        self.raise_if_failed()
        with self._command_lock:
            if self._latest_command is None:
                return None
            return self._latest_command.copy()

    def latest_command_sample(self):
        self.raise_if_failed()
        with self._command_lock:
            if self._latest_command is None:
                return None
            return self._copy_command_sample(self._command_history[-1])

    def command_at_time_ns(self, target_time_ns):
        self.raise_if_failed()
        with self._command_lock:
            if not self._command_history:
                return None
            sample = self._nearest_by_time(
                self._command_history,
                int(target_time_ns),
                "t_hand_command_host_ns",
            )
            return self._copy_command_sample(sample)

    def raise_if_failed(self):
        with self._error_lock:
            error = self._error
        if error is not None:
            raise error

    def _run_guarded(self):
        try:
            self._control_loop()
        except Exception as exc:
            with self._error_lock:
                if self._error is None:
                    self._error = RuntimeError(f"hand control loop failed: {exc}")
            self._stop_event.set()

    def _control_loop(self):
        period = 1.0 / self.control_frequency
        next_tick = time.monotonic()
        start_time = next_tick

        while not self._stop_event.is_set():
            sample = self.manus_source.latest_right_sample()
            if sample is None:
                wait_time = time.monotonic() - start_time
                if wait_time > self.manus_timeout:
                    raise RuntimeError(
                        f"no MANUS command received within "
                        f"{self.manus_timeout:.3f}s"
                    )
            else:
                sample_age = (time.monotonic_ns() - int(sample["t_host_ns"])) / 1e9
                if sample_age > self.manus_timeout:
                    raise RuntimeError(
                        f"MANUS command is stale ({sample_age:.3f}s > "
                        f"{self.manus_timeout:.3f}s)"
                    )

                self.smoother.update_target(sample["action"])
                with self._command_lock:
                    self._latest_manus_sample_time_ns = int(sample["t_host_ns"])
                    self._latest_manus_seq = int(sample["seq"])

            smooth_command = self.smoother.step(period)
            if smooth_command is not None:
                command = np.asarray(
                    self.hand_controller.apply(smooth_command),
                    dtype=np.int32,
                ).reshape(-1)
                if command.size != 6:
                    raise RuntimeError(
                        f"expected a 6D hand command, got {command.size} values"
                    )
                with self._command_lock:
                    self._latest_command = command.copy()
                    self._latest_command_time_ns = time.monotonic_ns()
                    command_sample = {
                        "command": command.copy(),
                        "t_hand_action_host_ns": self._latest_command_time_ns,
                        "t_hand_command_host_ns": self._latest_command_time_ns,
                        "t_manus_sample_host_ns": self._latest_manus_sample_time_ns,
                        "manus_seq": self._latest_manus_seq,
                    }
                    self._command_history.append(command_sample)

            next_tick = self._wait_until_next_tick(next_tick, period)

    def _wait_until_next_tick(self, previous_tick, period):
        next_tick = previous_tick + period
        delay = next_tick - time.monotonic()
        if delay > 0:
            self._stop_event.wait(delay)
            return next_tick

        missed_periods = int(-delay // period) + 1
        return next_tick + missed_periods * period

    @staticmethod
    def _copy_command_sample(sample):
        return {
            "command": np.asarray(sample["command"]).copy(),
            "t_hand_action_host_ns": sample.get("t_hand_action_host_ns"),
            "t_hand_command_host_ns": sample.get("t_hand_command_host_ns"),
            "t_manus_sample_host_ns": sample.get("t_manus_sample_host_ns"),
            "manus_seq": sample.get("manus_seq"),
        }

    @staticmethod
    def _nearest_by_time(samples, target_time_ns, time_key):
        samples = list(samples)
        best_sample = samples[-1]
        best_delta = abs(int(best_sample[time_key]) - target_time_ns)
        for sample in reversed(samples[:-1]):
            delta = abs(int(sample[time_key]) - target_time_ns)
            if delta < best_delta:
                best_sample = sample
                best_delta = delta
            else:
                break
        return best_sample
