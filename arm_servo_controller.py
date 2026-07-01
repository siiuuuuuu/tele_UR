"""High-rate Vive tracker sampling and UR servo control."""

import math
import threading
import time
from collections import deque

import numpy as np


class HighRateArmController:
    """Runs tracker sampling and servoL in threads independent of recording."""

    def __init__(
        self,
        robot,
        tracker_processor,
        tracker_device,
        tracker_frequency=60.0,
        servo_frequency=120.0,
        tracker_timeout=0.25,
        interpolation_delay=None,
        action_history_size=128,
    ):
        self.robot = robot
        self.tracker_processor = tracker_processor
        self.tracker_device = tracker_device
        self.tracker_frequency = float(tracker_frequency)
        self.servo_frequency = float(servo_frequency)
        self.tracker_timeout = float(tracker_timeout)
        self.action_history_size = int(action_history_size)

        if self.tracker_frequency <= 0:
            raise ValueError("tracker_frequency must be positive")
        if self.servo_frequency <= 0:
            raise ValueError("servo_frequency must be positive")
        if self.tracker_timeout <= 0:
            raise ValueError("tracker_timeout must be positive")
        if self.action_history_size <= 0:
            raise ValueError("action_history_size must be positive")

        self.interpolation_delay = (
            1.0 / self.tracker_frequency
            if interpolation_delay is None
            else float(interpolation_delay)
        )
        if self.interpolation_delay < 0:
            raise ValueError("interpolation_delay must be non-negative")

        servo_dt = 1.0 / self.servo_frequency
        if not np.isclose(self.robot.servo_dt, servo_dt, rtol=0.0, atol=1e-9):
            raise ValueError(
                "robot servo_dt must match servo_frequency: "
                f"{self.robot.servo_dt:.9f} != {servo_dt:.9f}"
            )
        control_frequency = self.robot.control_frequency
        if control_frequency is None or not np.isclose(
            control_frequency, self.servo_frequency, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                "robot RTDE control_frequency must match servo_frequency: "
                f"{control_frequency} != {self.servo_frequency:.9f}"
            )

        self._stop_event = threading.Event()
        self._target_ready = threading.Event()
        self._motion_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._tracker_samples = deque(maxlen=8)
        self._action_history = deque(maxlen=self.action_history_size)
        self._latest_tracker_time = None
        self._latest_tracker_time_ns = None
        self._latest_motion = None
        self._latest_motion_time = None
        self._latest_motion_time_ns = None
        self._error = None
        self._threads = []

    def start(self):
        if self._threads:
            raise RuntimeError("HighRateArmController is already running")

        self.tracker_processor.clear_reference()
        self._stop_event.clear()
        self._target_ready.clear()
        self._tracker_samples.clear()
        self._action_history.clear()
        self._latest_tracker_time = None
        self._latest_tracker_time_ns = None
        self._latest_motion = None
        self._latest_motion_time = None
        self._latest_motion_time_ns = None
        self._error = None

        self._threads = [
            threading.Thread(
                target=self._run_guarded,
                args=("tracker loop", self._tracker_loop),
                name="vive-tracker-loop",
                daemon=True,
            ),
            threading.Thread(
                target=self._run_guarded,
                args=("servo loop", self._servo_loop),
                name="ur-servo-loop",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self):
        self._stop_event.set()
        self._target_ready.set()
        for thread in self._threads:
            thread.join()
        self._threads = []
        try:
            if self.robot.stop_servo() is False:
                raise RuntimeError("servoStop command was rejected")
        except Exception as exc:
            self._record_error("servo stop", exc)

    def latest_motion(self):
        self.raise_if_failed()
        with self._motion_lock:
            if self._latest_motion is None:
                return None
            return {
                key: np.asarray(value).copy()
                for key, value in self._latest_motion.items()
            }

    def latest_action_time_ns(self):
        self.raise_if_failed()
        with self._motion_lock:
            return self._latest_motion_time_ns

    def motion_at_time_ns(self, target_time_ns):
        self.raise_if_failed()
        with self._motion_lock:
            if not self._action_history:
                return None
            motion = self._nearest_by_time(
                self._action_history,
                int(target_time_ns),
                "t_arm_action_host_ns",
            )
            return self._copy_motion(motion)

    def latest_tracker_motion(self):
        self.raise_if_failed()
        with self._motion_lock:
            if not self._tracker_samples:
                return None
            return self._copy_motion(self._tracker_samples[-1][1])

    def raise_if_failed(self):
        with self._error_lock:
            error = self._error
        if error is not None:
            raise error

    def _run_guarded(self, loop_name, loop_fn):
        try:
            loop_fn()
        except Exception as exc:
            self._record_error(loop_name, exc)
            self._stop_event.set()
            self._target_ready.set()

    def _record_error(self, operation, exc):
        with self._error_lock:
            if self._error is None:
                self._error = RuntimeError(f"{operation} failed: {exc}")

    def _tracker_loop(self):
        period = 1.0 / self.tracker_frequency
        next_tick = time.monotonic()
        reference_announced = False

        while not self._stop_event.is_set():
            tracker_mat = self.tracker_processor.read_tracker_mat(self.tracker_device)
            if tracker_mat is not None:
                if not self.tracker_processor.has_reference():
                    self.tracker_processor.reset_reference(tracker_mat)
                    reference_announced = True

                motion = self.tracker_processor.compute(tracker_mat)
                now = time.monotonic()
                now_ns = time.monotonic_ns()
                motion["t_tracker_host_ns"] = now_ns
                with self._motion_lock:
                    self._tracker_samples.append((now, self._copy_motion(motion)))
                    self._latest_tracker_time = now
                    self._latest_tracker_time_ns = now_ns
                self._target_ready.set()

                if reference_announced:
                    print("Initial tracking position recorded. Starting servo control...")
                    reference_announced = False

            next_tick = self._wait_until_next_tick(next_tick, period)

    def _servo_loop(self):
        while not self._stop_event.is_set():
            if not self._target_ready.wait(timeout=0.05):
                continue
            if self._stop_event.is_set():
                return

            period_start = self.robot.init_servo_period()
            now = time.monotonic()
            now_ns = time.monotonic_ns()
            with self._motion_lock:
                samples = list(self._tracker_samples)
                latest_tracker_time = self._latest_tracker_time
                latest_tracker_time_ns = self._latest_tracker_time_ns

            if not samples:
                self.robot.wait_servo_period(period_start)
                continue

            target_age = now - latest_tracker_time
            if target_age > self.tracker_timeout:
                raise RuntimeError(
                    f"tracker target is stale ({target_age:.3f}s > "
                    f"{self.tracker_timeout:.3f}s)"
                )

            target_time = now - self.interpolation_delay
            motion = self._motion_at(samples, target_time)
            motion["t_arm_action_host_ns"] = np.asarray(now_ns, dtype=np.int64)
            motion["t_arm_servo_host_ns"] = np.asarray(now_ns, dtype=np.int64)
            motion["t_tracker_latest_host_ns"] = np.asarray(
                latest_tracker_time_ns,
                dtype=np.int64,
            )
            target_pose = motion["target_pose"].copy()

            if self.robot.servo(target_pose) is False:
                raise RuntimeError("servoL command was rejected")
            with self._motion_lock:
                copied_motion = self._copy_motion(motion)
                self._latest_motion = copied_motion
                self._action_history.append(copied_motion)
                self._latest_motion_time = now
                self._latest_motion_time_ns = now_ns
            self.robot.wait_servo_period(period_start)

    def _wait_until_next_tick(self, previous_tick, period):
        next_tick = previous_tick + period
        delay = next_tick - time.monotonic()
        if delay > 0:
            self._stop_event.wait(delay)
            return next_tick

        missed_periods = int(-delay // period) + 1
        return next_tick + missed_periods * period

    def _motion_at(self, samples, target_time):
        if len(samples) == 1 or target_time <= samples[0][0]:
            return self._with_hold_metadata(samples[0][1])
        if target_time >= samples[-1][0]:
            return self._with_hold_metadata(samples[-1][1])

        for sample_index in range(1, len(samples)):
            t1, motion1 = samples[sample_index]
            if target_time <= t1:
                t0, motion0 = samples[sample_index - 1]
                dt = t1 - t0
                if dt <= 0:
                    return self._with_hold_metadata(motion1)
                alpha = (target_time - t0) / dt
                alpha = min(max(alpha, 0.0), 1.0)
                return self._interpolate_motion(motion0, motion1, alpha)

        return self._with_hold_metadata(samples[-1][1])

    def _with_hold_metadata(self, motion):
        motion = self._copy_motion(motion)
        tracker_ns = np.asarray(motion["t_tracker_host_ns"], dtype=np.int64)
        motion["t_arm_target_host_ns"] = tracker_ns.copy()
        motion["t_tracker0_host_ns"] = tracker_ns.copy()
        motion["t_tracker1_host_ns"] = tracker_ns.copy()
        motion["interpolation_alpha"] = np.asarray(0.0, dtype=np.float64)
        return motion

    def _interpolate_motion(self, motion0, motion1, alpha):
        result_matrix = self._interpolate_matrix(
            motion0["result_matrix"],
            motion1["result_matrix"],
            alpha,
        )
        tool = self.tracker_processor.tool
        motion = {
            "result_matrix": result_matrix,
            "target_pose": np.asarray(tool.mat2xyz_rotvec(result_matrix)),
            "arm_action": np.asarray(tool.mat2xyz_6drot(result_matrix)),
        }
        t0_ns = int(np.asarray(motion0["t_tracker_host_ns"]).item())
        t1_ns = int(np.asarray(motion1["t_tracker_host_ns"]).item())
        motion["t_arm_target_host_ns"] = np.asarray(
            round((1.0 - alpha) * t0_ns + alpha * t1_ns),
            dtype=np.int64,
        )
        motion["t_tracker0_host_ns"] = np.asarray(t0_ns, dtype=np.int64)
        motion["t_tracker1_host_ns"] = np.asarray(t1_ns, dtype=np.int64)
        motion["interpolation_alpha"] = np.asarray(alpha, dtype=np.float64)
        if hasattr(self.tracker_processor, "init_tcp_mat"):
            motion["increment_matrix"] = np.dot(
                tool.se3_inverse(self.tracker_processor.init_tcp_mat),
                result_matrix,
            )
        return motion

    def _interpolate_matrix(self, matrix0, matrix1, alpha):
        matrix0 = np.asarray(matrix0)
        matrix1 = np.asarray(matrix1)
        result_matrix = np.eye(4)
        result_matrix[:3, 3] = (
            (1.0 - alpha) * matrix0[:3, 3] + alpha * matrix1[:3, 3]
        )
        result_matrix[:3, :3] = self._interpolate_rotation(
            matrix0[:3, :3],
            matrix1[:3, :3],
            alpha,
        )
        return result_matrix

    @classmethod
    def _interpolate_rotation(cls, rotation0, rotation1, alpha):
        rotation0 = cls._project_rotation(rotation0)
        rotation1 = cls._project_rotation(rotation1)
        delta_rotation = cls._project_rotation(np.dot(rotation0.T, rotation1))
        delta_rotvec = cls._rotmat_to_rotvec(delta_rotation)
        return cls._project_rotation(
            np.dot(rotation0, cls._rotvec_to_rotmat(alpha * delta_rotvec))
        )

    @staticmethod
    def _project_rotation(rotation):
        u, _, vh = np.linalg.svd(rotation)
        projected = np.dot(u, vh)
        if np.linalg.det(projected) < 0:
            u[:, -1] *= -1
            projected = np.dot(u, vh)
        return projected

    @staticmethod
    def _rotmat_to_rotvec(rotation, eps=1e-9):
        cos_theta = (np.trace(rotation) - 1.0) / 2.0
        cos_theta = min(max(cos_theta, -1.0), 1.0)
        theta = math.acos(cos_theta)
        if theta < eps:
            return np.zeros(3)

        if math.pi - theta < 1e-6:
            diag = np.diag(rotation)
            axis_index = int(np.argmax(diag))
            axis = np.asarray(rotation[:, axis_index], dtype=np.float64).copy()
            axis[axis_index] += 1.0
            axis_norm = np.linalg.norm(axis)
            if axis_norm < eps:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = axis / axis_norm
            return axis * theta

        factor = theta / (2.0 * math.sin(theta))
        return factor * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )

    @staticmethod
    def _rotvec_to_rotmat(rotvec, eps=1e-9):
        theta = np.linalg.norm(rotvec)
        if theta < eps:
            return np.eye(3)

        axis = rotvec / theta
        axis_cross = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        return (
            np.eye(3)
            + math.sin(theta) * axis_cross
            + (1.0 - math.cos(theta)) * np.dot(axis_cross, axis_cross)
        )

    @staticmethod
    def _copy_motion(motion):
        return {key: np.asarray(value).copy() for key, value in motion.items()}

    @staticmethod
    def _nearest_by_time(samples, target_time_ns, time_key):
        samples = list(samples)
        best_sample = samples[-1]
        best_delta = abs(int(np.asarray(best_sample[time_key]).item()) - target_time_ns)
        for sample in reversed(samples[:-1]):
            delta = abs(int(np.asarray(sample[time_key]).item()) - target_time_ns)
            if delta < best_delta:
                best_sample = sample
                best_delta = delta
            else:
                break
        return best_sample
