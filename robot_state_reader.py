"""High-rate UR robot state reader with timestamped history."""

import threading
import time
from collections import deque

import numpy as np


class HighRateRobotStateReader:
    """Polls UR state independently so recording can align by timestamp."""

    def __init__(self, robot, read_frequency=125.0, history_size=128):
        self.robot = robot
        self.read_frequency = float(read_frequency)
        self.history_size = int(history_size)

        if self.read_frequency <= 0:
            raise ValueError("read_frequency must be positive")
        if self.history_size <= 0:
            raise ValueError("history_size must be positive")

        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._state_history = deque(maxlen=self.history_size)
        self._latest_obs = None
        self._error = None
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("HighRateRobotStateReader is already running")

        self._stop_event.clear()
        self._state_history.clear()
        self._latest_obs = None
        self._error = None
        self._thread = threading.Thread(
            target=self._run_guarded,
            name="ur-state-reader-loop",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def latest_obs(self):
        self.raise_if_failed()
        with self._state_lock:
            if self._latest_obs is None:
                return None
            return self._copy_obs(self._latest_obs)

    def obs_at_time_ns(self, target_time_ns):
        self.raise_if_failed()
        with self._state_lock:
            if not self._state_history:
                return None
            obs = self._nearest_by_time(
                self._state_history,
                int(target_time_ns),
                "t_robot_obs_host_ns",
            )
            return self._copy_obs(obs)

    def raise_if_failed(self):
        with self._error_lock:
            error = self._error
        if error is not None:
            raise error

    def _run_guarded(self):
        try:
            self._state_loop()
        except Exception as exc:
            with self._error_lock:
                if self._error is None:
                    self._error = RuntimeError(f"robot state reader failed: {exc}")
            self._stop_event.set()

    def _state_loop(self):
        period = 1.0 / self.read_frequency
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            obs = self.robot.get_obs()
            if obs is not None:
                obs = self._copy_obs(obs)
                obs["t_robot_obs_host_ns"] = np.asarray(
                    time.monotonic_ns(),
                    dtype=np.int64,
                )
                with self._state_lock:
                    self._state_history.append(obs)
                    self._latest_obs = self._copy_obs(obs)

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

    @staticmethod
    def _copy_obs(obs):
        return {key: np.asarray(value).copy() for key, value in obs.items()}
