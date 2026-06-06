"""High-rate Vive tracker sampling and UR servo control."""

import threading
import time

import numpy as np


class HighRateArmController:
    """Runs tracker sampling and servoL in threads independent of recording."""

    def __init__(
        self,
        robot,
        tracker_processor,
        tracker_device,
        tracker_frequency=90.0,
        servo_frequency=125.0,
        tracker_timeout=0.25,
    ):
        self.robot = robot
        self.tracker_processor = tracker_processor
        self.tracker_device = tracker_device
        self.tracker_frequency = float(tracker_frequency)
        self.servo_frequency = float(servo_frequency)
        self.tracker_timeout = float(tracker_timeout)

        if self.tracker_frequency <= 0:
            raise ValueError("tracker_frequency must be positive")
        if self.servo_frequency <= 0:
            raise ValueError("servo_frequency must be positive")
        if self.tracker_timeout <= 0:
            raise ValueError("tracker_timeout must be positive")

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
        self._latest_motion = None
        self._latest_target_time = None
        self._error = None
        self._threads = []

    def start(self):
        if self._threads:
            raise RuntimeError("HighRateArmController is already running")

        self.tracker_processor.clear_reference()
        self._stop_event.clear()
        self._target_ready.clear()
        self._latest_motion = None
        self._latest_target_time = None
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
                with self._motion_lock:
                    self._latest_motion = {
                        key: np.asarray(value).copy()
                        for key, value in motion.items()
                    }
                    self._latest_target_time = now
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
            with self._motion_lock:
                target_pose = self._latest_motion["target_pose"].copy()
                target_time = self._latest_target_time

            target_age = time.monotonic() - target_time
            if target_age > self.tracker_timeout:
                raise RuntimeError(
                    f"tracker target is stale ({target_age:.3f}s > "
                    f"{self.tracker_timeout:.3f}s)"
                )

            if self.robot.servo(target_pose) is False:
                raise RuntimeError("servoL command was rejected")
            self.robot.wait_servo_period(period_start)

    def _wait_until_next_tick(self, previous_tick, period):
        next_tick = previous_tick + period
        delay = next_tick - time.monotonic()
        if delay > 0:
            self._stop_event.wait(delay)
            return next_tick

        missed_periods = int(-delay // period) + 1
        return next_tick + missed_periods * period
