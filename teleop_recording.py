"""Recording helpers for timestamp-aligned teleoperation episodes."""

from dataclasses import dataclass
import time

import numpy as np


def scalar_int(value, default=-1):
    if value is None:
        return default
    return int(np.asarray(value).item())


def scalar_float(value, default=np.nan):
    if value is None:
        return default
    return float(np.asarray(value).item())


def time_delta_ms(value_ns, anchor_ns, default=np.nan):
    if value_ns is None or anchor_ns is None:
        return default
    return (scalar_int(value_ns) - scalar_int(anchor_ns)) / 1e6


def delta_out_of_tolerance(delta_ms, tolerance_ms):
    return not np.isfinite(delta_ms) or abs(delta_ms) > tolerance_ms


@dataclass
class AlignedInputs:
    record_start_ns: int
    t_camera_read_ns: int
    t_anchor_ns: int
    t_arm_read_ns: int
    t_hand_read_ns: int
    cam_dict: dict
    front_meta: dict
    wrist_meta: dict
    motion: dict
    robot_obs: dict
    hand_sample: dict
    t_arm_action_ns: int
    t_robot_obs_host_ns: int
    t_hand_action_ns: int
    arm_sync_delta_ms: float
    robot_sync_delta_ms: float
    hand_sync_delta_ms: float
    wrist_sync_delta_ms: float


@dataclass
class AlignedSample:
    robot_state: np.ndarray
    cam_dict: dict
    action: np.ndarray
    timestamps: dict


class TimestampBuilder:
    def build(self, aligned):
        timestamps = {
            "t_record_start_ns": aligned.record_start_ns,
            "t_anchor_ns": aligned.t_anchor_ns,
            "t_arm_read_ns": aligned.t_arm_read_ns,
            "t_arm_action_host_ns": aligned.t_arm_action_ns,
            "t_arm_servo_host_ns": scalar_int(
                aligned.motion.get("t_arm_servo_host_ns")
            ),
            "t_arm_target_host_ns": scalar_int(
                aligned.motion.get("t_arm_target_host_ns")
            ),
            "t_tracker0_host_ns": scalar_int(
                aligned.motion.get("t_tracker0_host_ns")
            ),
            "t_tracker1_host_ns": scalar_int(
                aligned.motion.get("t_tracker1_host_ns")
            ),
            "t_tracker_latest_host_ns": scalar_int(
                aligned.motion.get("t_tracker_latest_host_ns")
            ),
            "interpolation_alpha": scalar_float(
                aligned.motion.get("interpolation_alpha")
            ),
            "t_robot_obs_host_ns": aligned.t_robot_obs_host_ns,
            "t_camera_read_ns": aligned.t_camera_read_ns,
            "t_front_camera_host_ns": aligned.front_meta.get("t_host_ns"),
            "t_front_camera_receive_host_ns": aligned.front_meta.get(
                "t_receive_host_ns"
            ),
            "front_camera_dev_ts": aligned.front_meta.get("t_dev_ts"),
            "front_camera_sensor_timestamp_us": aligned.front_meta.get(
                "sensor_timestamp_us"
            ),
            "front_camera_frame_global_mono_ns": aligned.front_meta.get(
                "frame_global_mono_ns"
            ),
            "front_camera_timestamp_fit_residual_ms": aligned.front_meta.get(
                "timestamp_fit_residual_ms"
            ),
            "front_camera_receive_minus_image_ms": aligned.front_meta.get(
                "receive_minus_image_ms"
            ),
            "front_camera_seq": aligned.front_meta.get("seq"),
            "front_camera_frame_no": aligned.front_meta.get("frame_no"),
            "t_wrist_camera_host_ns": aligned.wrist_meta.get("t_host_ns"),
            "t_wrist_camera_receive_host_ns": aligned.wrist_meta.get(
                "t_receive_host_ns"
            ),
            "wrist_camera_dev_ts": aligned.wrist_meta.get("t_dev_ts"),
            "wrist_camera_sensor_timestamp_us": aligned.wrist_meta.get(
                "sensor_timestamp_us"
            ),
            "wrist_camera_frame_global_mono_ns": aligned.wrist_meta.get(
                "frame_global_mono_ns"
            ),
            "wrist_camera_timestamp_fit_residual_ms": aligned.wrist_meta.get(
                "timestamp_fit_residual_ms"
            ),
            "wrist_camera_receive_minus_image_ms": aligned.wrist_meta.get(
                "receive_minus_image_ms"
            ),
            "wrist_camera_seq": aligned.wrist_meta.get("seq"),
            "wrist_camera_frame_no": aligned.wrist_meta.get("frame_no"),
            "t_hand_read_ns": aligned.t_hand_read_ns,
            "t_hand_action_host_ns": aligned.t_hand_action_ns,
            "t_hand_command_host_ns": aligned.hand_sample.get(
                "t_hand_command_host_ns"
            ),
            "t_manus_sample_host_ns": aligned.hand_sample.get(
                "t_manus_sample_host_ns"
            ),
            "manus_seq": aligned.hand_sample.get("manus_seq"),
            "t_aligned_arm_action_ns": aligned.t_arm_action_ns,
            "t_aligned_robot_obs_ns": aligned.t_robot_obs_host_ns,
            "t_aligned_hand_action_ns": aligned.t_hand_action_ns,
            "sync_delta_arm_action_ms": aligned.arm_sync_delta_ms,
            "sync_delta_robot_obs_ms": aligned.robot_sync_delta_ms,
            "sync_delta_hand_action_ms": aligned.hand_sync_delta_ms,
            "sync_delta_wrist_camera_ms": aligned.wrist_sync_delta_ms,
        }
        timestamps["t_record_end_ns"] = time.monotonic_ns()
        return timestamps


class AlignedSampleProvider:
    def __init__(
        self,
        camera,
        arm_controller,
        hand_control_worker,
        robot_state_reader,
        use_wrist_img,
        alignment_tolerance_ms,
        timestamp_builder=None,
        wrist_stale_warning_interval_s=2.0,
        front_skip_warning_interval_s=2.0,
        camera_paced=True,
        camera_frame_timeout_ms=None,
        history_wait_timeout_ms=5.0,
        history_wait_sleep_s=0.0005,
    ):
        self.camera = camera
        self.arm_controller = arm_controller
        self.hand_control_worker = hand_control_worker
        self.robot_state_reader = robot_state_reader
        self.use_wrist_img = bool(use_wrist_img)
        self.alignment_tolerance_ms = float(alignment_tolerance_ms)
        self.timestamp_builder = timestamp_builder or TimestampBuilder()
        self.camera_paced = bool(camera_paced)
        self.camera_frame_timeout_ms = camera_frame_timeout_ms
        self.history_wait_timeout_ms = max(0.0, float(history_wait_timeout_ms))
        self.history_wait_sleep_s = max(0.0, float(history_wait_sleep_s))
        self.wrist_stale_warning_interval_ns = int(
            wrist_stale_warning_interval_s * 1e9
        )
        self.front_skip_warning_interval_ns = int(
            front_skip_warning_interval_s * 1e9
        )
        self.last_front_camera_seq = None
        self.last_wrist_stale_warn_ns = 0
        self.last_front_skip_warn_ns = 0

    def reset_warnings(self):
        self.last_wrist_stale_warn_ns = 0
        self.last_front_skip_warn_ns = 0

    def reset_episode(self):
        self.last_front_camera_seq = None
        self.reset_warnings()

    def read(self, episode_start_ns):
        record_start_ns = time.monotonic_ns()
        cam_dict = self._read_camera()
        if cam_dict is None:
            return None
        t_camera_read_ns = time.monotonic_ns()
        front_meta = cam_dict.get("front_meta", {})
        wrist_meta = cam_dict.get("right_meta", {})
        self._consume_front_seq(front_meta)
        t_anchor_ns = front_meta.get("t_host_ns")
        if t_anchor_ns is None or t_anchor_ns < episode_start_ns:
            return None

        self._wait_histories_cover_anchor(t_anchor_ns)

        motion = self.arm_controller.motion_at_time_ns(t_anchor_ns)
        t_arm_read_ns = time.monotonic_ns()
        if motion is None:
            return None

        robot_obs = self.robot_state_reader.obs_at_time_ns(t_anchor_ns)
        if robot_obs is None:
            return None

        hand_sample = self.hand_control_worker.command_at_time_ns(t_anchor_ns)
        t_hand_read_ns = time.monotonic_ns()
        if hand_sample is None:
            return None

        t_arm_action_ns = scalar_int(motion.get("t_arm_action_host_ns"))
        t_robot_obs_host_ns = scalar_int(robot_obs.get("t_robot_obs_host_ns"))
        t_hand_action_ns = scalar_int(hand_sample.get("t_hand_action_host_ns"))
        arm_sync_delta_ms = time_delta_ms(t_arm_action_ns, t_anchor_ns)
        robot_sync_delta_ms = time_delta_ms(t_robot_obs_host_ns, t_anchor_ns)
        hand_sync_delta_ms = time_delta_ms(t_hand_action_ns, t_anchor_ns)
        wrist_sync_delta_ms = (
            time_delta_ms(wrist_meta.get("t_host_ns"), t_anchor_ns)
            if self.use_wrist_img
            else 0.0
        )

        if self._wrist_is_stale(wrist_sync_delta_ms):
            return None
        if (
            delta_out_of_tolerance(
                arm_sync_delta_ms,
                self.alignment_tolerance_ms,
            )
            or delta_out_of_tolerance(
                robot_sync_delta_ms,
                self.alignment_tolerance_ms,
            )
            or delta_out_of_tolerance(
                hand_sync_delta_ms,
                self.alignment_tolerance_ms,
            )
        ):
            return None

        hand_command = hand_sample["command"]
        hand_action_array = hand_command.astype(np.float32) / 1000.0
        arm_action = motion["arm_action"]
        action = np.concatenate((arm_action, hand_action_array))

        aligned = AlignedInputs(
            record_start_ns=record_start_ns,
            t_camera_read_ns=t_camera_read_ns,
            t_anchor_ns=t_anchor_ns,
            t_arm_read_ns=t_arm_read_ns,
            t_hand_read_ns=t_hand_read_ns,
            cam_dict=cam_dict,
            front_meta=front_meta,
            wrist_meta=wrist_meta,
            motion=motion,
            robot_obs=robot_obs,
            hand_sample=hand_sample,
            t_arm_action_ns=t_arm_action_ns,
            t_robot_obs_host_ns=t_robot_obs_host_ns,
            t_hand_action_ns=t_hand_action_ns,
            arm_sync_delta_ms=arm_sync_delta_ms,
            robot_sync_delta_ms=robot_sync_delta_ms,
            hand_sync_delta_ms=hand_sync_delta_ms,
            wrist_sync_delta_ms=wrist_sync_delta_ms,
        )
        return AlignedSample(
            robot_state=robot_obs["state"],
            cam_dict=cam_dict,
            action=action,
            timestamps=self.timestamp_builder.build(aligned),
        )

    def _wrist_is_stale(self, wrist_sync_delta_ms):
        stale = self.use_wrist_img and delta_out_of_tolerance(
            wrist_sync_delta_ms,
            self.alignment_tolerance_ms,
        )
        if not stale:
            return False

        now_ns = time.monotonic_ns()
        if now_ns - self.last_wrist_stale_warn_ns > self.wrist_stale_warning_interval_ns:
            print(
                "Skipping frame: wrist camera timestamp is "
                f"{wrist_sync_delta_ms:.1f} ms from front camera "
                f"(tolerance {self.alignment_tolerance_ms:.1f} ms).\r"
            )
            self.last_wrist_stale_warn_ns = now_ns
        return True

    def _read_camera(self):
        if self.camera_paced and hasattr(self.camera, "read_next"):
            return self.camera.read_next(
                last_front_seq=self.last_front_camera_seq,
                timeout_ms=self.camera_frame_timeout_ms,
            )
        return self.camera()

    def _consume_front_seq(self, front_meta):
        front_seq = scalar_int(front_meta.get("seq"), default=None)
        if front_seq is None:
            return

        previous_seq = self.last_front_camera_seq
        self.last_front_camera_seq = front_seq
        if previous_seq is None:
            return

        skipped = front_seq - previous_seq - 1
        if skipped <= 0:
            return

        now_ns = time.monotonic_ns()
        if now_ns - self.last_front_skip_warn_ns > self.front_skip_warning_interval_ns:
            print(
                "Skipping frame: front camera sequence jumped by "
                f"{skipped} frame(s) "
                f"({previous_seq} -> {front_seq}).\r"
            )
            self.last_front_skip_warn_ns = now_ns

    def _wait_histories_cover_anchor(self, t_anchor_ns):
        if self.history_wait_timeout_ms <= 0.0:
            return False

        t_anchor_ns = int(t_anchor_ns)
        deadline_ns = time.monotonic_ns() + int(
            self.history_wait_timeout_ms * 1e6
        )
        while time.monotonic_ns() < deadline_ns:
            if self._histories_cover_anchor(t_anchor_ns):
                return True
            time.sleep(self.history_wait_sleep_s)

        return self._histories_cover_anchor(t_anchor_ns)

    def _histories_cover_anchor(self, t_anchor_ns):
        arm_time_ns = self.arm_controller.latest_action_time_ns()
        robot_time_ns = self.robot_state_reader.latest_obs_time_ns()
        hand_time_ns = self.hand_control_worker.latest_command_time_ns()
        return (
            self._covers_anchor(arm_time_ns, t_anchor_ns)
            and self._covers_anchor(robot_time_ns, t_anchor_ns)
            and self._covers_anchor(hand_time_ns, t_anchor_ns)
        )

    @staticmethod
    def _covers_anchor(latest_time_ns, t_anchor_ns):
        return latest_time_ns is not None and int(latest_time_ns) >= t_anchor_ns


class TeleopRuntime:
    def __init__(
        self,
        robot,
        camera,
        hand_manus,
        hand_controller,
        keyboard_control,
        arm_controller,
        hand_control_worker,
        robot_state_reader,
    ):
        self.robot = robot
        self.camera = camera
        self.hand_manus = hand_manus
        self.hand_controller = hand_controller
        self.keyboard_control = keyboard_control
        self.arm_controller = arm_controller
        self.hand_control_worker = hand_control_worker
        self.robot_state_reader = robot_state_reader
        self.control_workers_running = False
        self.closed = False

    def start_episode_workers(self):
        self.arm_controller.start()
        self.control_workers_running = True
        self.hand_control_worker.start()
        self.robot_state_reader.start()

    def stop_episode_workers(
        self,
        clear_recording=True,
        reset_hand=True,
        check_failures=True,
    ):
        if self.control_workers_running:
            self.hand_control_worker.stop()
            self.arm_controller.stop()
            self.robot_state_reader.stop()
            self.control_workers_running = False

        if clear_recording:
            self.keyboard_control.clear_recording()
        if reset_hand:
            print("Resetting Inspire hand after recording...\r")
            self.hand_controller.reset(settle_time=0.2)
        if check_failures:
            self.hand_control_worker.raise_if_failed()
            self.arm_controller.raise_if_failed()
            self.robot_state_reader.raise_if_failed()

    def close(self, stop_script=True):
        if self.closed:
            return
        if self.control_workers_running:
            self.stop_episode_workers(
                clear_recording=False,
                reset_hand=False,
                check_failures=False,
            )
        self.keyboard_control.stop()
        self.camera.finalize()
        self.hand_manus.finalize()
        self.hand_controller.close()
        self.robot.close(stop_script=stop_script)
        self.closed = True
