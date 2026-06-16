"""Episode recording buffer for servoL teleoperation."""

import numpy as np


TIMESTAMP_INT_KEYS = (
    "t_record_start_ns",
    "t_record_end_ns",
    "t_arm_read_ns",
    "t_arm_action_host_ns",
    "t_arm_servo_host_ns",
    "t_arm_target_host_ns",
    "t_tracker0_host_ns",
    "t_tracker1_host_ns",
    "t_tracker_latest_host_ns",
    "t_robot_obs_host_ns",
    "t_camera_read_ns",
    "t_front_camera_host_ns",
    "front_camera_seq",
    "front_camera_frame_no",
    "t_wrist_camera_host_ns",
    "wrist_camera_seq",
    "wrist_camera_frame_no",
    "t_hand_read_ns",
    "t_hand_command_host_ns",
    "t_manus_sample_host_ns",
    "manus_seq",
)

TIMESTAMP_FLOAT_KEYS = (
    "interpolation_alpha",
    "front_camera_dev_ts",
    "wrist_camera_dev_ts",
)


class EpisodeBuffer:
    """Accumulates one teleoperation episode and writes it to HDF5."""

    def __init__(self, use_wrist_img=False):
        self.use_wrist_img = use_wrist_img
        self.robot_states = []
        self.front_images = []
        self.wrist_images = []
        self.actions = []
        self.timestamps = {
            key: [] for key in TIMESTAMP_INT_KEYS + TIMESTAMP_FLOAT_KEYS
        }

    def __len__(self):
        return len(self.actions)

    def append(self, robot_state, cam_dict, action, timestamp_dict=None):
        self.robot_states.append(np.asarray(robot_state))
        self.front_images.append(cam_dict["front_color"])
        if self.use_wrist_img:
            self.wrist_images.append(cam_dict["right_color"])
        self.actions.append(np.asarray(action))
        self._append_timestamps(timestamp_dict or {})

    def to_arrays(self):
        arrays = {
            "color": np.asarray(self.front_images),
            "env_qpos_proprioception": np.asarray(self.robot_states),
            "action": np.asarray(self.actions),
            "timestamps": {
                key: np.asarray(self.timestamps[key], dtype=np.int64)
                for key in TIMESTAMP_INT_KEYS
            },
        }
        arrays["timestamps"].update(
            {
                key: np.asarray(self.timestamps[key], dtype=np.float64)
                for key in TIMESTAMP_FLOAT_KEYS
            }
        )
        if self.use_wrist_img:
            arrays["wrist_color"] = np.asarray(self.wrist_images)
        return arrays

    def save_h5(self, record_file_name):
        import h5py

        arrays = self.to_arrays()
        with h5py.File(record_file_name, "w") as f:
            f.create_dataset("color", data=arrays["color"])
            if self.use_wrist_img:
                f.create_dataset("wrist_color", data=arrays["wrist_color"])
            f.create_dataset(
                "env_qpos_proprioception",
                data=arrays["env_qpos_proprioception"],
            )
            f.create_dataset("action", data=arrays["action"])
            timestamp_group = f.create_group("timestamps")
            for key, value in arrays["timestamps"].items():
                timestamp_group.create_dataset(key, data=value)

        return {
            "seq_length": len(self),
            "color_shape": arrays["color"].shape,
            "wrist_color_shape": arrays.get("wrist_color", None).shape
            if self.use_wrist_img
            else None,
            "action_shape": arrays["action"].shape,
            "env_qpos_shape": arrays["env_qpos_proprioception"].shape,
            "timestamp_keys": tuple(arrays["timestamps"].keys()),
            "record_file_name": record_file_name,
        }

    def _append_timestamps(self, timestamp_dict):
        for key in TIMESTAMP_INT_KEYS:
            value = timestamp_dict.get(key, -1)
            self.timestamps[key].append(-1 if value is None else int(value))
        for key in TIMESTAMP_FLOAT_KEYS:
            value = timestamp_dict.get(key, np.nan)
            self.timestamps[key].append(np.nan if value is None else float(value))
