"""Episode recording buffer for servoL teleoperation."""

import numpy as np


class EpisodeBuffer:
    """Accumulates one teleoperation episode and writes it to HDF5."""

    def __init__(self, use_wrist_img=False):
        self.use_wrist_img = use_wrist_img
        self.robot_states = []
        self.front_images = []
        self.wrist_images = []
        self.actions = []

    def __len__(self):
        return len(self.actions)

    def append(self, robot_state, cam_dict, action):
        self.robot_states.append(np.asarray(robot_state))
        self.front_images.append(cam_dict["front_color"])
        if self.use_wrist_img:
            self.wrist_images.append(cam_dict["right_color"])
        self.actions.append(np.asarray(action))

    def to_arrays(self):
        arrays = {
            "color": np.asarray(self.front_images),
            "env_qpos_proprioception": np.asarray(self.robot_states),
            "action": np.asarray(self.actions),
        }
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

        return {
            "seq_length": len(self),
            "color_shape": arrays["color"].shape,
            "wrist_color_shape": arrays.get("wrist_color", None).shape
            if self.use_wrist_img
            else None,
            "action_shape": arrays["action"].shape,
            "env_qpos_shape": arrays["env_qpos_proprioception"].shape,
            "record_file_name": record_file_name,
        }
