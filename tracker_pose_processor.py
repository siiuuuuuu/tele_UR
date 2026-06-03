"""Tracker pose processing for UR teleoperation."""

import math

import numpy as np


def default_tracker_to_tcp_mat():
    transformation_matrix_1 = np.eye(4)
    theta = -math.pi / 2
    transformation_matrix_1[1, 1] = math.cos(theta)
    transformation_matrix_1[1, 2] = -math.sin(theta)
    transformation_matrix_1[2, 1] = math.sin(theta)
    transformation_matrix_1[2, 2] = math.cos(theta)

    transformation_matrix_2 = np.eye(4)
    theta_2 = math.pi
    transformation_matrix_2[0, 0] = math.cos(theta_2)
    transformation_matrix_2[0, 2] = math.sin(theta_2)
    transformation_matrix_2[2, 0] = -math.sin(theta_2)
    transformation_matrix_2[2, 2] = math.cos(theta_2)

    return np.dot(transformation_matrix_1, transformation_matrix_2)


class TrackerPoseProcessor:
    """Converts OpenVR tracker poses into UR target poses and arm actions."""

    def __init__(self, tool, init_tcp_pose, tracker_to_tcp_mat=None):
        self.tool = tool
        self.tracker_to_tcp_mat = (
            default_tracker_to_tcp_mat()
            if tracker_to_tcp_mat is None
            else np.asarray(tracker_to_tcp_mat)
        )
        self.init_tcp_mat = self._tcp_pose_to_mat(init_tcp_pose)
        self.init_tracker_mat = None

    def _tcp_pose_to_mat(self, tcp_pose):
        tcp_pose = np.asarray(tcp_pose)
        rot_mat = self.tool.rotvec2mat(tcp_pose[3:6])
        tcp_mat = np.eye(4)
        tcp_mat[:3, :3] = rot_mat
        tcp_mat[0, 3] = tcp_pose[0]
        tcp_mat[1, 3] = tcp_pose[1]
        tcp_mat[2, 3] = tcp_pose[2]
        return tcp_mat

    def read_tracker_mat(self, tracker_device):
        pose_mat = tracker_device.get_pose_matrix()
        if pose_mat is None:
            return None

        pose_np = np.array(
            [
                [pose_mat[0][0], pose_mat[0][1], pose_mat[0][2], pose_mat[0][3]],
                [pose_mat[1][0], pose_mat[1][1], pose_mat[1][2], pose_mat[1][3]],
                [pose_mat[2][0], pose_mat[2][1], pose_mat[2][2], pose_mat[2][3]],
            ]
        )
        tracker_mat = np.eye(4)
        tracker_mat[:3, :4] = pose_np
        return np.dot(tracker_mat, self.tracker_to_tcp_mat)

    def has_reference(self):
        return self.init_tracker_mat is not None

    def clear_reference(self):
        self.init_tracker_mat = None

    def reset_reference(self, tracker_mat):
        self.init_tracker_mat = np.asarray(tracker_mat).copy()

    def compute(self, tracker_mat):
        if self.init_tracker_mat is None:
            raise RuntimeError("Tracker reference is not initialized")

        increment_mat = np.dot(
            self.tool.se3_inverse(self.init_tracker_mat),
            tracker_mat,
        )
        result_matrix = np.dot(self.init_tcp_mat, increment_mat)
        return {
            "increment_matrix": increment_mat,
            "result_matrix": result_matrix,
            "arm_action": self.tool.mat2xyz_6drot(result_matrix),
            "target_pose": self.tool.mat2xyz_rotvec(result_matrix),
        }
