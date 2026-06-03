#!/usr/bin/env python3
"""
UR5 Robot RTDE Control Program
Uses OpenVR tracker to control UR5 robot motion
Modified version: Uses RTDE interface with servoL for smooth teleoperation
"""

import time
import sys
import numpy as np
import triad_openvr
from episode_buffer import EpisodeBuffer
from keyboard_control import KeyboardControl
from SM_multi_realsense import MultiRealSense
from tools import MATHTOOLS
import os
from termcolor import cprint
import argparse
from datetime import datetime
from SM_inspire_manus_process import inspire_Manus
from teleop_interfaces import InspireHandController, URArmInterface
from tracker_pose_processor import TrackerPoseProcessor
# RTDE connection configuration
#UR_HOST = "192.168.1.178"  # UR5 robot IP address
UR_HOST = "192.168.3.6" #210

def main(args):
    # Workspace safety limits (meters)
    workspace_limits = {
        'x': [-1, 1.],
        'y': [-1, 1], 
        'z': [0, 1]
    }
    dt = 1.0/25               # 控制频率 (25Hz)
    tool=MATHTOOLS()
    max_length=1600
    # Initialize RTDE connection
    try:
        robot=URArmInterface(
            UR_HOST,
            workspace_limits,
            servo_speed=0.005,
            servo_acceleration=0.005,
            servo_dt=dt/2,
            lookahead_time=0.2,
            gain=500,
        )
    except Exception:
        print("Cannot connect to UR5, exiting")
        sys.exit(1)
    #init data saving  
    data_dir = args.demo_dir
    #demo_name = args.demo_name
    use_wrist_img=args.use_wrist_img
    os.makedirs(data_dir, exist_ok=True)
    #record_file_name = os.path.join(data_dir, demo_name+".h5")

    #initialize inspire_Manus
    hand_Manus=inspire_Manus(use_right_hand=True, use_left_hand=False)
    hand_controller=InspireHandController("/dev/ttyUSB0", 115200)

    # initialize realsense
    cam_context=MultiRealSense(use_right_cam=use_wrist_img, front_num_points=20000, 
                         use_grid_sampling=True, use_crop=False,img_size=256)

    hand_Manus.start()#开启manus进程
    cam_context.start()#开启相机进程

    current_pose = robot.get_tcp_pose()
    print(current_pose)
    initial_pose =[0.248,0.1212,0.3978,1.16,1.25,1.28]
    #initial_pose =[0.248,0.0812,0.3978,1.16,1.25,1.28]
    # initial_pose = current_pose
    print("Moving to initial position")
    robot.move_l(initial_pose, 0.3, 0.3)
    tracker_processor = TrackerPoseProcessor(tool, initial_pose)

    # 初始化 OpenVR
    print("Initializing OpenVR")
    v = triad_openvr.triad_openvr()
    v.print_discovered_objects()

    print("\n" + "="*50)
    print("请按下 'a' 键 结束数据采集循环...")
    print("按下 's' 键开始录制，再次按下 's' 键结束录制")
    print("="*50 + "\n")
    #time.sleep(1)
    keyboard_control = KeyboardControl()
    keyboard_control.start()#开启监听线程
    time.sleep(0.1)

    try:
        while not keyboard_control.should_stop_collection():

            # 等待按键或超时，等环境重置和人手对齐就绪
            keyboard_control.wait_recording()
            episode = EpisodeBuffer(use_wrist_img=use_wrist_img)
            tracker_processor.clear_reference()
            #time.sleep(1)

            print(f"Control frequency set to: {1/dt:.2f} Hz\r")
            print("Starting recording tele-operation...\r")
            print("录制已开始，按下 's' 键结束录制\r")    
            step = 0
            while step < max_length and keyboard_control.is_recording():#遥操作主循环，按s键结束
                start_time = time.time()
                # 检查机器人状态
                if robot.is_ready():
                    current_tracker_mat = tracker_processor.read_tracker_mat(v.devices["tracker_1"])
                    if current_tracker_mat is None:
                        time.sleep(0.01)
                        continue
                    if not tracker_processor.has_reference():
                        tracker_processor.reset_reference(current_tracker_mat)
                        print("Initial tracking position recorded. Starting servo control...")
                        current_pose = robot.get_tcp_pose()
                        robot.servo(current_pose)
                        continue
                    robot_obs=robot.get_obs()#获取当前状态
                    if robot_obs is None:
                        time.sleep(0.01)
                        continue
                    #用于给相对位姿计算为xyz+rotvec 6维 robot state格式前6维为关节位姿，后6维为TCP位姿
                    cam_dict=cam_context()#回调获取当前最新的观察
                    motion = tracker_processor.compute(current_tracker_mat)
                    #相对位姿即该观测下机器人应做出的动作
                    hand_action_dict=hand_Manus()#回调获取最新手套动作
                    hand_action_raw=np.asarray(hand_action_dict['right'], dtype=np.float32)
                    hand_command=hand_controller.apply(hand_action_raw)
                    hand_action_array=hand_command.astype(np.float32) / 1000.0
                    arm_action=motion["arm_action"]#转换为xyz+6drot形式
                    action=np.concatenate((arm_action,hand_action_array))
                    episode.append(robot_obs["state"], cam_dict, action)
                    target_pose = motion["target_pose"]
                    step += 1
                
                    # 发送servoL指令；越界时在URArmInterface内裁剪xyz
                    robot.servo(target_pose)
                else:
                    print("Robot is stopped (protective or emergency).")
                    break # 退出循环

                # 控制循环频率
                elapsed = time.time() - start_time
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            robot.stop_servo() # 停止伺服控制线程
            #保存该episode的数据
            if len(episode)>0:
                user_choice = input("是否保存录制数据？(y/n): ").lower().strip()
                if user_choice=='y':
                    record_file_name = os.path.join(data_dir, datetime.now().strftime("demo_%Y%m%d_%H%M%S")+".h5")
                    print("Data recording")
                    summary = episode.save_h5(record_file_name)    
                    print("Data recording done.")
                    cprint(f"color shape: {summary['color_shape']}", "yellow")
                    if use_wrist_img:
                        cprint(f"wrist_color shape: {summary['wrist_color_shape']}", "yellow")
                    cprint(f"action shape: {summary['action_shape']}", "yellow")
                    cprint(f"env_qpos shape: {summary['env_qpos_shape']}", "yellow")
                    cprint(f"save data at step: {summary['seq_length']} in {summary['record_file_name']}", "yellow")
                else:
                    print("不保存数据")
            else:
                print("无数据，不保存")

            #UR返回初始位置进行reset（阻塞），中间可以确定是否结束采集，而manus不需要reset
            robot.move_l(initial_pose, 0.3, 0.3)

            #结束一条epsisode后看是否继续采集
            if keyboard_control.should_stop_collection():
                break

        print("collecting completed.\r")
        keyboard_control.stop()
        cam_context.finalize()
        hand_Manus.finalize()
        hand_controller.close()
        print("Stopping servo control and disconnecting.\r")
        robot.close(stop_script=False)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping control.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping servo control and disconnecting.")
        keyboard_control.stop()#停止监听线程
        #意外停止仍然正确关闭
        cam_context.finalize()#关闭相机进程
        hand_Manus.finalize()#关闭manus进程
        hand_controller.close()
        robot.close(stop_script=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", type=str, default=os.path.expanduser("~/dp_data/new_task1_expertdata"))
    parser.add_argument("--demo_name", type=str, default=datetime.now().strftime("demo_%Y%m%d_%H%M%S"))
    parser.add_argument("--use_wrist_img", type=bool, default=True)
    args = parser.parse_args()
    main(args)
