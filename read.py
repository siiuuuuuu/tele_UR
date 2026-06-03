import h5py
import numpy as np
import cv2
import os
import glob

def play_h5_video(file_path):
    """播放单个H5文件中的视频数据"""
    try:
        with h5py.File(file_path, 'r') as f:
            if 'color' not in f:
                print(f"文件 {os.path.basename(file_path)} 中没有找到color数据，跳过")
                return False
            #color_data = f['color'][:]
            color_data = f['wrist_color'][:]
            #env_qpos_data = f['env_qpos_proprioception'][:]
            #action_data = f['action'][:]
        
        print(f"\n正在播放: {os.path.basename(file_path)}")
        print(f"Color data shape: {color_data.shape}")
        print(f"Total frames: {color_data.shape[0]}")
        print(f"Frame dimensions: {color_data.shape[1]}x{color_data.shape[2]}")
        
        # 视频播放参数
        fps = 25  # 播放帧率
        frame_delay = int(1000 / fps)  # 每帧延迟时间（毫秒）
        
        print("播放控制:")
        print("  按 'q' 键退出播放")
        print("  按 'n' 键播放下一个文件")
        print("  按空格键暂停/继续")
        print("  按 'r' 键重新开始当前文件")
        print("  按 'f' 键快进")
        print("  按 'b' 键快退")
        
        paused = False
        frame_idx = 0
        
        while frame_idx < len(color_data):
            if not paused:
                # 获取当前帧
                frame = color_data[frame_idx]
                
                # 转换颜色格式（如果需要）
                # 如果数据是RGB格式，OpenCV需要BGR格式
                if frame.shape[2] == 3:  # 确保是3通道
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                # 添加帧信息
                info_text = f"Frame: {frame_idx + 1}/{len(color_data)} | File: {os.path.basename(file_path)}"
                cv2.putText(frame_bgr, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow('H5 Video Playback', frame_bgr)
                
                frame_idx += 1
            
            # 等待按键
            key = cv2.waitKey(frame_delay) & 0xFF
            
            if key == ord('q'):  # 按q退出所有播放
                cv2.destroyAllWindows()
                return 'quit_all'
            elif key == ord('n'):  # 按n播放下一个文件
                cv2.destroyAllWindows()
                return 'next_file'
            elif key == ord(' '):  # 按空格暂停/继续
                paused = not paused
                print(f"{'暂停' if paused else '继续播放'}")
            elif key == ord('r'):  # 按r重新开始
                frame_idx = 0
                paused = False
            elif key == ord('f'):  # 按f快进
                frame_idx = min(frame_idx + 10, len(color_data) - 1)
            elif key == ord('b'):  # 按b快退
                frame_idx = max(frame_idx - 10, 0)
        
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"播放文件 {os.path.basename(file_path)} 时出错: {e}")
        return False

#,demo_20260422_165710.h5 demo_20260422_170146.h5
if __name__ == '__main__':
    # 设置要遍历的文件夹路径
    #data_folder = '/home/lrz/dp_data/offlineRL_data/task3_iter2' 
    data_folder = '/home/lrz/dp_data/new_task1_expertdata'
    # 获取所有.h5文件
    #h5_files = glob.glob(os.path.join(data_folder, '*.h5'))
    h5_files = glob.glob(os.path.join(data_folder, 'demo_*.h5'))
    if not h5_files:
        print(f"在文件夹 {data_folder} 中没有找到.h5文件")
        exit()
    
    print(f"找到 {len(h5_files)} 个.h5文件:")
    for i, file in enumerate(h5_files):
        print(f"  {i+1}. {os.path.basename(file)}")
    
    # 依次播放每个文件
    current_file_idx = 0
    while current_file_idx < len(h5_files):
        file_path = h5_files[current_file_idx]
        result = play_h5_video(file_path)
        
        if result == 'quit_all':
            print("\n退出播放")
            break
        elif result == 'next_file':
            current_file_idx += 1
            if current_file_idx >= len(h5_files):
                print("\n已经播放完所有文件")
                break
        elif result:
            # 当前文件播放完成，自动播放下一个
            current_file_idx += 1
            if current_file_idx < len(h5_files):
                print(f"\n3秒后将自动播放下一个文件...")
                cv2.waitKey(3000)  # 等待3秒
            else:
                print("\n已经播放完所有文件")
        else:
            # 播放出错，尝试下一个文件
            current_file_idx += 1
            if current_file_idx < len(h5_files):
                print(f"\n尝试播放下一个文件...")
    
    print("播放结束")