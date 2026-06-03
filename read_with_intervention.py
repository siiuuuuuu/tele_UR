import glob
import os

import cv2
import h5py
import numpy as np


TRUE_WORDS = {"1", "true", "t", "yes", "y", "success", "ok"}
FALSE_WORDS = {"0", "false", "f", "no", "n", "fail", "failed"}


def _extract_success_bool(success_attr):
    """将H5属性中的success解析为bool，无法判断时返回None。"""
    if success_attr is None:
        return None

    # h5属性可能是标量、数组、bytes 或 tuple(保存时末尾逗号可能导致)
    if isinstance(success_attr, np.ndarray):
        if success_attr.size == 0:
            return None
        success_attr = success_attr.reshape(-1)[0]
    elif isinstance(success_attr, (tuple, list)):
        if len(success_attr) == 0:
            return None
        success_attr = success_attr[0]

    if isinstance(success_attr, bytes):
        success_attr = success_attr.decode("utf-8", errors="ignore")

    if isinstance(success_attr, str):
        text = success_attr.strip().lower()
        if text in TRUE_WORDS:
            return True
        if text in FALSE_WORDS:
            return False
        return None

    try:
        return bool(success_attr)
    except Exception:
        return None


def _success_to_str(success_bool):
    if success_bool is True:
        return "True"
    if success_bool is False:
        return "False"
    return "Unknown"


def collect_folder_success_stats(h5_files):
    """统计文件夹内success为True/False/Unknown的文件数。"""
    success_count = 0
    fail_count = 0
    unknown_count = 0

    for file_path in h5_files:
        try:
            with h5py.File(file_path, "r") as f:
                success_bool = _extract_success_bool(f.attrs.get("success"))
            if success_bool is True:
                success_count += 1
            elif success_bool is False:
                fail_count += 1
            else:
                unknown_count += 1
        except Exception as exc:
            print(f"读取统计时出错 {os.path.basename(file_path)}: {exc}")
            unknown_count += 1

    return success_count, fail_count, unknown_count


def play_h5_video(file_path):
    """播放单个H5文件中的视频数据，并显示 intervention/autonomous 与 success 状态。"""
    try:
        with h5py.File(file_path, "r") as f:
            if "color" not in f:
                print(f"文件 {os.path.basename(file_path)} 中没有找到color数据，跳过")
                return False

            color_data = f["wrist_color"][:]

            if "intervention" in f:
                intervention_data = np.array(f["intervention"][:]).reshape(-1)
            else:
                intervention_data = np.zeros(len(color_data), dtype=np.uint8)
                print(f"文件 {os.path.basename(file_path)} 中没有找到intervention数据，默认按autonomous显示")

            if len(intervention_data) != len(color_data):
                print(
                    f"警告: intervention帧数({len(intervention_data)})与视频帧数({len(color_data)})不一致，"
                    "超出部分将按autonomous处理"
                )

            success_bool = _extract_success_bool(f.attrs.get("success"))
            success_str = _success_to_str(success_bool)

        print(f"\n正在播放: {os.path.basename(file_path)}")
        print(f"Success 标记: {success_str}")
        print(f"Color data shape: {color_data.shape}")
        print(f"Total frames: {color_data.shape[0]}")
        print(f"Frame dimensions: {color_data.shape[1]}x{color_data.shape[2]}")

        fps = 25
        frame_delay = int(1000 / fps)

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
                frame = color_data[frame_idx]

                if frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame

                intervention_flag = 0
                if frame_idx < len(intervention_data):
                    intervention_flag = int(intervention_data[frame_idx])

                if intervention_flag == 1:
                    mode_text = "INTERVENTION"
                    mode_color = (0, 0, 255)
                else:
                    mode_text = "AUTONOMOUS"
                    mode_color = (0, 255, 0)

                if success_bool is True:
                    success_text = "SUCCESS: true"
                    success_color = (0, 255, 0)
                elif success_bool is False:
                    success_text = "SUCCESS: false"
                    success_color = (0, 0, 255)
                else:
                    success_text = "SUCCESS: unknown"
                    success_color = (0, 255, 255)

                cv2.putText(
                    frame_bgr,
                    mode_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    mode_color,
                    2,
                )

                cv2.putText(
                    frame_bgr,
                    success_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    success_color,
                    2,
                )

                info_text = f"Frame: {frame_idx + 1}/{len(color_data)} | File: {os.path.basename(file_path)}"
                cv2.putText(
                    frame_bgr,
                    info_text,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("H5 Video Playback", frame_bgr)
                frame_idx += 1

            key = cv2.waitKey(frame_delay) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return "quit_all"
            if key == ord("n"):
                cv2.destroyAllWindows()
                return "next_file"
            if key == ord(" "):
                paused = not paused
                print(f"{'暂停' if paused else '继续播放'}")
            if key == ord("r"):
                frame_idx = 0
                paused = False
            if key == ord("f"):
                frame_idx = min(frame_idx + 10, len(color_data) - 1)
            if key == ord("b"):
                frame_idx = max(frame_idx - 10, 0)

        cv2.destroyAllWindows()
        return True

    except Exception as e:
        print(f"播放文件 {os.path.basename(file_path)} 时出错: {e}")
        return False


if __name__ == "__main__":
    data_folder = "~/dp_data/offlineRL_data/task2_iter1" 


    h5_files = sorted(glob.glob(os.path.join(data_folder, "demo_*.h5")))
    if not h5_files:
        print(f"在文件夹 {data_folder} 中没有找到.h5文件")
        raise SystemExit(0)

    print(f"找到 {len(h5_files)} 个.h5文件:")
    for i, file in enumerate(h5_files):
        print(f"  {i + 1}. {os.path.basename(file)}")

    success_count, fail_count, unknown_count = collect_folder_success_stats(h5_files)

    current_file_idx = 0
    while current_file_idx < len(h5_files):
        file_path = h5_files[current_file_idx]
        result = play_h5_video(file_path)

        if result == "quit_all":
            print("\n退出播放")
            break
        if result == "next_file":
            current_file_idx += 1
            if current_file_idx >= len(h5_files):
                print("\n已经播放完所有文件")
                break
        elif result:
            current_file_idx += 1
            if current_file_idx < len(h5_files):
                print("\n3秒后将自动播放下一个文件...")
                cv2.waitKey(3000)
            else:
                print("\n已经播放完所有文件")
        else:
            current_file_idx += 1
            if current_file_idx < len(h5_files):
                print("\n尝试播放下一个文件...")

    print("\n文件夹success统计:")
    print(f"  Success(True) 数量: {success_count}")
    print(f"  Failure(False) 数量: {fail_count}")
    if unknown_count > 0:
        print(f"  Unknown 数量: {unknown_count}")

    print("播放结束")
