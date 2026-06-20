import argparse
import glob
import os
import time

import cv2
import h5py
import numpy as np


WINDOW_NAME = "H5 Dual-Camera Viewer"
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 820
HEADER_HEIGHT = 96
FOOTER_HEIGHT = 112
MARGIN = 20
PANEL_GAP = 16
DEFAULT_FPS = 25.0
DEFAULT_DATA_FOLDER = os.path.expanduser("~/dp_data/test_demo")

COLOR_BACKGROUND = (18, 20, 25)
COLOR_HEADER = (25, 28, 35)
COLOR_PANEL = (30, 34, 42)
COLOR_PANEL_BORDER = (58, 64, 76)
COLOR_TEXT = (238, 241, 246)
COLOR_MUTED = (157, 164, 178)
COLOR_ACCENT = (235, 164, 52)
COLOR_SUCCESS = (91, 201, 125)
COLOR_WARNING = (62, 190, 245)


def _draw_text(image, text, origin, scale, color, thickness=1):
    """Draw readable text with a subtle shadow."""
    x, y = origin
    cv2.putText(
        image,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _fit_text(text, max_width, scale, thickness=1):
    """Truncate text to fit the requested pixel width."""
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
        return text

    suffix = "..."
    while text:
        candidate = text + suffix
        width = cv2.getTextSize(
            candidate,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )[0][0]
        if width <= max_width:
            return candidate
        text = text[:-1]
    return suffix


def _to_bgr(frame):
    """Convert an H5 image frame to an OpenCV-compatible BGR uint8 image."""
    image = np.asarray(frame)

    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        image = np.nan_to_num(image)
        if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"unsupported frame shape: {image.shape}")
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"unsupported channel count: {image.shape[2]}")


def _place_fitted_image(canvas, image, x, y, width, height):
    """Place an image into a rectangle while preserving its aspect ratio."""
    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)
    target_width = max(1, int(round(image_width * scale)))
    target_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    image_x = x + (width - target_width) // 2
    image_y = y + (height - target_height) // 2
    canvas[image_y : image_y + target_height, image_x : image_x + target_width] = resized


def _draw_camera_panel(canvas, frame, rect, title, dataset_name):
    x, y, width, height = rect
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_PANEL_BORDER, 1)

    label_height = 48
    cv2.line(
        canvas,
        (x, y + label_height),
        (x + width, y + label_height),
        COLOR_PANEL_BORDER,
        1,
    )
    _draw_text(canvas, title, (x + 16, y + 31), 0.63, COLOR_TEXT, 2)
    _draw_text(canvas, dataset_name, (x + 16, y + height - 14), 0.48, COLOR_MUTED, 1)

    image_x = x + 12
    image_y = y + label_height + 12
    image_width = width - 24
    image_height = height - label_height - 42

    if frame is None:
        message = "CAMERA STREAM NOT AVAILABLE"
        message_width = cv2.getTextSize(
            message,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            2,
        )[0][0]
        _draw_text(
            canvas,
            message,
            (x + (width - message_width) // 2, image_y + image_height // 2),
            0.62,
            COLOR_WARNING,
            2,
        )
        return

    try:
        image = _to_bgr(frame)
        resolution = f"{image.shape[1]} x {image.shape[0]}"
        resolution_width = cv2.getTextSize(
            resolution,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            1,
        )[0][0]
        _draw_text(
            canvas,
            resolution,
            (x + width - resolution_width - 16, y + 30),
            0.48,
            COLOR_MUTED,
            1,
        )
        _place_fitted_image(canvas, image, image_x, image_y, image_width, image_height)
    except ValueError as exc:
        message = _fit_text(str(exc).upper(), image_width - 20, 0.52, 1)
        _draw_text(
            canvas,
            message,
            (image_x + 10, image_y + image_height // 2),
            0.52,
            COLOR_WARNING,
            1,
        )


def _format_time(seconds):
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def compose_dashboard(
    front_frame,
    wrist_frame,
    file_name,
    frame_idx,
    total_frames,
    fps,
    paused=False,
    warning=None,
):
    """Compose the complete dual-camera playback dashboard."""
    canvas = np.full(
        (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
        COLOR_BACKGROUND,
        dtype=np.uint8,
    )
    cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, HEADER_HEIGHT), COLOR_HEADER, -1)

    _draw_text(canvas, "DUAL CAMERA PLAYBACK", (MARGIN, 37), 0.86, COLOR_TEXT, 2)
    file_text = _fit_text(file_name, WINDOW_WIDTH - 330, 0.52, 1)
    _draw_text(canvas, file_text, (MARGIN, 69), 0.52, COLOR_MUTED, 1)
    if warning:
        warning_text = _fit_text(warning, WINDOW_WIDTH - 330, 0.45, 1)
        _draw_text(canvas, warning_text, (MARGIN, 89), 0.45, COLOR_WARNING, 1)

    status_text = "PAUSED" if paused else "PLAYING"
    status_color = COLOR_WARNING if paused else COLOR_SUCCESS
    badge_width = 130
    badge_height = 38
    badge_x = WINDOW_WIDTH - MARGIN - badge_width
    badge_y = 27
    cv2.rectangle(
        canvas,
        (badge_x, badge_y),
        (badge_x + badge_width, badge_y + badge_height),
        status_color,
        -1,
    )
    status_size = cv2.getTextSize(
        status_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        2,
    )[0]
    _draw_text(
        canvas,
        status_text,
        (
            badge_x + (badge_width - status_size[0]) // 2,
            badge_y + (badge_height + status_size[1]) // 2,
        ),
        0.56,
        COLOR_BACKGROUND,
        2,
    )

    panel_y = HEADER_HEIGHT + MARGIN
    panel_height = WINDOW_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT - 2 * MARGIN
    panel_width = (WINDOW_WIDTH - 2 * MARGIN - PANEL_GAP) // 2
    left_rect = (MARGIN, panel_y, panel_width, panel_height)
    right_rect = (MARGIN + panel_width + PANEL_GAP, panel_y, panel_width, panel_height)
    _draw_camera_panel(canvas, front_frame, left_rect, "FRONT CAMERA", "color")
    _draw_camera_panel(canvas, wrist_frame, right_rect, "WRIST CAMERA", "wrist_color")

    footer_y = WINDOW_HEIGHT - FOOTER_HEIGHT
    cv2.rectangle(canvas, (0, footer_y), (WINDOW_WIDTH, WINDOW_HEIGHT), COLOR_HEADER, -1)

    progress_x = MARGIN
    progress_y = footer_y + 18
    progress_width = WINDOW_WIDTH - 2 * MARGIN
    progress_height = 8
    cv2.rectangle(
        canvas,
        (progress_x, progress_y),
        (progress_x + progress_width, progress_y + progress_height),
        COLOR_PANEL_BORDER,
        -1,
    )
    progress = min(1.0, max(0.0, (frame_idx + 1) / max(1, total_frames)))
    cv2.rectangle(
        canvas,
        (progress_x, progress_y),
        (progress_x + int(progress_width * progress), progress_y + progress_height),
        COLOR_ACCENT,
        -1,
    )

    current_time = _format_time(frame_idx / fps)
    total_time = _format_time(total_frames / fps)
    playback_info = (
        f"Frame {frame_idx + 1:,} / {total_frames:,}    "
        f"{current_time} / {total_time}    {fps:g} FPS"
    )
    _draw_text(canvas, playback_info, (MARGIN, footer_y + 57), 0.56, COLOR_TEXT, 1)

    controls = "SPACE  Pause / Resume     F / B  +/- 10 frames     R  Restart     N  Next file     Q  Quit"
    _draw_text(canvas, controls, (MARGIN, footer_y + 91), 0.48, COLOR_MUTED, 1)
    return canvas


def play_h5_video(file_path, fps=DEFAULT_FPS):
    """Play synchronized color and wrist_color streams from one H5 file."""
    file_name = os.path.basename(file_path)
    try:
        with h5py.File(file_path, "r") as h5_file:
            if "color" not in h5_file:
                print(f"Skipping {file_name}: dataset 'color' was not found.")
                return False

            front_data = h5_file["color"]
            wrist_data = h5_file.get("wrist_color")
            front_frames = len(front_data)
            warning = None

            if front_frames == 0:
                print(f"Skipping {file_name}: dataset 'color' is empty.")
                return False

            if wrist_data is None or len(wrist_data) == 0:
                total_frames = front_frames
                wrist_data = None
                warning = "wrist_color is unavailable; displaying a placeholder."
            else:
                wrist_frames = len(wrist_data)
                total_frames = min(front_frames, wrist_frames)
                if front_frames != wrist_frames:
                    warning = (
                        f"Frame count mismatch: color={front_frames}, "
                        f"wrist_color={wrist_frames}; playing {total_frames} synchronized frames."
                    )

            print(f"\nPlaying: {file_name}")
            print(f"  color: {front_data.shape}")
            print(f"  wrist_color: {None if wrist_data is None else wrist_data.shape}")
            print(f"  synchronized frames: {total_frames}")
            print("  Controls: SPACE pause/resume, F/B seek, R restart, N next, Q quit")

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

            paused = False
            frame_idx = 0
            frame_delay_ms = max(1, int(round(1000.0 / fps)))

            while frame_idx < total_frames:
                loop_start = time.monotonic()
                front_frame = front_data[frame_idx]
                wrist_frame = None if wrist_data is None else wrist_data[frame_idx]
                dashboard = compose_dashboard(
                    front_frame=front_frame,
                    wrist_frame=wrist_frame,
                    file_name=file_name,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    fps=fps,
                    paused=paused,
                    warning=warning,
                )
                cv2.imshow(WINDOW_NAME, dashboard)

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    return "quit_all"

                render_ms = int((time.monotonic() - loop_start) * 1000)
                wait_ms = 30 if paused else max(1, frame_delay_ms - render_ms)
                key = cv2.waitKey(wait_ms) & 0xFF

                if key == ord("q"):
                    return "quit_all"
                if key == ord("n"):
                    return "next_file"
                if key == ord(" "):
                    paused = not paused
                    continue
                if key == ord("r"):
                    frame_idx = 0
                    paused = False
                    continue
                if key == ord("f"):
                    frame_idx = min(frame_idx + 10, total_frames - 1)
                    continue
                if key == ord("b"):
                    frame_idx = max(frame_idx - 10, 0)
                    continue
                if not paused:
                    frame_idx += 1

            return True

    except Exception as exc:
        print(f"Error while playing {file_name}: {exc}")
        return False
    finally:
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play synchronized color and wrist_color streams from H5 files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_DATA_FOLDER,
        help="H5 file or directory containing H5 files.",
    )
    parser.add_argument(
        "--pattern",
        default="demo_*.h5",
        help="Glob pattern used when path is a directory.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Playback frame rate.")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    return args


def collect_h5_files(path, pattern):
    expanded_path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(expanded_path):
        return [expanded_path]
    return sorted(glob.glob(os.path.join(expanded_path, pattern)))


def main():
    args = parse_args()
    h5_files = collect_h5_files(args.path, args.pattern)
    if not h5_files:
        print(f"No H5 files found at {os.path.expanduser(args.path)}")
        raise SystemExit(0)

    print(f"Found {len(h5_files)} H5 file(s):")
    for index, file_path in enumerate(h5_files, start=1):
        print(f"  {index}. {os.path.basename(file_path)}")

    for file_path in h5_files:
        result = play_h5_video(file_path, fps=args.fps)
        if result == "quit_all":
            print("\nPlayback stopped.")
            break

    print("Playback finished.")


if __name__ == "__main__":
    main()
