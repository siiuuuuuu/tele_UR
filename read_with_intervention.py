import argparse
import glob
import os
import time

import cv2
import h5py
import numpy as np

import read as dual_viewer


WINDOW_NAME = "H5 Intervention Viewer"
DEFAULT_FPS = 30.0
DEFAULT_DATA_FOLDER = os.path.expanduser("~/dp_data/offlineRL_data/test_task4_iter1")

TRUE_WORDS = {"1", "true", "t", "yes", "y", "success", "ok"}
FALSE_WORDS = {"0", "false", "f", "no", "n", "fail", "failed"}

COLOR_AUTONOMOUS = (82, 139, 91)
COLOR_INTERVENTION = (60, 74, 211)
COLOR_SUCCESS = (91, 201, 125)
COLOR_FAILURE = (70, 78, 220)
COLOR_UNKNOWN = (62, 190, 245)
COLOR_PLAYHEAD = (250, 250, 250)


def _extract_success_bool(success_attr):
    """Parse an H5 success attribute into bool, returning None if unknown."""
    if success_attr is None:
        return None

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


def _success_badge(success_bool):
    if success_bool is True:
        return "EPISODE SUCCESS", COLOR_SUCCESS
    if success_bool is False:
        return "EPISODE FAILURE", COLOR_FAILURE
    return "SUCCESS UNKNOWN", COLOR_UNKNOWN


def collect_folder_success_stats(h5_files):
    """Count successful, failed, and unknown episodes."""
    success_count = 0
    fail_count = 0
    unknown_count = 0

    for file_path in h5_files:
        try:
            with h5py.File(file_path, "r") as h5_file:
                success_bool = _extract_success_bool(h5_file.attrs.get("success"))
            if success_bool is True:
                success_count += 1
            elif success_bool is False:
                fail_count += 1
            else:
                unknown_count += 1
        except Exception as exc:
            print(f"Error reading success status from {os.path.basename(file_path)}: {exc}")
            unknown_count += 1

    return success_count, fail_count, unknown_count


def _draw_badge(canvas, text, right_x, top_y, color, width):
    height = 38
    left_x = right_x - width
    cv2.rectangle(canvas, (left_x, top_y), (right_x, top_y + height), color, -1)
    text_size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        2,
    )[0]
    dual_viewer._draw_text(
        canvas,
        text,
        (
            left_x + (width - text_size[0]) // 2,
            top_y + (height + text_size[1]) // 2,
        ),
        0.52,
        dual_viewer.COLOR_BACKGROUND,
        2,
    )
    return left_x


def _draw_mode_border(canvas, mode_color):
    panel_y = dual_viewer.HEADER_HEIGHT + dual_viewer.MARGIN
    panel_height = (
        dual_viewer.WINDOW_HEIGHT
        - dual_viewer.HEADER_HEIGHT
        - dual_viewer.FOOTER_HEIGHT
        - 2 * dual_viewer.MARGIN
    )
    panel_width = (
        dual_viewer.WINDOW_WIDTH
        - 2 * dual_viewer.MARGIN
        - dual_viewer.PANEL_GAP
    ) // 2
    panel_rects = (
        (dual_viewer.MARGIN, panel_y, panel_width, panel_height),
        (
            dual_viewer.MARGIN + panel_width + dual_viewer.PANEL_GAP,
            panel_y,
            panel_width,
            panel_height,
        ),
    )

    for x, y, width, height in panel_rects:
        cv2.rectangle(
            canvas,
            (x, y),
            (x + width, y + height),
            mode_color,
            6,
            cv2.LINE_AA,
        )


def _draw_intervention_timeline(canvas, intervention_data, frame_idx):
    footer_y = dual_viewer.WINDOW_HEIGHT - dual_viewer.FOOTER_HEIGHT
    timeline_x = dual_viewer.MARGIN
    timeline_y = footer_y + 16
    timeline_width = dual_viewer.WINDOW_WIDTH - 2 * dual_viewer.MARGIN
    timeline_height = 12

    intervention_data = np.asarray(intervention_data, dtype=bool).reshape(-1)
    if len(intervention_data) == 0:
        intervention_data = np.zeros(1, dtype=bool)

    timeline = np.empty((1, len(intervention_data), 3), dtype=np.uint8)
    timeline[0, ~intervention_data] = COLOR_AUTONOMOUS
    timeline[0, intervention_data] = COLOR_INTERVENTION
    timeline = cv2.resize(
        timeline,
        (timeline_width, timeline_height),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas[
        timeline_y : timeline_y + timeline_height,
        timeline_x : timeline_x + timeline_width,
    ] = timeline
    cv2.rectangle(
        canvas,
        (timeline_x, timeline_y),
        (timeline_x + timeline_width, timeline_y + timeline_height),
        dual_viewer.COLOR_PANEL_BORDER,
        1,
    )

    progress = min(1.0, max(0.0, (frame_idx + 1) / len(intervention_data)))
    playhead_x = timeline_x + int(progress * timeline_width)
    cv2.line(
        canvas,
        (playhead_x, timeline_y - 4),
        (playhead_x, timeline_y + timeline_height + 4),
        COLOR_PLAYHEAD,
        2,
        cv2.LINE_AA,
    )


def compose_intervention_dashboard(
    front_frame,
    wrist_frame,
    file_name,
    frame_idx,
    total_frames,
    fps,
    intervention_data,
    success_bool,
    paused=False,
    warning=None,
):
    """Compose a synchronized dual-camera intervention playback dashboard."""
    current_intervention = bool(intervention_data[frame_idx])
    mode_text = "INTERVENTION" if current_intervention else "AUTONOMOUS"
    mode_color = COLOR_INTERVENTION if current_intervention else COLOR_AUTONOMOUS

    canvas = np.full(
        (dual_viewer.WINDOW_HEIGHT, dual_viewer.WINDOW_WIDTH, 3),
        dual_viewer.COLOR_BACKGROUND,
        dtype=np.uint8,
    )
    cv2.rectangle(
        canvas,
        (0, 0),
        (dual_viewer.WINDOW_WIDTH, dual_viewer.HEADER_HEIGHT),
        dual_viewer.COLOR_HEADER,
        -1,
    )

    dual_viewer._draw_text(
        canvas,
        "INTERVENTION PLAYBACK",
        (dual_viewer.MARGIN, 37),
        0.86,
        dual_viewer.COLOR_TEXT,
        2,
    )
    file_text = dual_viewer._fit_text(
        file_name,
        dual_viewer.WINDOW_WIDTH - 620,
        0.52,
        1,
    )
    dual_viewer._draw_text(
        canvas,
        file_text,
        (dual_viewer.MARGIN, 69),
        0.52,
        dual_viewer.COLOR_MUTED,
        1,
    )
    if warning:
        warning_text = dual_viewer._fit_text(
            warning,
            dual_viewer.WINDOW_WIDTH - 620,
            0.45,
            1,
        )
        dual_viewer._draw_text(
            canvas,
            warning_text,
            (dual_viewer.MARGIN, 89),
            0.45,
            dual_viewer.COLOR_WARNING,
            1,
        )

    right_x = dual_viewer.WINDOW_WIDTH - dual_viewer.MARGIN
    success_text, success_color = _success_badge(success_bool)
    right_x = _draw_badge(canvas, success_text, right_x, 27, success_color, 205) - 12
    playback_text = "PAUSED" if paused else "PLAYING"
    playback_color = dual_viewer.COLOR_WARNING if paused else dual_viewer.COLOR_SUCCESS
    right_x = _draw_badge(canvas, playback_text, right_x, 27, playback_color, 125) - 12
    mode_width = 190 if current_intervention else 175
    _draw_badge(canvas, mode_text, right_x, 27, mode_color, mode_width)

    panel_y = dual_viewer.HEADER_HEIGHT + dual_viewer.MARGIN
    panel_height = (
        dual_viewer.WINDOW_HEIGHT
        - dual_viewer.HEADER_HEIGHT
        - dual_viewer.FOOTER_HEIGHT
        - 2 * dual_viewer.MARGIN
    )
    panel_width = (
        dual_viewer.WINDOW_WIDTH
        - 2 * dual_viewer.MARGIN
        - dual_viewer.PANEL_GAP
    ) // 2
    left_rect = (dual_viewer.MARGIN, panel_y, panel_width, panel_height)
    right_rect = (
        dual_viewer.MARGIN + panel_width + dual_viewer.PANEL_GAP,
        panel_y,
        panel_width,
        panel_height,
    )
    dual_viewer._draw_camera_panel(
        canvas,
        front_frame,
        left_rect,
        "FRONT CAMERA",
        "color",
    )
    dual_viewer._draw_camera_panel(
        canvas,
        wrist_frame,
        right_rect,
        "WRIST CAMERA",
        "wrist_color",
    )
    _draw_mode_border(canvas, mode_color)

    footer_y = dual_viewer.WINDOW_HEIGHT - dual_viewer.FOOTER_HEIGHT
    cv2.rectangle(
        canvas,
        (0, footer_y),
        (dual_viewer.WINDOW_WIDTH, dual_viewer.WINDOW_HEIGHT),
        dual_viewer.COLOR_HEADER,
        -1,
    )
    _draw_intervention_timeline(canvas, intervention_data, frame_idx)

    intervention_steps = int(np.count_nonzero(intervention_data))
    autonomous_steps = total_frames - intervention_steps
    current_time = dual_viewer._format_time(frame_idx / fps)
    total_time = dual_viewer._format_time(total_frames / fps)
    playback_info = (
        f"Frame {frame_idx + 1:,} / {total_frames:,}    "
        f"{current_time} / {total_time}    {fps:g} FPS    "
        f"Autonomous {autonomous_steps:,}    Intervention {intervention_steps:,}"
    )
    dual_viewer._draw_text(
        canvas,
        playback_info,
        (dual_viewer.MARGIN, footer_y + 58),
        0.53,
        dual_viewer.COLOR_TEXT,
        1,
    )
    controls = (
        "Green  Autonomous     Red  Intervention     SPACE  Pause/Resume     "
        "F/B  +/-10 frames     R  Restart     P  Previous eps     N  Next eps     Q  Quit"
    )
    dual_viewer._draw_text(
        canvas,
        controls,
        (dual_viewer.MARGIN, footer_y + 91),
        0.43,
        dual_viewer.COLOR_MUTED,
        1,
    )
    return canvas


def play_h5_video(file_path, fps=DEFAULT_FPS):
    """Play synchronized camera streams with intervention and success overlays."""
    file_name = os.path.basename(file_path)
    try:
        with h5py.File(file_path, "r") as h5_file:
            if "color" not in h5_file:
                print(f"Skipping {file_name}: dataset 'color' was not found.")
                return False

            front_data = h5_file["color"]
            wrist_data = h5_file.get("wrist_color")
            front_frames = len(front_data)
            warnings = []

            if front_frames == 0:
                print(f"Skipping {file_name}: dataset 'color' is empty.")
                return False

            total_frames = front_frames
            if wrist_data is None or len(wrist_data) == 0:
                wrist_data = None
                warnings.append("wrist_color is unavailable")
            else:
                total_frames = min(total_frames, len(wrist_data))
                if front_frames != len(wrist_data):
                    warnings.append(
                        f"camera frame mismatch: color={front_frames}, wrist_color={len(wrist_data)}"
                    )

            if "intervention" in h5_file:
                intervention_data = np.asarray(
                    h5_file["intervention"][:],
                    dtype=bool,
                ).reshape(-1)
                if len(intervention_data) != total_frames:
                    warnings.append(
                        f"intervention length={len(intervention_data)}, video length={total_frames}"
                    )
                    if len(intervention_data) < total_frames:
                        intervention_data = np.pad(
                            intervention_data,
                            (0, total_frames - len(intervention_data)),
                            constant_values=False,
                        )
                    else:
                        intervention_data = intervention_data[:total_frames]
            else:
                intervention_data = np.zeros(total_frames, dtype=bool)
                warnings.append("intervention is unavailable; treating all frames as autonomous")

            success_bool = _extract_success_bool(h5_file.attrs.get("success"))
            warning = "; ".join(warnings) if warnings else None

            print(f"\nPlaying: {file_name}")
            print(f"  success: {_success_badge(success_bool)[0]}")
            print(f"  color: {front_data.shape}")
            print(f"  wrist_color: {None if wrist_data is None else wrist_data.shape}")
            print(f"  synchronized frames: {total_frames}")
            print(
                f"  autonomous/intervention: "
                f"{total_frames - int(np.count_nonzero(intervention_data))}/"
                f"{int(np.count_nonzero(intervention_data))}"
            )
            print("  Controls: SPACE pause/resume, F/B seek, R restart, P previous eps, N next eps, Q quit")

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                WINDOW_NAME,
                dual_viewer.WINDOW_WIDTH,
                dual_viewer.WINDOW_HEIGHT,
            )

            paused = False
            frame_idx = 0
            frame_delay_ms = max(1, int(round(1000.0 / fps)))

            while frame_idx < total_frames:
                loop_start = time.monotonic()
                front_frame = front_data[frame_idx]
                wrist_frame = None if wrist_data is None else wrist_data[frame_idx]
                dashboard = compose_intervention_dashboard(
                    front_frame=front_frame,
                    wrist_frame=wrist_frame,
                    file_name=file_name,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    fps=fps,
                    intervention_data=intervention_data,
                    success_bool=success_bool,
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
                if key == ord("p"):
                    return "previous_file"
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
        description=(
            "Play synchronized color and wrist_color streams with intervention "
            "and episode success overlays."
        )
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

    success_count, fail_count, unknown_count = collect_folder_success_stats(h5_files)

    file_index = 0
    while file_index < len(h5_files):
        file_path = h5_files[file_index]
        result = play_h5_video(file_path, fps=args.fps)
        if result == "quit_all":
            print("\nPlayback stopped.")
            break
        if result == "previous_file":
            file_index = max(file_index - 1, 0)
            continue
        file_index += 1

    print("\nFolder success summary:")
    print(f"  Success: {success_count}")
    print(f"  Failure: {fail_count}")
    if unknown_count:
        print(f"  Unknown: {unknown_count}")
    print("Playback finished.")


if __name__ == "__main__":
    main()
