#!/usr/bin/env python3
"""Estimate camera content latency by filming a flashing screen.

This is a quick, no-extra-hardware check for whether there is a large
content-time offset like the t_camera term reported in some VLA systems.

What it measures:
    display_toggle_app_time -> image content change in RealSense frames

Because a monitor has its own scanout/pixel-response delay, the measured
content offset includes display latency. It is still useful for detecting
large 50 ms-class camera/content offsets before building an LED rig.
"""

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import statistics
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs


WINDOW_NAME = "RealSense screen flash latency target"
ROI_NAMES = ("center", "top", "middle", "bottom")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive float")
    return value


def nonnegative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative float")
    return value


def enum_name(value):
    text = str(value)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def is_global_domain(domain_name):
    text = str(domain_name).lower()
    return "global" in text or "system" in text


def percentile(values, pct):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(pct) / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(name, values, unit="ms"):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        print(f"{name}: no samples")
        return
    print(f"{name} ({unit}):")
    print(f"  count: {len(values)}")
    print(f"  min:   {min(values):.3f}")
    print(f"  p05:   {percentile(values, 5):.3f}")
    print(f"  p50:   {percentile(values, 50):.3f}")
    print(f"  mean:  {statistics.fmean(values):.3f}")
    print(f"  p95:   {percentile(values, 95):.3f}")
    print(f"  max:   {max(values):.3f}")
    if len(values) > 1:
        print(f"  std:   {statistics.stdev(values):.3f}")


def list_devices():
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        devices.append((serial, name))
    devices.sort()
    return devices


def maybe_get_metadata(frame, metadata_name):
    metadata = getattr(rs.frame_metadata_value, metadata_name, None)
    if metadata is None:
        return None
    try:
        if frame.supports_frame_metadata(metadata):
            return frame.get_frame_metadata(metadata)
    except Exception:
        return None
    return None


def set_global_time(profile, enabled=True):
    changed = []
    for sensor in profile.get_device().query_sensors():
        try:
            name = sensor.get_info(rs.camera_info.name)
        except Exception:
            name = "unknown sensor"
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1.0 if enabled else 0.0)
            changed.append((name, True))
        else:
            changed.append((name, False))
    return changed


def configure_color_sensor(profile, args):
    for sensor in profile.get_device().query_sensors():
        try:
            name = sensor.get_info(rs.camera_info.name)
        except Exception:
            name = ""
        if "rgb" not in name.lower() and "color" not in name.lower():
            continue
        if args.manual_exposure_us is not None:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            if sensor.supports(rs.option.exposure):
                sensor.set_option(rs.option.exposure, float(args.manual_exposure_us))
        if args.disable_auto_white_balance and sensor.supports(
            rs.option.enable_auto_white_balance
        ):
            sensor.set_option(rs.option.enable_auto_white_balance, 0.0)


def start_pipeline(serial, args):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.rgb8,
        args.fps,
    )
    profile = pipeline.start(config)
    for sensor_name, supported in set_global_time(profile, enabled=True):
        status = "enabled" if supported else "not supported"
        print(f"global_time_enabled on {sensor_name}: {status}")
    configure_color_sensor(profile, args)
    return pipeline


def frame_brightness(image, roi_fraction):
    h, w = image.shape[:2]
    frac = max(0.05, min(1.0, float(roi_fraction)))
    roi_w = max(1, int(w * frac))
    roi_h = max(1, int(h * frac))
    x0 = (w - roi_w) // 2
    y0 = (h - roi_h) // 2
    crop = image[y0 : y0 + roi_h, x0 : x0 + roi_w]
    return float(np.mean(crop))


def mean_roi(image, x_center_frac, y_center_frac, width_frac, height_frac):
    h, w = image.shape[:2]
    roi_w = max(1, int(w * max(0.05, min(1.0, float(width_frac)))))
    roi_h = max(1, int(h * max(0.03, min(1.0, float(height_frac)))))
    cx = int(w * float(x_center_frac))
    cy = int(h * float(y_center_frac))
    x0 = min(max(0, cx - roi_w // 2), max(0, w - roi_w))
    y0 = min(max(0, cy - roi_h // 2), max(0, h - roi_h))
    crop = image[y0 : y0 + roi_h, x0 : x0 + roi_w]
    return float(np.mean(crop))


def frame_brightnesses(image, args):
    center = frame_brightness(image, args.roi_fraction)
    width_frac = args.roi_fraction
    band_frac = args.roi_band_fraction
    return {
        "center": center,
        "top": mean_roi(image, 0.5, 0.2, width_frac, band_frac),
        "middle": mean_roi(image, 0.5, 0.5, width_frac, band_frac),
        "bottom": mean_roi(image, 0.5, 0.8, width_frac, band_frac),
    }


def camera_worker(records, errors, ready_event, run_event, stop_event, serial, args):
    pipeline = None
    try:
        pipeline = start_pipeline(serial, args)
        warmup_end = time.monotonic() + args.camera_warmup_s
        while time.monotonic() < warmup_end and not stop_event.is_set():
            try:
                pipeline.wait_for_frames(args.timeout_ms)
            except Exception:
                pass

        ready_event.set()
        if not run_event.wait(timeout=30.0):
            return

        capture_end = time.monotonic() + args.duration_s + args.post_roll_s
        while not stop_event.is_set() and time.monotonic() < capture_end:
            wait_start_mono_ns = time.monotonic_ns()
            frames = pipeline.wait_for_frames(args.timeout_ms)
            wait_return_epoch_ns = time.time_ns()
            wait_return_mono_ns = time.monotonic_ns()
            color = frames.get_color_frame()
            if not color:
                continue

            image = np.asanyarray(color.get_data(), dtype=np.uint8)
            brightnesses = frame_brightnesses(image, args)
            frame_ts_ms = float(color.get_timestamp())
            domain = enum_name(color.get_frame_timestamp_domain())
            epoch_minus_mono_ns = wait_return_epoch_ns - wait_return_mono_ns
            frame_global_mono_ns = (
                int(round(frame_ts_ms * 1e6)) - epoch_minus_mono_ns
                if is_global_domain(domain)
                else None
            )

            records.append(
                {
                    "index": len(records),
                    "frame_number": int(color.get_frame_number()),
                    "timestamp_domain": domain,
                    "frame_timestamp_ms": frame_ts_ms,
                    "frame_global_mono_ns": frame_global_mono_ns,
                    "sensor_timestamp_us": maybe_get_metadata(
                        color,
                        "sensor_timestamp",
                    ),
                    "backend_timestamp": maybe_get_metadata(
                        color,
                        "backend_timestamp",
                    ),
                    "time_of_arrival": maybe_get_metadata(
                        color,
                        "time_of_arrival",
                    ),
                    "wait_start_mono_ns": wait_start_mono_ns,
                    "wait_return_mono_ns": wait_return_mono_ns,
                    "brightness": brightnesses["center"],
                    "brightness_center": brightnesses["center"],
                    "brightness_top": brightnesses["top"],
                    "brightness_middle": brightnesses["middle"],
                    "brightness_bottom": brightnesses["bottom"],
                }
            )
    except Exception as exc:
        errors.append(exc)
        ready_event.set()
        stop_event.set()
    finally:
        if pipeline is not None:
            pipeline.stop()


def draw_countdown_frame(width, height, text):
    image = np.full((height, width, 3), 96, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(2.0, min(width, height) / 180.0)
    thickness = max(2, int(scale * 2))
    size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = (width - size[0]) // 2
    y = (height + size[1]) // 2
    cv2.putText(image, text, (x, y), font, scale, (255, 255, 255), thickness)
    return image


def show_for_seconds(image, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        cv2.imshow(WINDOW_NAME, image)
        if cv2.waitKey(10) == 27:
            return False
    return True


def run_display(events, run_event, stop_event, args):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
    else:
        cv2.resizeWindow(WINDOW_NAME, args.display_width, args.display_height)

    if args.countdown_s > 0:
        for remaining in range(int(math.ceil(args.countdown_s)), 0, -1):
            frame = draw_countdown_frame(
                args.display_width,
                args.display_height,
                f"Point RealSense at this window: {remaining}",
            )
            if not show_for_seconds(frame, 1.0):
                stop_event.set()
                return

    half_period_s = 1.0 / (2.0 * args.flash_hz)
    black = np.zeros((args.display_height, args.display_width, 3), dtype=np.uint8)
    white = np.full((args.display_height, args.display_width, 3), 255, dtype=np.uint8)
    frames = {0: black, 1: white}

    run_event.set()
    start_ns = time.monotonic_ns()
    end_ns = start_ns + int(args.duration_s * 1e9)
    last_state = None
    while not stop_event.is_set() and time.monotonic_ns() < end_ns:
        now_ns = time.monotonic_ns()
        phase_index = int((now_ns - start_ns) / (half_period_s * 1e9))
        state = phase_index % 2
        if state != last_state:
            cv2.imshow(WINDOW_NAME, frames[state])
            cv2.waitKey(1)
            event_ns = time.monotonic_ns()
            events.append(
                {
                    "index": len(events),
                    "event_mono_ns": event_ns,
                    "state": int(state),
                    "kind": "initial" if last_state is None else "transition",
                }
            )
            last_state = state
        if cv2.waitKey(1) == 27:
            stop_event.set()
            break
        time.sleep(0.001)

    stop_event.set()
    cv2.destroyWindow(WINDOW_NAME)


def brightness_key_for_roi(roi_name):
    return "brightness" if roi_name == "center" else f"brightness_{roi_name}"


def state_key_for_roi(roi_name):
    return (
        "classified_state"
        if roi_name == "center"
        else f"classified_state_{roi_name}"
    )


def classify_states(records, roi_name="center"):
    brightness_key = brightness_key_for_roi(roi_name)
    brightness = np.asarray(
        [row[brightness_key] for row in records],
        dtype=np.float64,
    )
    if brightness.size < 2:
        return [], float("nan"), float("nan")
    low = percentile(brightness, 10)
    high = percentile(brightness, 90)
    threshold = (low + high) / 2.0
    upper = low + 0.65 * (high - low)
    lower = low + 0.35 * (high - low)

    states = []
    state = 1 if brightness[0] >= threshold else 0
    for value in brightness:
        if state == 0 and value >= upper:
            state = 1
        elif state == 1 and value <= lower:
            state = 0
        states.append(state)
    return states, threshold, high - low


def interpolate_crossing(t0, t1, b0, b1, threshold):
    if t0 is None or t1 is None:
        return None
    if b1 == b0:
        return int(t1)
    alpha = (threshold - b0) / (b1 - b0)
    alpha = min(1.0, max(0.0, alpha))
    return int(round(int(t0) + alpha * (int(t1) - int(t0))))


def detect_camera_edges(records, states, threshold, roi_name="center"):
    brightness_key = brightness_key_for_roi(roi_name)
    edges = []
    for idx in range(1, len(records)):
        if states[idx] == states[idx - 1]:
            continue
        prev = records[idx - 1]
        curr = records[idx]
        reported_edge_ns = interpolate_crossing(
            prev["frame_global_mono_ns"],
            curr["frame_global_mono_ns"],
            prev[brightness_key],
            curr[brightness_key],
            threshold,
        )
        receive_edge_ns = interpolate_crossing(
            prev["wait_return_mono_ns"],
            curr["wait_return_mono_ns"],
            prev[brightness_key],
            curr[brightness_key],
            threshold,
        )
        edges.append(
            {
                "index": len(edges),
                "roi": roi_name,
                "frame_index": idx,
                "frame_number": curr["frame_number"],
                "state": int(states[idx]),
                "reported_edge_mono_ns": reported_edge_ns,
                "receive_edge_mono_ns": receive_edge_ns,
                "brightness_prev": prev[brightness_key],
                "brightness_curr": curr[brightness_key],
            }
        )
    return edges


def match_edges(display_events, camera_edges, args):
    transitions = [event for event in display_events if event["kind"] == "transition"]
    if not transitions or not camera_edges:
        return []

    match_window_ns = int(args.match_window_ms * 1e6)
    before_window_ns = int(args.match_before_ms * 1e6)
    matches = []
    used_edges = set()
    for event in transitions:
        best = None
        best_score = None
        for edge in camera_edges:
            if edge["index"] in used_edges:
                continue
            if edge["state"] != event["state"]:
                continue
            edge_time = edge["reported_edge_mono_ns"] or edge["receive_edge_mono_ns"]
            if edge_time is None:
                continue
            delta_ns = int(edge_time) - int(event["event_mono_ns"])
            if delta_ns < -before_window_ns or delta_ns > match_window_ns:
                continue
            score = abs(delta_ns)
            if best_score is None or score < best_score:
                best = edge
                best_score = score
        if best is None:
            continue
        used_edges.add(best["index"])
        reported_delta_ms = (
            (best["reported_edge_mono_ns"] - event["event_mono_ns"]) / 1e6
            if best["reported_edge_mono_ns"] is not None
            else float("nan")
        )
        receive_delta_ms = (
            (best["receive_edge_mono_ns"] - event["event_mono_ns"]) / 1e6
            if best["receive_edge_mono_ns"] is not None
            else float("nan")
        )
        readout_delta_ms = (
            (best["receive_edge_mono_ns"] - best["reported_edge_mono_ns"]) / 1e6
            if best["receive_edge_mono_ns"] is not None
            and best["reported_edge_mono_ns"] is not None
            else float("nan")
        )
        matches.append(
            {
                "index": len(matches),
                "roi": best.get("roi", "center"),
                "display_event_index": event["index"],
                "camera_edge_index": best["index"],
                "state": event["state"],
                "display_event_mono_ns": event["event_mono_ns"],
                "reported_edge_mono_ns": best["reported_edge_mono_ns"],
                "receive_edge_mono_ns": best["receive_edge_mono_ns"],
                "reported_minus_display_ms": reported_delta_ms,
                "receive_minus_display_ms": receive_delta_ms,
                "receive_minus_reported_ms": readout_delta_ms,
                "frame_number": best["frame_number"],
            }
        )
    return matches


def analyze(records, display_events, args):
    analyses = {}
    all_edges = []
    all_matches = []
    for roi_name in ROI_NAMES:
        states, threshold, contrast = classify_states(records, roi_name)
        camera_edges = (
            detect_camera_edges(records, states, threshold, roi_name)
            if states
            else []
        )
        matches = match_edges(display_events, camera_edges, args)
        for row, state in zip(records, states):
            row[state_key_for_roi(roi_name)] = int(state)
        analyses[roi_name] = {
            "states": states,
            "threshold": threshold,
            "contrast": contrast,
            "camera_edges": camera_edges,
            "matches": matches,
        }
        all_edges.extend(camera_edges)
        all_matches.extend(matches)

    center = analyses["center"]
    camera_edges = center["camera_edges"]
    matches = center["matches"]
    threshold = center["threshold"]
    contrast = center["contrast"]

    print()
    print("=== Screen Flash Latency Report ===")
    print(f"Captured frames: {len(records)}")
    print(f"Display events: {len(display_events)}")
    print(f"Camera edges: {len(camera_edges)}")
    print(f"Matched edges: {len(matches)}")
    print(f"Brightness threshold: {threshold:.3f}")
    print(f"Brightness contrast p90-p10: {contrast:.3f}")
    if contrast < args.min_contrast:
        print(
            "WARNING: low contrast. Point the camera closer to the flashing "
            "window or adjust exposure."
        )

    domains = sorted({row["timestamp_domain"] for row in records})
    print(f"Timestamp domains: {', '.join(domains) if domains else 'none'}")
    global_count = sum(row["frame_global_mono_ns"] is not None for row in records)
    print(f"Global-time comparable frames: {global_count}/{len(records)}")
    print()

    summarize(
        "reported_timestamp_edge - display_toggle_app_time",
        [row["reported_minus_display_ms"] for row in matches],
    )
    print()
    summarize(
        "receive_edge_time - display_toggle_app_time",
        [row["receive_minus_display_ms"] for row in matches],
    )
    print()
    summarize(
        "receive_edge_time - reported_timestamp_edge",
        [row["receive_minus_reported_ms"] for row in matches],
    )

    print()
    print("Interpretation:")
    print(
        "  reported-display is the screen-content offset relative to the "
        "RealSense reported timestamp."
    )
    print(
        "  receive-display is the total offset until Python receives the changed image."
    )
    print(
        "  Both include monitor scanout/pixel latency because the stimulus is a screen."
    )

    print()
    print("=== Multi-ROI Scanout Check ===")
    roi_summary = {}
    for roi_name in ROI_NAMES:
        roi_matches = analyses[roi_name]["matches"]
        roi_edges = analyses[roi_name]["camera_edges"]
        roi_threshold = analyses[roi_name]["threshold"]
        roi_contrast = analyses[roi_name]["contrast"]
        reported_values = [
            row["reported_minus_display_ms"] for row in roi_matches
        ]
        receive_values = [row["receive_minus_display_ms"] for row in roi_matches]
        readout_values = [
            row["receive_minus_reported_ms"] for row in roi_matches
        ]
        roi_summary[roi_name] = {
            "reported_p50": percentile(reported_values, 50),
            "receive_p50": percentile(receive_values, 50),
            "readout_p50": percentile(readout_values, 50),
        }
        print(
            f"{roi_name}: edges={len(roi_edges)} matches={len(roi_matches)} "
            f"threshold={roi_threshold:.3f} contrast={roi_contrast:.3f}"
        )
        print(
            "  p50 reported-display="
            f"{roi_summary[roi_name]['reported_p50']:.3f} ms, "
            "receive-display="
            f"{roi_summary[roi_name]['receive_p50']:.3f} ms, "
            "receive-reported="
            f"{roi_summary[roi_name]['readout_p50']:.3f} ms"
        )

    if all(
        math.isfinite(roi_summary[name]["reported_p50"])
        for name in ("top", "middle", "bottom")
    ):
        print()
        print("Vertical reported-display offsets:")
        top = roi_summary["top"]["reported_p50"]
        middle = roi_summary["middle"]["reported_p50"]
        bottom = roi_summary["bottom"]["reported_p50"]
        print(f"  middle - top:  {middle - top:.3f} ms")
        print(f"  bottom - top:  {bottom - top:.3f} ms")
        print(f"  bottom - middle: {bottom - middle:.3f} ms")
        print(
            "  On a 60 Hz display, a full top-to-bottom scan is about 16.667 ms."
        )

    return all_edges, all_matches


def write_csv(base_path, records, display_events, camera_edges, matches):
    base_path = Path(base_path).expanduser()
    base_path.parent.mkdir(parents=True, exist_ok=True)

    frames_path = base_path.with_name(base_path.stem + "_frames" + base_path.suffix)
    with frames_path.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "index",
            "frame_number",
            "timestamp_domain",
            "frame_timestamp_ms",
            "frame_global_mono_ns",
            "sensor_timestamp_us",
            "backend_timestamp",
            "time_of_arrival",
            "wait_start_mono_ns",
            "wait_return_mono_ns",
            "brightness",
            "brightness_center",
            "brightness_top",
            "brightness_middle",
            "brightness_bottom",
            "classified_state",
            "classified_state_top",
            "classified_state_middle",
            "classified_state_bottom",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Frames CSV written to: {frames_path}")

    events_path = base_path.with_name(base_path.stem + "_display_events" + base_path.suffix)
    with events_path.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = ["index", "event_mono_ns", "state", "kind"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(display_events)
    print(f"Display events CSV written to: {events_path}")

    edges_path = base_path.with_name(base_path.stem + "_camera_edges" + base_path.suffix)
    with edges_path.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "index",
            "roi",
            "frame_index",
            "frame_number",
            "state",
            "reported_edge_mono_ns",
            "receive_edge_mono_ns",
            "brightness_prev",
            "brightness_curr",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(camera_edges)
    print(f"Camera edges CSV written to: {edges_path}")

    matches_path = base_path.with_name(base_path.stem + "_matches" + base_path.suffix)
    with matches_path.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "index",
            "roi",
            "display_event_index",
            "camera_edge_index",
            "state",
            "display_event_mono_ns",
            "reported_edge_mono_ns",
            "receive_edge_mono_ns",
            "reported_minus_display_ms",
            "receive_minus_display_ms",
            "receive_minus_reported_ms",
            "frame_number",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)
    print(f"Matches CSV written to: {matches_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--width", type=positive_int, default=640)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--timeout_ms", type=positive_int, default=2000)
    parser.add_argument("--duration_s", type=positive_float, default=20.0)
    parser.add_argument("--post_roll_s", type=nonnegative_float, default=1.0)
    parser.add_argument("--camera_warmup_s", type=nonnegative_float, default=1.0)
    parser.add_argument("--countdown_s", type=nonnegative_float, default=3.0)
    parser.add_argument("--flash_hz", type=positive_float, default=2.0)
    parser.add_argument("--roi_fraction", type=positive_float, default=0.6)
    parser.add_argument(
        "--roi_band_fraction",
        type=positive_float,
        default=0.18,
        help="Frame-height fraction for each top/middle/bottom ROI band.",
    )
    parser.add_argument("--display_width", type=positive_int, default=1280)
    parser.add_argument("--display_height", type=positive_int, default=720)
    parser.add_argument("--fullscreen", action="store_true", default=True)
    parser.add_argument("--windowed", action="store_false", dest="fullscreen")
    parser.add_argument("--manual_exposure_us", type=positive_float, default=None)
    parser.add_argument("--disable_auto_white_balance", action="store_true")
    parser.add_argument("--match_window_ms", type=positive_float, default=250.0)
    parser.add_argument("--match_before_ms", type=nonnegative_float, default=50.0)
    parser.add_argument("--min_contrast", type=positive_float, default=20.0)
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV base path. Use 'auto' to write under /tmp.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    devices = list_devices()
    if not devices:
        raise RuntimeError("no RealSense devices found")

    print("RealSense devices:")
    for serial, name in devices:
        marker = " *" if args.serial and serial == args.serial else ""
        print(f"  {serial}  {name}{marker}")
    serial = args.serial or devices[0][0]
    if args.serial is None:
        print(f"Using first device: {serial}")

    records = []
    display_events = []
    errors = []
    ready_event = threading.Event()
    run_event = threading.Event()
    stop_event = threading.Event()

    camera_thread = threading.Thread(
        target=camera_worker,
        args=(records, errors, ready_event, run_event, stop_event, serial, args),
        name="realsense-capture",
        daemon=True,
    )
    camera_thread.start()
    print("Starting RealSense and warming up...")
    ready_event.wait()
    if errors:
        raise RuntimeError(f"camera worker failed: {errors[0]}")

    print("Opening flashing target window. Press Esc to stop early.")
    try:
        run_display(display_events, run_event, stop_event, args)
    finally:
        stop_event.set()
        camera_thread.join(timeout=5.0)
        cv2.destroyAllWindows()

    if errors:
        raise RuntimeError(f"camera worker failed: {errors[0]}")
    camera_edges, matches = analyze(records, display_events, args)

    if args.csv:
        if args.csv == "auto":
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"/tmp/realsense_screen_flash_latency_{stamp}.csv"
        else:
            csv_path = args.csv
        write_csv(csv_path, records, display_events, camera_edges, matches)


if __name__ == "__main__":
    main()
