#!/usr/bin/env python3
"""Estimate RealSense visual timestamp offset by filming a QR timecode.

The target window displays QR codes whose payload is the host monotonic time
when that QR image was generated. The camera thread decodes the QR code in each
captured frame and compares the QR time to both:

    * RealSense global timestamp mapped to host monotonic_ns
    * host monotonic_ns when wait_for_frames returned

This is intended as a practical check for large image-content timestamp offsets.
It includes monitor scanout and pixel response, so it is not a sub-millisecond
photodiode-style exposure calibration.
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


WINDOW_NAME = "RealSense QR timecode latency target"
QR_PREFIX = "RSQR:"


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


def parse_qr_payload(text):
    text = str(text).strip()
    if not text.startswith(QR_PREFIX):
        return None
    value = text[len(QR_PREFIX) :]
    if not value.isdigit():
        return None
    return int(value)


def create_qr_encoder():
    if not hasattr(cv2, "QRCodeEncoder_create"):
        raise RuntimeError("OpenCV QRCodeEncoder_create is unavailable")
    return cv2.QRCodeEncoder_create()


def make_qr_canvas(encoder, payload, args):
    qr = encoder.encode(payload)
    if qr.ndim == 3:
        qr = cv2.cvtColor(qr, cv2.COLOR_BGR2GRAY)
    qr = np.asarray(qr, dtype=np.uint8)
    qr = cv2.copyMakeBorder(
        qr,
        args.qr_quiet_modules,
        args.qr_quiet_modules,
        args.qr_quiet_modules,
        args.qr_quiet_modules,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    qr = cv2.resize(
        qr,
        (args.qr_size, args.qr_size),
        interpolation=cv2.INTER_NEAREST,
    )
    qr_rgb = cv2.cvtColor(qr, cv2.COLOR_GRAY2RGB)
    canvas = np.full(
        (args.display_height, args.display_width, 3),
        255,
        dtype=np.uint8,
    )
    x0 = max(0, (args.display_width - args.qr_size) // 2)
    y0 = max(0, (args.display_height - args.qr_size) // 2)
    x1 = min(args.display_width, x0 + args.qr_size)
    y1 = min(args.display_height, y0 + args.qr_size)
    canvas[y0:y1, x0:x1] = qr_rgb[: y1 - y0, : x1 - x0]
    return canvas


def draw_countdown_frame(width, height, text):
    image = np.full((height, width, 3), 96, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(1.4, min(width, height) / 260.0)
    thickness = max(2, int(scale * 2))
    size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (width - size[0]) // 2)
    y = max(size[1] + 8, (height + size[1]) // 2)
    cv2.putText(image, text, (x, y), font, scale, (255, 255, 255), thickness)
    return image


def show_for_seconds(image, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        cv2.imshow(WINDOW_NAME, image)
        if cv2.waitKey(10) == 27:
            return False
    return True


def decode_qr(detector, image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    text, corners, _ = detector.detectAndDecode(gray)
    if not text and hasattr(detector, "detectAndDecodeCurved"):
        text, corners, _ = detector.detectAndDecodeCurved(gray)
    return text, corners


def corner_area(corners):
    if corners is None:
        return float("nan")
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if len(points) < 4:
        return float("nan")
    return float(abs(cv2.contourArea(points)))


def finite_ms(a_ns, b_ns):
    if a_ns is None or b_ns is None:
        return float("nan")
    return (int(a_ns) - int(b_ns)) / 1e6


def camera_worker(
    records,
    display_event_by_payload,
    display_lock,
    errors,
    ready_event,
    run_event,
    stop_event,
    serial,
    args,
):
    pipeline = None
    detector = cv2.QRCodeDetector()
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
            decode_start_mono_ns = time.monotonic_ns()
            qr_text, corners = decode_qr(detector, image)
            decode_end_mono_ns = time.monotonic_ns()
            qr_payload_mono_ns = parse_qr_payload(qr_text)

            frame_ts_ms = float(color.get_timestamp())
            domain = enum_name(color.get_frame_timestamp_domain())
            epoch_minus_mono_ns = wait_return_epoch_ns - wait_return_mono_ns
            frame_global_mono_ns = (
                int(round(frame_ts_ms * 1e6)) - epoch_minus_mono_ns
                if is_global_domain(domain)
                else None
            )

            display_event = None
            if qr_payload_mono_ns is not None:
                with display_lock:
                    display_event = display_event_by_payload.get(qr_payload_mono_ns)

            display_call_mono_ns = None
            if display_event is not None:
                display_call_mono_ns = display_event["display_call_mono_ns"]

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
                    "actual_exposure_us": maybe_get_metadata(
                        color,
                        "actual_exposure",
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
                    "qr_text": qr_text,
                    "qr_payload_mono_ns": qr_payload_mono_ns,
                    "qr_decoded": int(qr_payload_mono_ns is not None),
                    "qr_corner_area_px": corner_area(corners),
                    "qr_decode_ms": (decode_end_mono_ns - decode_start_mono_ns)
                    / 1e6,
                    "display_call_mono_ns": display_call_mono_ns,
                    "reported_minus_qr_payload_ms": finite_ms(
                        frame_global_mono_ns,
                        qr_payload_mono_ns,
                    ),
                    "receive_minus_qr_payload_ms": finite_ms(
                        wait_return_mono_ns,
                        qr_payload_mono_ns,
                    ),
                    "reported_minus_display_call_ms": finite_ms(
                        frame_global_mono_ns,
                        display_call_mono_ns,
                    ),
                    "receive_minus_display_call_ms": finite_ms(
                        wait_return_mono_ns,
                        display_call_mono_ns,
                    ),
                    "receive_minus_reported_ms": finite_ms(
                        wait_return_mono_ns,
                        frame_global_mono_ns,
                    ),
                }
            )
    except Exception as exc:
        errors.append(exc)
        ready_event.set()
        stop_event.set()
    finally:
        if pipeline is not None:
            pipeline.stop()


def run_display(
    display_events,
    display_event_by_payload,
    display_lock,
    run_event,
    stop_event,
    args,
):
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

    encoder = create_qr_encoder()
    period_s = 1.0 / args.qr_hz
    run_event.set()
    start_s = time.monotonic()
    update_end_s = start_s + args.duration_s
    display_end_s = update_end_s + args.post_roll_s
    next_update_s = start_s
    last_image = None

    while not stop_event.is_set() and time.monotonic() < display_end_s:
        now_s = time.monotonic()
        should_update = now_s >= next_update_s and now_s < update_end_s
        if should_update or last_image is None:
            payload_mono_ns = time.monotonic_ns()
            payload = f"{QR_PREFIX}{payload_mono_ns}"
            image = make_qr_canvas(encoder, payload, args)
            cv2.imshow(WINDOW_NAME, image)
            keycode = cv2.waitKey(1)
            display_call_mono_ns = time.monotonic_ns()
            event = {
                "index": len(display_events),
                "qr_payload": payload,
                "qr_payload_mono_ns": payload_mono_ns,
                "display_call_mono_ns": display_call_mono_ns,
                "display_call_minus_payload_ms": (
                    display_call_mono_ns - payload_mono_ns
                )
                / 1e6,
            }
            with display_lock:
                display_events.append(event)
                display_event_by_payload[payload_mono_ns] = event
            last_image = image
            if keycode == 27:
                stop_event.set()
                return
            if now_s - next_update_s > period_s:
                next_update_s = now_s + period_s
            else:
                next_update_s += period_s
        else:
            if last_image is not None:
                cv2.imshow(WINDOW_NAME, last_image)
            keycode = cv2.waitKey(1)
            if keycode == 27:
                stop_event.set()
                return
            time.sleep(min(0.002, max(0.0, next_update_s - time.monotonic())))


def first_decoded_per_qr(records):
    first = []
    seen = set()
    for record in records:
        payload_ns = record.get("qr_payload_mono_ns")
        if payload_ns is None or payload_ns in seen:
            continue
        seen.add(payload_ns)
        row = dict(record)
        row["first_detection_index"] = len(first)
        first.append(row)
    return first


def print_report(records, display_events, first_records):
    print()
    print("QR timecode latency report")
    print(f"Frames captured: {len(records)}")
    print(f"QR display events: {len(display_events)}")
    decoded = [row for row in records if row.get("qr_payload_mono_ns") is not None]
    print(f"Decoded frames: {len(decoded)}")
    if records:
        print(f"Frame decode rate: {len(decoded) / len(records):.2%}")
    if display_events:
        print(
            f"QR first-detection rate: "
            f"{len(first_records) / len(display_events):.2%}"
        )

    domains = sorted({row["timestamp_domain"] for row in records})
    if domains:
        print(f"Timestamp domains: {', '.join(domains)}")

    print()
    print("Main estimates. Use first decoded frame per QR to reduce QR-age bias.")
    summarize(
        "first decoded: reported timestamp - QR payload",
        [row["reported_minus_qr_payload_ms"] for row in first_records],
    )
    summarize(
        "first decoded: reported timestamp - display call",
        [row["reported_minus_display_call_ms"] for row in first_records],
    )
    print()
    summarize(
        "first decoded: receive time - display call",
        [row["receive_minus_display_call_ms"] for row in first_records],
    )
    summarize(
        "first decoded: receive time - reported timestamp",
        [row["receive_minus_reported_ms"] for row in first_records],
    )

    print()
    print("Diagnostics.")
    summarize(
        "all decoded: QR decode time",
        [row["qr_decode_ms"] for row in decoded],
    )
    summarize(
        "display call - QR payload",
        [row["display_call_minus_payload_ms"] for row in display_events],
    )
    summarize(
        "all decoded: reported timestamp - display call",
        [row["reported_minus_display_call_ms"] for row in decoded],
    )
    summarize(
        "all decoded: receive time - display call",
        [row["receive_minus_display_call_ms"] for row in decoded],
    )

    print()
    print("Error budget rule of thumb:")
    print("  60 FPS camera: roughly 20-40 ms absolute uncertainty with a normal monitor.")
    print("  30 FPS camera: roughly 40-70 ms absolute uncertainty with a normal monitor.")
    print("  A 100 ms-class offset should be visible; a 500 ms-class offset is unambiguous.")
    print(
        "  The residual comes mostly from monitor refresh/scanout, rolling shutter, "
        "and missed first QR detections."
    )


def write_csv(base_path, records, display_events, first_records):
    base_path = Path(base_path).expanduser()
    base_path.parent.mkdir(parents=True, exist_ok=True)

    frames_path = base_path
    frame_fields = [
        "index",
        "frame_number",
        "timestamp_domain",
        "frame_timestamp_ms",
        "frame_global_mono_ns",
        "sensor_timestamp_us",
        "actual_exposure_us",
        "backend_timestamp",
        "time_of_arrival",
        "wait_start_mono_ns",
        "wait_return_mono_ns",
        "qr_text",
        "qr_payload_mono_ns",
        "qr_decoded",
        "qr_corner_area_px",
        "qr_decode_ms",
        "display_call_mono_ns",
        "reported_minus_qr_payload_ms",
        "receive_minus_qr_payload_ms",
        "reported_minus_display_call_ms",
        "receive_minus_display_call_ms",
        "receive_minus_reported_ms",
    ]
    with frames_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Frames CSV written to: {frames_path}")

    events_path = base_path.with_name(
        base_path.stem + "_display_events" + base_path.suffix
    )
    event_fields = [
        "index",
        "qr_payload",
        "qr_payload_mono_ns",
        "display_call_mono_ns",
        "display_call_minus_payload_ms",
    ]
    with events_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(display_events)
    print(f"Display events CSV written to: {events_path}")

    first_path = base_path.with_name(
        base_path.stem + "_first_decoded" + base_path.suffix
    )
    first_fields = ["first_detection_index"] + frame_fields
    with first_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=first_fields)
        writer.writeheader()
        writer.writerows(first_records)
    print(f"First detections CSV written to: {first_path}")


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
    parser.add_argument("--qr_hz", type=positive_float, default=30.0)
    parser.add_argument("--qr_size", type=positive_int, default=720)
    parser.add_argument("--qr_quiet_modules", type=nonnegative_float, default=4)
    parser.add_argument("--display_width", type=positive_int, default=1280)
    parser.add_argument("--display_height", type=positive_int, default=720)
    parser.add_argument("--fullscreen", action="store_true", default=True)
    parser.add_argument("--windowed", action="store_false", dest="fullscreen")
    parser.add_argument("--manual_exposure_us", type=positive_float, default=None)
    parser.add_argument("--disable_auto_white_balance", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV base path. Use 'auto' to write under /tmp.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.qr_quiet_modules = int(args.qr_quiet_modules)
    if args.qr_size > min(args.display_width, args.display_height):
        raise ValueError("--qr_size must fit inside the display window")

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

    cv2.setNumThreads(1)
    records = []
    display_events = []
    display_event_by_payload = {}
    display_lock = threading.Lock()
    errors = []
    ready_event = threading.Event()
    run_event = threading.Event()
    stop_event = threading.Event()

    camera_thread = threading.Thread(
        target=camera_worker,
        args=(
            records,
            display_event_by_payload,
            display_lock,
            errors,
            ready_event,
            run_event,
            stop_event,
            serial,
            args,
        ),
        name="realsense-qr-capture",
        daemon=True,
    )
    camera_thread.start()
    print("Starting RealSense and warming up...")
    ready_event.wait()
    if errors:
        raise RuntimeError(f"camera worker failed: {errors[0]}")

    print("Opening QR target window. Press Esc to stop early.")
    try:
        run_display(
            display_events,
            display_event_by_payload,
            display_lock,
            run_event,
            stop_event,
            args,
        )
    finally:
        stop_event.set()
        camera_thread.join(timeout=5.0)
        cv2.destroyAllWindows()

    if errors:
        raise RuntimeError(f"camera worker failed: {errors[0]}")
    first_records = first_decoded_per_qr(records)
    print_report(records, display_events, first_records)

    if args.output:
        if args.output == "auto":
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/tmp/realsense_qr_timecode_latency_{stamp}.csv"
        else:
            output_path = args.output
        write_csv(output_path, records, display_events, first_records)


if __name__ == "__main__":
    main()
