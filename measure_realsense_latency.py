#!/usr/bin/env python3
"""Measure RealSense frame timestamp to host receive latency.

This script enables RealSense global time when supported, captures color
frames, and compares frame.get_timestamp() with the host wall-clock time when
wait_for_frames() returns.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import statistics
import time

rs = None
cv2 = None
np = None


def load_rs():
    global rs
    if rs is not None:
        return rs
    try:
        import pyrealsense2 as _rs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyrealsense2 is not installed in this Python environment. "
            "Run this script with the same environment you use for RealSense collection."
        ) from exc
    rs = _rs
    return rs


def load_image_libs():
    global cv2, np
    if cv2 is not None and np is not None:
        return cv2, np
    try:
        import cv2 as _cv2
        import numpy as _np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "cv2/numpy are required when emulating the collection image path."
        ) from exc
    cv2 = _cv2
    np = _np
    return cv2, np


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * float(pct) / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def summarize(name, values, unit="ms"):
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


def enum_name(value):
    text = str(value)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def list_devices():
    rs_mod = load_rs()
    ctx = rs_mod.context()
    devices = []
    for dev in ctx.query_devices():
        serial = dev.get_info(rs_mod.camera_info.serial_number)
        name = dev.get_info(rs_mod.camera_info.name)
        devices.append((serial, name))
    return devices


def set_global_time(profile, enabled=True):
    rs_mod = load_rs()
    changed = []
    for sensor in profile.get_device().query_sensors():
        name = sensor.get_info(rs_mod.camera_info.name)
        if sensor.supports(rs_mod.option.global_time_enabled):
            sensor.set_option(rs_mod.option.global_time_enabled, 1.0 if enabled else 0.0)
            changed.append((name, True))
        else:
            changed.append((name, False))
    return changed


def maybe_get_metadata(frame, metadata_name):
    rs_mod = load_rs()
    metadata = getattr(rs_mod.frame_metadata_value, metadata_name, None)
    if metadata is None:
        return None
    try:
        if frame.supports_frame_metadata(metadata):
            return frame.get_frame_metadata(metadata)
    except Exception:
        return None
    return None


def capture(args):
    rs_mod = load_rs()
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

    pipeline = rs_mod.pipeline()
    config = rs_mod.config()
    config.enable_device(serial)
    config.enable_stream(
        rs_mod.stream.color,
        args.width,
        args.height,
        rs_mod.format.rgb8,
        args.fps,
    )

    profile = pipeline.start(config)
    try:
        changed = set_global_time(profile, enabled=not args.disable_global_time)
        for sensor_name, supported in changed:
            status = "enabled" if supported and not args.disable_global_time else "disabled"
            if not supported:
                status = "not supported"
            print(f"global_time_enabled on {sensor_name}: {status}")

        total_frames = args.warmup + args.frames
        records = []
        print(
            f"Capturing {args.frames} frames "
            f"(warmup {args.warmup}, {args.width}x{args.height}@{args.fps})..."
        )
        for frame_idx in range(total_frames):
            wait_start_mono_ns = time.monotonic_ns()
            frameset = pipeline.wait_for_frames(timeout_ms=args.timeout_ms)
            host_wait_return_epoch_ns = time.time_ns()
            host_after_mono_ns = time.monotonic_ns()
            wait_ms = (host_after_mono_ns - wait_start_mono_ns) / 1e6

            color = frameset.get_color_frame()
            if not color:
                continue

            if not args.no_emulate_collection:
                cv2_mod, np_mod = load_image_libs()
                image = np_mod.asanyarray(color.get_data(), dtype=np_mod.uint8)
                if args.target_size > 0:
                    image = cv2_mod.resize(
                        image,
                        (args.target_size, args.target_size),
                        interpolation=cv2_mod.INTER_LINEAR,
                    )

            host_after_epoch_ns = time.time_ns()
            if frame_idx < args.warmup:
                continue

            frame_ts_ms = float(color.get_timestamp())
            host_wait_return_epoch_ms = host_wait_return_epoch_ns / 1e6
            host_after_epoch_ms = host_after_epoch_ns / 1e6
            wait_return_latency_ms = host_wait_return_epoch_ms - frame_ts_ms
            latency_ms = host_after_epoch_ms - frame_ts_ms
            image_processing_ms = (host_after_epoch_ns - host_wait_return_epoch_ns) / 1e6
            domain = enum_name(color.get_frame_timestamp_domain())
            frame_no = int(color.get_frame_number())
            backend_ts = maybe_get_metadata(color, "backend_timestamp")
            sensor_ts = maybe_get_metadata(color, "sensor_timestamp")
            arrival_ts = maybe_get_metadata(color, "time_of_arrival")

            records.append(
                {
                    "index": len(records),
                    "frame_number": frame_no,
                    "timestamp_domain": domain,
                    "frame_timestamp_ms": frame_ts_ms,
                    "host_wait_return_epoch_ms": host_wait_return_epoch_ms,
                    "host_after_epoch_ms": host_after_epoch_ms,
                    "host_after_mono_ns": host_after_mono_ns,
                    "wait_return_latency_ms": wait_return_latency_ms,
                    "latency_ms": latency_ms,
                    "image_processing_ms": image_processing_ms,
                    "wait_ms": wait_ms,
                    "backend_timestamp": backend_ts,
                    "sensor_timestamp": sensor_ts,
                    "time_of_arrival": arrival_ts,
                }
            )
    finally:
        pipeline.stop()

    return serial, records


def write_csv(path, records):
    if not path:
        return
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "frame_number",
        "timestamp_domain",
        "frame_timestamp_ms",
        "host_wait_return_epoch_ms",
        "host_after_epoch_ms",
        "host_after_mono_ns",
        "wait_return_latency_ms",
        "latency_ms",
        "image_processing_ms",
        "wait_ms",
        "backend_timestamp",
        "sensor_timestamp",
        "time_of_arrival",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV written to: {path}")


def print_report(serial, records):
    if not records:
        print("No records captured.")
        return

    domains = sorted({row["timestamp_domain"] for row in records})
    print()
    print(f"Device: {serial}")
    print(f"Timestamp domains: {', '.join(domains)}")

    raw_latency = [float(row["latency_ms"]) for row in records]
    raw_wait_return_latency = [
        float(row["wait_return_latency_ms"]) for row in records
    ]
    plausible_latency = [v for v in raw_latency if -1000.0 < v < 5000.0]
    if len(plausible_latency) == len(raw_latency):
        summarize("collection_t_host - frame_timestamp", plausible_latency)
        print()
        summarize(
            "wait_for_frames_return - frame_timestamp",
            raw_wait_return_latency,
        )
        print()
        print("Suggested starting compensation:")
        print(
            "  use p05/min as lower-bound pipeline delay, "
            "p50 as typical receive-time delay"
        )
        print(f"  p05 ~= {percentile(plausible_latency, 5):.1f} ms")
        print(f"  p50 ~= {percentile(plausible_latency, 50):.1f} ms")
    else:
        print("Absolute latency was not plausible.")
        print("This usually means frame timestamps are not in the host wall-clock domain.")
        print("Check that global_time_enabled is supported and timestamp_domain is global/system time.")
        summarize("raw host_receive_time - frame_timestamp", raw_latency)

    frame_ts = [float(row["frame_timestamp_ms"]) for row in records]
    host_ms = [float(row["host_after_epoch_ms"]) for row in records]
    if len(records) > 1:
        frame_intervals = [b - a for a, b in zip(frame_ts[:-1], frame_ts[1:])]
        host_intervals = [b - a for a, b in zip(host_ms[:-1], host_ms[1:])]
        print()
        summarize("frame timestamp interval", frame_intervals)
        print()
        summarize("host receive interval", host_intervals)

    wait_ms = [float(row["wait_ms"]) for row in records]
    image_processing_ms = [float(row["image_processing_ms"]) for row in records]
    print()
    summarize("image materialize/resize time", image_processing_ms)
    print()
    summarize("wait_for_frames blocking time", wait_ms)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", type=str, default=None, help="RealSense serial number.")
    parser.add_argument("--width", type=positive_int, default=640)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--frames", type=positive_int, default=300)
    parser.add_argument("--warmup", type=nonnegative_int, default=60)
    parser.add_argument("--timeout_ms", type=positive_int, default=2000)
    parser.add_argument(
        "--target_size",
        type=nonnegative_int,
        default=256,
        help="Resize color image to this square size when emulating collection; 0 disables resize.",
    )
    parser.add_argument(
        "--no_emulate_collection",
        action="store_true",
        help="Only measure wait_for_frames return time; skip image materialize/resize.",
    )
    parser.add_argument(
        "--disable_global_time",
        action="store_true",
        help="Do not enable RealSense global_time_enabled.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path. Default writes to /tmp with timestamp.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"/tmp/realsense_latency_{stamp}.csv"
    serial, records = capture(args)
    write_csv(args.output, records)
    print_report(serial, records)


if __name__ == "__main__":
    main()
