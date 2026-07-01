#!/usr/bin/env python3
"""Measure the camera timestamp delta relevant to servoL.py.

servoL.py aligns robot/action samples to front_meta["t_host_ns"]. In
SM_multi_realsense.py that timestamp is taken after wait_for_frames(), numpy
materialization, and resize. This script mirrors that path and compares it with
RealSense frame timestamps.

The absolute comparison is only valid when frame.get_timestamp() is in a
host/global timestamp domain. SENSOR_TIMESTAMP is still printed and summarized,
but its absolute value is a device clock and cannot be directly subtracted from
time.monotonic_ns().
"""

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import statistics
import time

import numpy as np
import pyrealsense2 as rs


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


def enum_name(value):
    text = str(value)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def is_host_time_domain(domain_name):
    text = str(domain_name).lower()
    return "global" in text or "system" in text


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
        print(f"{name}: no comparable samples")
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


def maybe_load_cv2(target_size):
    if target_size <= 0:
        return None
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "cv2 is required when --target_size is greater than 0"
        ) from exc
    return cv2


def capture(args):
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

    cv2 = maybe_load_cv2(args.target_size)
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
    try:
        changed = set_global_time(profile, enabled=not args.disable_global_time)
        for sensor_name, supported in changed:
            if not supported:
                status = "not supported"
            else:
                status = "enabled" if not args.disable_global_time else "disabled"
            print(f"global_time_enabled on {sensor_name}: {status}")

        # Let global-time estimation and auto exposure settle a little.
        time.sleep(args.settle_ms / 1000.0)

        records = []
        total_frames = args.warmup + args.frames
        print(
            f"Capturing {args.frames} frames "
            f"(warmup {args.warmup}, {args.width}x{args.height}@{args.fps}, "
            f"target_size={args.target_size})..."
        )
        for idx in range(total_frames):
            wait_start_mono_ns = time.monotonic_ns()
            frames = pipeline.wait_for_frames(args.timeout_ms)
            wait_return_epoch_ns = time.time_ns()
            wait_return_mono_ns = time.monotonic_ns()

            color = frames.get_color_frame()
            if not color:
                continue

            frame_timestamp_ms = float(color.get_timestamp())
            timestamp_domain = enum_name(color.get_frame_timestamp_domain())
            sensor_timestamp_us = maybe_get_metadata(color, "sensor_timestamp")
            actual_exposure_us = maybe_get_metadata(color, "actual_exposure")
            backend_timestamp = maybe_get_metadata(color, "backend_timestamp")
            time_of_arrival = maybe_get_metadata(color, "time_of_arrival")
            frame_no = int(color.get_frame_number())

            image = np.asanyarray(color.get_data(), dtype=np.uint8)
            if args.target_size > 0:
                image = cv2.resize(
                    image,
                    (args.target_size, args.target_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            # This is the timestamp location that mirrors SM_multi_realsense.py.
            servoL_anchor_mono_ns = time.monotonic_ns()
            servoL_anchor_epoch_ns = time.time_ns()

            if idx < args.warmup:
                continue

            epoch_minus_mono_ns = wait_return_epoch_ns - wait_return_mono_ns
            if is_host_time_domain(timestamp_domain):
                frame_timestamp_mono_ns = (
                    int(round(frame_timestamp_ms * 1e6)) - epoch_minus_mono_ns
                )
                wait_return_minus_frame_ms = (
                    wait_return_mono_ns - frame_timestamp_mono_ns
                ) / 1e6
                servoL_anchor_minus_frame_ms = (
                    servoL_anchor_mono_ns - frame_timestamp_mono_ns
                ) / 1e6
            else:
                frame_timestamp_mono_ns = None
                wait_return_minus_frame_ms = float("nan")
                servoL_anchor_minus_frame_ms = float("nan")

            if (
                sensor_timestamp_us is not None
                and not is_host_time_domain(timestamp_domain)
            ):
                frame_timestamp_minus_sensor_ms = (
                    frame_timestamp_ms - float(sensor_timestamp_us) / 1000.0
                )
            else:
                frame_timestamp_minus_sensor_ms = float("nan")

            records.append(
                {
                    "index": len(records),
                    "frame_number": frame_no,
                    "timestamp_domain": timestamp_domain,
                    "frame_timestamp_ms": frame_timestamp_ms,
                    "frame_timestamp_mono_ns": frame_timestamp_mono_ns,
                    "sensor_timestamp_us": sensor_timestamp_us,
                    "actual_exposure_us": actual_exposure_us,
                    "backend_timestamp": backend_timestamp,
                    "time_of_arrival": time_of_arrival,
                    "wait_start_mono_ns": wait_start_mono_ns,
                    "wait_return_mono_ns": wait_return_mono_ns,
                    "wait_return_epoch_ns": wait_return_epoch_ns,
                    "servoL_anchor_mono_ns": servoL_anchor_mono_ns,
                    "servoL_anchor_epoch_ns": servoL_anchor_epoch_ns,
                    "wait_block_ms": (
                        wait_return_mono_ns - wait_start_mono_ns
                    ) / 1e6,
                    "post_wait_to_servoL_anchor_ms": (
                        servoL_anchor_mono_ns - wait_return_mono_ns
                    ) / 1e6,
                    "wait_return_minus_frame_timestamp_ms": (
                        wait_return_minus_frame_ms
                    ),
                    "servoL_anchor_minus_frame_timestamp_ms": (
                        servoL_anchor_minus_frame_ms
                    ),
                    "frame_get_timestamp_minus_sensor_timestamp_ms": (
                        frame_timestamp_minus_sensor_ms
                    ),
                }
            )
    finally:
        pipeline.stop()

    add_relative_sensor_fields(records)
    return records


def add_relative_sensor_fields(records):
    base = None
    for record in records:
        sensor_timestamp_us = record.get("sensor_timestamp_us")
        if sensor_timestamp_us is None:
            continue
        base = record
        break
    if base is None:
        return

    base_sensor_us = float(base["sensor_timestamp_us"])
    base_anchor_ns = int(base["servoL_anchor_mono_ns"])
    previous = None
    for record in records:
        sensor_timestamp_us = record.get("sensor_timestamp_us")
        if sensor_timestamp_us is None:
            record["sensor_rel_ms"] = float("nan")
            record["servoL_anchor_rel_ms"] = float("nan")
            record["servoL_anchor_rel_minus_sensor_rel_ms"] = float("nan")
            record["sensor_interval_ms"] = float("nan")
            record["servoL_anchor_interval_ms"] = float("nan")
            continue

        sensor_rel_ms = (float(sensor_timestamp_us) - base_sensor_us) / 1000.0
        anchor_rel_ms = (
            int(record["servoL_anchor_mono_ns"]) - base_anchor_ns
        ) / 1e6
        record["sensor_rel_ms"] = sensor_rel_ms
        record["servoL_anchor_rel_ms"] = anchor_rel_ms
        record["servoL_anchor_rel_minus_sensor_rel_ms"] = (
            anchor_rel_ms - sensor_rel_ms
        )

        if previous is None or previous.get("sensor_timestamp_us") is None:
            record["sensor_interval_ms"] = float("nan")
            record["servoL_anchor_interval_ms"] = float("nan")
        else:
            record["sensor_interval_ms"] = (
                float(sensor_timestamp_us) - float(previous["sensor_timestamp_us"])
            ) / 1000.0
            record["servoL_anchor_interval_ms"] = (
                int(record["servoL_anchor_mono_ns"])
                - int(previous["servoL_anchor_mono_ns"])
            ) / 1e6
        previous = record


def write_csv(path, records):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "frame_number",
        "timestamp_domain",
        "frame_timestamp_ms",
        "frame_timestamp_mono_ns",
        "sensor_timestamp_us",
        "actual_exposure_us",
        "backend_timestamp",
        "time_of_arrival",
        "wait_start_mono_ns",
        "wait_return_mono_ns",
        "wait_return_epoch_ns",
        "servoL_anchor_mono_ns",
        "servoL_anchor_epoch_ns",
        "wait_block_ms",
        "post_wait_to_servoL_anchor_ms",
        "wait_return_minus_frame_timestamp_ms",
        "servoL_anchor_minus_frame_timestamp_ms",
        "frame_get_timestamp_minus_sensor_timestamp_ms",
        "sensor_rel_ms",
        "servoL_anchor_rel_ms",
        "servoL_anchor_rel_minus_sensor_rel_ms",
        "sensor_interval_ms",
        "servoL_anchor_interval_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV written to: {path}")


def print_report(records):
    if not records:
        print("No records captured.")
        return

    domains = sorted({row["timestamp_domain"] for row in records})
    print()
    print(f"Timestamp domains: {', '.join(domains)}")
    comparable_count = sum(is_host_time_domain(row["timestamp_domain"]) for row in records)
    print(f"Host/global comparable frames: {comparable_count}/{len(records)}")
    print()

    summarize(
        "servoL_anchor - frame.get_timestamp",
        [row["servoL_anchor_minus_frame_timestamp_ms"] for row in records],
    )
    print()
    summarize(
        "frame.get_timestamp - SENSOR_TIMESTAMP",
        [
            row["frame_get_timestamp_minus_sensor_timestamp_ms"]
            for row in records
        ],
    )
    print()
    summarize(
        "wait_for_frames_return - frame.get_timestamp",
        [row["wait_return_minus_frame_timestamp_ms"] for row in records],
    )
    print()
    summarize(
        "servoL_anchor - wait_for_frames_return",
        [row["post_wait_to_servoL_anchor_ms"] for row in records],
    )
    print()
    summarize(
        "wait_for_frames blocking time",
        [row["wait_block_ms"] for row in records],
    )
    print()
    summarize(
        "SENSOR_TIMESTAMP frame interval",
        [row.get("sensor_interval_ms", float("nan")) for row in records],
    )
    print()
    summarize(
        "servoL_anchor frame interval",
        [row.get("servoL_anchor_interval_ms", float("nan")) for row in records],
    )
    print()
    summarize(
        "relative servoL_anchor drift vs SENSOR_TIMESTAMP",
        [
            row.get("servoL_anchor_rel_minus_sensor_rel_ms", float("nan"))
            for row in records
        ],
    )

    comparable_values = [
        row["servoL_anchor_minus_frame_timestamp_ms"]
        for row in records
        if math.isfinite(float(row["servoL_anchor_minus_frame_timestamp_ms"]))
    ]
    if comparable_values:
        print()
        print("Suggested compensation if you keep servoL's current host anchor:")
        print(f"  lower-bound delay p05: {percentile(comparable_values, 5):.3f} ms")
        print(f"  typical delay p50:     {percentile(comparable_values, 50):.3f} ms")
        print(
            "  subtract roughly p50 from t_front_camera_host_ns to approximate "
            "the image exposure time."
        )
    else:
        print()
        print("Absolute delta was not comparable.")
        print(
            "SENSOR_TIMESTAMP is a camera hardware clock. Use the relative drift "
            "summary above, or enable RealSense global_time_enabled so "
            "frame.get_timestamp() lands in a host/global time domain."
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--width", type=positive_int, default=640)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--frames", type=positive_int, default=300)
    parser.add_argument("--warmup", type=nonnegative_int, default=60)
    parser.add_argument("--timeout_ms", type=positive_int, default=2000)
    parser.add_argument("--settle_ms", type=nonnegative_int, default=1000)
    parser.add_argument(
        "--target_size",
        type=nonnegative_int,
        default=256,
        help="Resize to NxN to mirror servoL collection; 0 disables resize.",
    )
    parser.add_argument(
        "--disable_global_time",
        action="store_true",
        help="Do not enable RealSense global_time_enabled.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV output path. Use 'auto' to write under /tmp.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = capture(args)
    print_report(records)
    if args.csv:
        if args.csv == "auto":
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"/tmp/servoL_camera_timestamp_delta_{stamp}.csv"
        else:
            csv_path = args.csv
        write_csv(csv_path, records)


if __name__ == "__main__":
    main()
