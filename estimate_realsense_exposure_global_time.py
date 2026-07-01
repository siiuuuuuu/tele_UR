#!/usr/bin/env python3
"""Estimate true RealSense exposure timestamps in host monotonic time.

This is an offline validation script. It does not modify the current data
collection path.

Goal:
    SENSOR_TIMESTAMP (camera hardware clock, exposure midpoint)
        -> host monotonic_ns time axis

Method:
    1. Disable RealSense global time and measure the fixed coordinate offset
       between frame.get_timestamp() and SENSOR_TIMESTAMP in the camera hardware
       clock. This is diagnostic only; it is not treated as USB/queue delay.
    2. Enable RealSense global time and fit an affine map from SENSOR_TIMESTAMP
       to frame.get_timestamp() converted to host monotonic_ns.
    3. Use that fitted map directly for the exposure timestamp. The constant
       hardware-clock coordinate offset is absorbed by the affine intercept.
"""

import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
import math
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


def is_global_domain(domain_name):
    return "global" in str(domain_name).lower() or "system" in str(domain_name).lower()


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


def summarize(name, values, unit=""):
    values = [float(v) for v in values if math.isfinite(float(v))]
    suffix = f" ({unit})" if unit else ""
    if not values:
        print(f"{name}{suffix}: no samples")
        return
    print(f"{name}{suffix}:")
    print(f"  count: {len(values)}")
    print(f"  min:   {min(values):.6f}")
    print(f"  p05:   {percentile(values, 5):.6f}")
    print(f"  p50:   {percentile(values, 50):.6f}")
    print(f"  mean:  {statistics.fmean(values):.6f}")
    print(f"  p95:   {percentile(values, 95):.6f}")
    print(f"  max:   {max(values):.6f}")
    if len(values) > 1:
        print(f"  std:   {statistics.stdev(values):.6f}")


def list_devices():
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        devices.append((serial, name))
    devices.sort()
    return devices


def set_global_time(profile, enabled):
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


def print_global_time_status(changed, enabled):
    for sensor_name, supported in changed:
        if not supported:
            status = "not supported"
        else:
            status = "enabled" if enabled else "disabled"
        print(f"global_time_enabled on {sensor_name}: {status}")


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


def start_pipeline(serial, args, global_time_enabled):
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
    changed = set_global_time(profile, enabled=global_time_enabled)
    print_global_time_status(changed, enabled=global_time_enabled)
    time.sleep(args.settle_ms / 1000.0)
    return pipeline


def emulate_collection_work(frame, cv2, target_size):
    image = np.asanyarray(frame.get_data(), dtype=np.uint8)
    if target_size > 0:
        image = cv2.resize(
            image,
            (target_size, target_size),
            interpolation=cv2.INTER_LINEAR,
        )
    return image


def capture_hardware_offset(serial, args):
    cv2 = maybe_load_cv2(args.target_size)
    print()
    print("Phase 1: measuring hardware event offset")
    pipeline = start_pipeline(serial, args, global_time_enabled=False)
    records = []
    try:
        total = args.warmup + args.frames
        for idx in range(total):
            frames = pipeline.wait_for_frames(args.timeout_ms)
            color = frames.get_color_frame()
            if not color:
                continue
            _ = emulate_collection_work(color, cv2, args.target_size)
            if idx < args.warmup:
                continue

            sensor_ts_us = maybe_get_metadata(color, "sensor_timestamp")
            if sensor_ts_us is None:
                continue
            frame_ts_ms = float(color.get_timestamp())
            domain = enum_name(color.get_frame_timestamp_domain())
            records.append(
                {
                    "frame_number": int(color.get_frame_number()),
                    "timestamp_domain": domain,
                    "sensor_timestamp_us": float(sensor_ts_us),
                    "frame_timestamp_ms": frame_ts_ms,
                    "frame_minus_sensor_ms": (
                        frame_ts_ms - float(sensor_ts_us) / 1000.0
                    ),
                }
            )
    finally:
        pipeline.stop()
    return records


def capture_global_fit_samples(serial, args):
    cv2 = maybe_load_cv2(args.target_size)
    print()
    print("Phase 2: fitting SENSOR_TIMESTAMP -> host monotonic_ns")
    pipeline = start_pipeline(serial, args, global_time_enabled=True)
    records = []
    try:
        total = args.warmup + args.frames
        for idx in range(total):
            frames = pipeline.wait_for_frames(args.timeout_ms)
            wait_return_epoch_ns = time.time_ns()
            wait_return_mono_ns = time.monotonic_ns()
            color = frames.get_color_frame()
            if not color:
                continue

            frame_ts_ms = float(color.get_timestamp())
            domain = enum_name(color.get_frame_timestamp_domain())
            sensor_ts_us = maybe_get_metadata(color, "sensor_timestamp")
            actual_exposure_us = maybe_get_metadata(color, "actual_exposure")

            _ = emulate_collection_work(color, cv2, args.target_size)
            collection_anchor_mono_ns = time.monotonic_ns()

            if idx < args.warmup:
                continue
            if sensor_ts_us is None:
                continue

            epoch_minus_mono_ns = wait_return_epoch_ns - wait_return_mono_ns
            frame_global_mono_ns = (
                int(round(frame_ts_ms * 1e6)) - epoch_minus_mono_ns
                if is_global_domain(domain)
                else None
            )
            records.append(
                {
                    "frame_number": int(color.get_frame_number()),
                    "timestamp_domain": domain,
                    "sensor_timestamp_us": float(sensor_ts_us),
                    "actual_exposure_us": (
                        float(actual_exposure_us)
                        if actual_exposure_us is not None
                        else float("nan")
                    ),
                    "frame_global_timestamp_ms": frame_ts_ms,
                    "frame_global_mono_ns": frame_global_mono_ns,
                    "wait_return_mono_ns": wait_return_mono_ns,
                    "collection_anchor_mono_ns": collection_anchor_mono_ns,
                }
            )
    finally:
        pipeline.stop()
    return records


def fit_affine_sensor_to_frame_global(records):
    valid = [
        row
        for row in records
        if row["frame_global_mono_ns"] is not None
        and math.isfinite(float(row["sensor_timestamp_us"]))
    ]
    if len(valid) < 2:
        raise RuntimeError(
            "not enough global_time samples; check that timestamp_domain is global_time"
        )

    x0 = float(valid[0]["sensor_timestamp_us"])
    y0 = float(valid[0]["frame_global_mono_ns"])
    x = np.asarray([float(row["sensor_timestamp_us"]) - x0 for row in valid])
    y = np.asarray([float(row["frame_global_mono_ns"]) - y0 for row in valid])
    slope_ns_per_us, intercept_rel_ns = np.polyfit(x, y, 1)
    intercept_ns = y0 + intercept_rel_ns - slope_ns_per_us * x0

    predicted = slope_ns_per_us * np.asarray(
        [float(row["sensor_timestamp_us"]) for row in valid]
    ) + intercept_ns
    residual_ms = (
        np.asarray([float(row["frame_global_mono_ns"]) for row in valid])
        - predicted
    ) / 1e6
    return float(slope_ns_per_us), float(intercept_ns), residual_ms.tolist()


def apply_exposure_mapping(records, slope_ns_per_us, exposure_intercept_ns):
    previous = None
    for row in records:
        sensor_us = float(row["sensor_timestamp_us"])
        exposure_mono_ns = slope_ns_per_us * sensor_us + exposure_intercept_ns
        row["estimated_frame_event_mono_ns"] = int(round(exposure_mono_ns))
        row["estimated_exposure_mono_ns"] = int(round(exposure_mono_ns))
        row["collection_anchor_minus_exposure_ms"] = (
            int(row["collection_anchor_mono_ns"]) - exposure_mono_ns
        ) / 1e6
        row["wait_return_minus_exposure_ms"] = (
            int(row["wait_return_mono_ns"]) - exposure_mono_ns
        ) / 1e6
        if row["frame_global_mono_ns"] is None:
            row["fit_residual_ms"] = float("nan")
        else:
            row["fit_residual_ms"] = (
                int(row["frame_global_mono_ns"]) - exposure_mono_ns
            ) / 1e6
        if previous is None:
            row["exposure_interval_ms"] = float("nan")
        else:
            row["exposure_interval_ms"] = (
                exposure_mono_ns - float(previous["estimated_exposure_mono_ns"])
            ) / 1e6
        previous = row
    return exposure_intercept_ns


def rolling_offset_check(records, window):
    valid = [
        row
        for row in records
        if row["frame_global_mono_ns"] is not None
        and math.isfinite(float(row["sensor_timestamp_us"]))
    ]
    offsets = deque(maxlen=max(2, int(window)))
    estimates = []
    for row in valid:
        offset = int(row["frame_global_mono_ns"]) - float(row["sensor_timestamp_us"]) * 1000.0
        offsets.append(offset)
        estimates.append(statistics.median(offsets))
    return estimates


def write_csv(path, hardware_records, global_records):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    hw_path = path.with_name(path.stem + "_hardware_offset" + path.suffix)
    with hw_path.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "frame_number",
            "timestamp_domain",
            "sensor_timestamp_us",
            "frame_timestamp_ms",
            "frame_minus_sensor_ms",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hardware_records)
    print(f"Hardware offset CSV written to: {hw_path}")

    global_path = path.with_name(path.stem + "_global_fit" + path.suffix)
    with global_path.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "frame_number",
            "timestamp_domain",
            "sensor_timestamp_us",
            "actual_exposure_us",
            "frame_global_timestamp_ms",
            "frame_global_mono_ns",
            "wait_return_mono_ns",
            "collection_anchor_mono_ns",
            "estimated_frame_event_mono_ns",
            "estimated_exposure_mono_ns",
            "wait_return_minus_exposure_ms",
            "collection_anchor_minus_exposure_ms",
            "exposure_interval_ms",
            "fit_residual_ms",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(global_records)
    print(f"Global fit CSV written to: {global_path}")


def print_report(hardware_records, global_records, slope, exposure_intercept, residuals):
    print()
    print("=== Hardware timestamp offset ===")
    domains = sorted({row["timestamp_domain"] for row in hardware_records})
    print(f"Timestamp domains: {', '.join(domains) if domains else 'none'}")
    hw_offsets_ms = [row["frame_minus_sensor_ms"] for row in hardware_records]
    summarize("frame.get_timestamp - SENSOR_TIMESTAMP", hw_offsets_ms, "ms")

    print()
    print("=== Global fit ===")
    domains = sorted({row["timestamp_domain"] for row in global_records})
    print(f"Timestamp domains: {', '.join(domains) if domains else 'none'}")
    print(f"slope: {slope:.9f} ns/us")
    print(f"slope error from 1000 ns/us: {(slope / 1000.0 - 1.0) * 1e6:.3f} ppm")
    print(f"exposure intercept:    {exposure_intercept:.3f} ns")
    print("hardware offset usage: diagnostic only, not subtracted")
    summarize("fit residual", residuals, "ms")

    print()
    print("=== Exposure-time validation ===")
    summarize(
        "collection_anchor - estimated_exposure",
        [row["collection_anchor_minus_exposure_ms"] for row in global_records],
        "ms",
    )
    print()
    summarize(
        "wait_for_frames_return - estimated_exposure",
        [row["wait_return_minus_exposure_ms"] for row in global_records],
        "ms",
    )
    print()
    summarize(
        "estimated exposure interval",
        [row["exposure_interval_ms"] for row in global_records],
        "ms",
    )
    print()
    summarize(
        "actual exposure metadata",
        [row["actual_exposure_us"] for row in global_records],
        "us",
    )

    print()
    print("Use this mapping for this camera/run:")
    print(
        "  t_exposure_mono_ns = "
        f"{slope:.9f} * SENSOR_TIMESTAMP_us + {exposure_intercept:.3f}"
    )
    print()
    print("For current servoL-style host anchoring, the validation value to watch is:")
    print(
        "  collection_anchor - estimated_exposure "
        f"p50 ~= {percentile([row['collection_anchor_minus_exposure_ms'] for row in global_records], 50):.3f} ms"
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
        "--csv",
        type=str,
        default=None,
        help="Optional CSV output path. Use 'auto' to write under /tmp.",
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

    hardware_records = capture_hardware_offset(serial, args)
    hw_offsets_ms = [row["frame_minus_sensor_ms"] for row in hardware_records]
    if not hw_offsets_ms:
        raise RuntimeError("SENSOR_TIMESTAMP metadata was not available in hardware phase")
    event_offset_ms = percentile(hw_offsets_ms, 50)

    global_records = capture_global_fit_samples(serial, args)
    slope, intercept, residuals = fit_affine_sensor_to_frame_global(global_records)
    exposure_intercept = apply_exposure_mapping(global_records, slope, intercept)

    print_report(
        hardware_records,
        global_records,
        slope,
        exposure_intercept,
        residuals,
    )

    if args.csv:
        if args.csv == "auto":
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"/tmp/realsense_exposure_global_time_{stamp}.csv"
        else:
            csv_path = args.csv
        write_csv(csv_path, hardware_records, global_records)


if __name__ == "__main__":
    main()
