#!/usr/bin/env python3
"""Benchmark OpenVR Vive tracker pose polling from this project."""

import argparse
import os
import time

import numpy as np


DEFAULT_INITIAL_POSE = [0.248, 0.1212, 0.3978, 1.16, 1.25, 1.28]
STEAM_RUNTIME_LIB_DIRS = (
    "/home/lrz/.steam/debian-installation/ubuntu12_32/steam-runtime/usr/lib/x86_64-linux-gnu",
    "/home/lrz/.steam/debian-installation/steamapps/common/SteamLinuxRuntime_sniper/var/tmp-RDA0Q3/usr/lib/x86_64-linux-gnu",
)


def prepend_runtime_libs():
    existing = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    paths = [path for path in STEAM_RUNTIME_LIB_DIRS if os.path.isdir(path)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(paths + existing)


def pose_to_tuple(pose_mat):
    return tuple(float(pose_mat[row][col]) for row in range(3) for col in range(4))


def summarize(name, calls, valid, changed, latencies_ns, elapsed_s):
    lat_us = np.asarray(latencies_ns, dtype=np.float64) / 1000.0
    print(f"\n{name}")
    print(f"  elapsed_s: {elapsed_s:.3f}")
    print(f"  calls: {calls}")
    print(f"  valid: {valid}")
    print(f"  none: {calls - valid}")
    print(f"  call_hz: {calls / elapsed_s:.1f}")
    print(f"  valid_hz: {valid / elapsed_s:.1f}")
    print(f"  changed_hz: {changed / elapsed_s:.1f}")
    print(f"  latency_us_mean: {float(np.mean(lat_us)):.2f}")
    print(f"  latency_us_p50: {float(np.percentile(lat_us, 50)):.2f}")
    print(f"  latency_us_p95: {float(np.percentile(lat_us, 95)):.2f}")
    print(f"  latency_us_p99: {float(np.percentile(lat_us, 99)):.2f}")


def summarize_scheduled(name, target_hz, calls, valid, missed, latencies_ns, jitters_us, elapsed_s):
    lat_us = np.asarray(latencies_ns, dtype=np.float64) / 1000.0
    jitter = np.asarray(jitters_us, dtype=np.float64)
    print(f"\n{name} @ {target_hz:g} Hz")
    print(f"  elapsed_s: {elapsed_s:.3f}")
    print(f"  calls: {calls}")
    print(f"  valid: {valid}")
    print(f"  missed_ticks: {missed}")
    print(f"  achieved_hz: {calls / elapsed_s:.1f}")
    print(f"  valid_hz: {valid / elapsed_s:.1f}")
    print(f"  read_latency_us_p50: {float(np.percentile(lat_us, 50)):.2f}")
    print(f"  read_latency_us_p95: {float(np.percentile(lat_us, 95)):.2f}")
    print(f"  schedule_jitter_us_p50: {float(np.percentile(jitter, 50)):.2f}")
    print(f"  schedule_jitter_us_p95: {float(np.percentile(jitter, 95)):.2f}")


def run_loop(duration_s, read_fn, matrix_fn):
    end_s = time.perf_counter() + duration_s
    calls = 0
    valid = 0
    changed = 0
    previous = None
    latencies_ns = []
    start_s = time.perf_counter()

    while time.perf_counter() < end_s:
        t0 = time.perf_counter_ns()
        result = read_fn()
        t1 = time.perf_counter_ns()
        calls += 1
        latencies_ns.append(t1 - t0)
        if result is None:
            continue
        valid += 1
        current = matrix_fn(result)
        if previous is not None and current != previous:
            changed += 1
        previous = current

    elapsed_s = time.perf_counter() - start_s
    return calls, valid, changed, latencies_ns, elapsed_s


def run_scheduled(duration_s, target_hz, read_fn):
    period_s = 1.0 / target_hz
    end_s = time.perf_counter() + duration_s
    next_tick_s = time.perf_counter()
    calls = 0
    valid = 0
    missed = 0
    latencies_ns = []
    jitters_us = []
    start_s = time.perf_counter()

    while not time.perf_counter() >= end_s:
        now_s = time.perf_counter()
        delay_s = next_tick_s - now_s
        if delay_s > 0:
            time.sleep(delay_s)

        actual_s = time.perf_counter()
        jitters_us.append((actual_s - next_tick_s) * 1e6)
        t0 = time.perf_counter_ns()
        result = read_fn()
        t1 = time.perf_counter_ns()
        calls += 1
        latencies_ns.append(t1 - t0)
        if result is not None:
            valid += 1

        next_tick_s += period_s
        behind_s = time.perf_counter() - next_tick_s
        if behind_s > 0:
            skipped = int(behind_s // period_s) + 1
            missed += skipped
            next_tick_s += skipped * period_s

    elapsed_s = time.perf_counter() - start_s
    return calls, valid, missed, latencies_ns, jitters_us, elapsed_s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="tracker_1")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--initial_pose",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_POSE,
    )
    parser.add_argument(
        "--scheduled_rates",
        type=float,
        nargs="*",
        default=[60.0, 80.0, 90.0, 120.0, 240.0, 500.0, 1000.0],
    )
    args = parser.parse_args()

    prepend_runtime_libs()

    import triad_openvr
    from tools import MATHTOOLS
    from tracker_pose_processor import TrackerPoseProcessor

    v = triad_openvr.triad_openvr()
    for _ in range(20):
        if args.device in v.devices:
            break
        v.poll_vr_events()
        time.sleep(0.1)

    print(f"devices: {sorted(v.devices.keys())}")
    if args.device not in v.devices:
        raise SystemExit(f"device not found: {args.device}")

    tracker_device = v.devices[args.device]
    processor = TrackerPoseProcessor(MATHTOOLS(), args.initial_pose)

    raw = run_loop(
        args.duration,
        tracker_device.get_pose_matrix,
        pose_to_tuple,
    )
    summarize("raw tracker_device.get_pose_matrix()", *raw)

    processed = run_loop(
        args.duration,
        lambda: processor.read_tracker_mat(tracker_device),
        lambda mat: tuple(np.asarray(mat, dtype=np.float64).reshape(-1)),
    )
    summarize("project TrackerPoseProcessor.read_tracker_mat()", *processed)

    for target_hz in args.scheduled_rates:
        result = run_scheduled(
            args.duration,
            target_hz,
            lambda: processor.read_tracker_mat(tracker_device),
        )
        summarize_scheduled("scheduled read_tracker_mat()", target_hz, *result)


if __name__ == "__main__":
    main()
