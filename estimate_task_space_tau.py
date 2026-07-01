#!/usr/bin/env python3
"""Estimate first-order task-space tracking parameters from HDF5 demos.

The fitted model is:

    q[k + 1] = q[k] + beta * (u[k] - q[k])
    a = 1 - beta = exp(-dt / tau)

where u is action xyz and q is observed TCP xyz.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np


TIMESTAMP_DT_KEYS = (
    "t_anchor_ns",
    "t_front_camera_host_ns",
    "t_camera_read_ns",
    "t_robot_obs_host_ns",
    "t_record_end_ns",
)


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


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def collect_h5_files(paths, pattern, recursive):
    files = []
    for item in paths:
        path = Path(os.path.expanduser(item))
        if path.is_file():
            files.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"path does not exist: {path}")
        globber = path.rglob if recursive else path.glob
        files.extend(sorted(globber(pattern)))

    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def finite_positive_deltas_ns(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        return None
    deltas = np.diff(values) / 1e9
    return deltas[np.isfinite(deltas) & (deltas > 0)]


def infer_dt_seconds(h5_file, fallback_dt=None):
    if "timestamps" in h5_file:
        timestamps = h5_file["timestamps"]
        for key in TIMESTAMP_DT_KEYS:
            if key not in timestamps:
                continue
            deltas = finite_positive_deltas_ns(timestamps[key][:])
            if deltas is not None and deltas.size > 0:
                return float(np.median(deltas)), key, deltas

    if fallback_dt is not None:
        return float(fallback_dt), "fallback", None
    raise ValueError(
        "could not infer dt from timestamps; pass --dt for timestamp-less files"
    )


def load_demo(path, fallback_dt=None):
    with h5py.File(path, "r") as h5_file:
        if "action" not in h5_file:
            raise ValueError("missing dataset: action")
        if "env_qpos_proprioception" not in h5_file:
            raise ValueError("missing dataset: env_qpos_proprioception")

        action = np.asarray(h5_file["action"][:], dtype=np.float64)
        state = np.asarray(h5_file["env_qpos_proprioception"][:], dtype=np.float64)
        dt, dt_source, dt_deltas = infer_dt_seconds(h5_file, fallback_dt=fallback_dt)

    if action.ndim != 2 or action.shape[1] < 3:
        raise ValueError(f"expected action shape [T, >=3], got {action.shape}")
    if state.ndim != 2 or state.shape[1] < 12:
        raise ValueError(f"expected state shape [T, >=12], got {state.shape}")

    length = min(action.shape[0], state.shape[0])
    if length < 3:
        raise ValueError(f"need at least 3 frames, got {length}")

    action = action[:length]
    state = state[:length]
    if not np.all(np.isfinite(action[:, :3])):
        raise ValueError("action xyz contains NaN or infinite values")
    if not np.all(np.isfinite(state[:, -6:-3])):
        raise ValueError("state TCP xyz contains NaN or infinite values")

    return {
        "path": str(path),
        "action_xyz": action[:, :3],
        "state_xyz": state[:, -6:-3],
        "dt": dt,
        "dt_source": dt_source,
        "dt_deltas": dt_deltas,
        "frames": length,
    }


def beta_to_tau(beta, dt):
    a = 1.0 - beta
    if not (0.0 < a < 1.0):
        return a, np.nan
    return a, float(-dt / np.log(a))


def fit_first_order(action_xyz, state_xyz, dt, min_gap=0.0):
    q0 = state_xyz[:-1]
    q1 = state_xyz[1:]
    u0 = action_xyz[:-1]

    gap = np.linalg.norm(u0 - q0, axis=1)
    mask = np.isfinite(gap) & (gap >= min_gap)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"not enough valid samples after --min_gap={min_gap}: "
            f"{np.count_nonzero(mask)}"
        )

    q0 = q0[mask]
    q1 = q1[mask]
    u0 = u0[mask]
    d = u0 - q0
    y = q1 - q0
    denom = float(np.sum(d * d))
    if denom <= 0.0:
        raise ValueError("zero command-state excitation; cannot fit beta")

    beta = float(np.sum(d * y) / denom)
    a, tau = beta_to_tau(beta, dt)
    pred = q0 + beta * d
    residual = q1 - pred
    rmse_xyz = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    rmse_axis = np.sqrt(np.mean(residual * residual, axis=0))

    axis = []
    for idx, name in enumerate(("x", "y", "z")):
        denom_axis = float(np.sum(d[:, idx] * d[:, idx]))
        if denom_axis <= 0.0:
            beta_axis = np.nan
            a_axis = np.nan
            tau_axis = np.nan
            rmse = np.nan
        else:
            beta_axis = float(np.sum(d[:, idx] * y[:, idx]) / denom_axis)
            a_axis, tau_axis = beta_to_tau(beta_axis, dt)
            pred_axis = q0[:, idx] + beta_axis * d[:, idx]
            rmse = float(np.sqrt(np.mean((q1[:, idx] - pred_axis) ** 2)))
        axis.append(
            {
                "axis": name,
                "beta": beta_axis,
                "a": a_axis,
                "tau_s": tau_axis,
                "rmse_m": rmse,
            }
        )

    return {
        "samples": int(mask.size),
        "used_samples": int(np.count_nonzero(mask)),
        "beta": beta,
        "a": a,
        "tau_s": tau,
        "rmse_xyz_m": rmse_xyz,
        "rmse_axis_m": rmse_axis,
        "axis": axis,
        "mean_gap_m": float(np.mean(gap[mask])),
        "p95_gap_m": float(np.percentile(gap[mask], 95)),
    }


def fit_combined(demos, min_gap=0.0):
    action_parts = []
    state_parts = []
    dts = []
    for demo in demos:
        action_parts.append(demo["action_xyz"])
        state_parts.append(demo["state_xyz"])
        dts.append(demo["dt"])

    # Same-config folders should have nearly identical dt. Using the mean keeps
    # the scalar beta fit simple while still reporting the measured dt spread.
    dt = float(np.mean(dts))
    action_xyz = np.vstack(
        [part[:-1] for part in action_parts]
    )
    state_q0 = np.vstack(
        [part[:-1] for part in state_parts]
    )
    state_q1 = np.vstack(
        [part[1:] for part in state_parts]
    )
    gap = np.linalg.norm(action_xyz - state_q0, axis=1)
    mask = np.isfinite(gap) & (gap >= min_gap)
    if np.count_nonzero(mask) < 2:
        raise ValueError("not enough valid combined samples")
    q0 = state_q0[mask]
    q1 = state_q1[mask]
    u0 = action_xyz[mask]
    d = u0 - q0
    y = q1 - q0
    beta = float(np.sum(d * y) / np.sum(d * d))
    a, tau = beta_to_tau(beta, dt)
    pred = q0 + beta * d
    residual = q1 - pred
    rmse_xyz = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    rmse_axis = np.sqrt(np.mean(residual * residual, axis=0))

    axis = []
    for idx, name in enumerate(("x", "y", "z")):
        denom_axis = float(np.sum(d[:, idx] * d[:, idx]))
        beta_axis = float(np.sum(d[:, idx] * y[:, idx]) / denom_axis)
        a_axis, tau_axis = beta_to_tau(beta_axis, dt)
        pred_axis = q0[:, idx] + beta_axis * d[:, idx]
        rmse = float(np.sqrt(np.mean((q1[:, idx] - pred_axis) ** 2)))
        axis.append(
            {
                "axis": name,
                "beta": beta_axis,
                "a": a_axis,
                "tau_s": tau_axis,
                "rmse_m": rmse,
            }
        )

    return {
        "samples": int(mask.size),
        "used_samples": int(np.count_nonzero(mask)),
        "dt_s": dt,
        "dt_min_s": float(np.min(dts)),
        "dt_max_s": float(np.max(dts)),
        "beta": beta,
        "a": a,
        "tau_s": tau,
        "rmse_xyz_m": rmse_xyz,
        "rmse_axis_m": rmse_axis,
        "axis": axis,
        "mean_gap_m": float(np.mean(gap[mask])),
        "p95_gap_m": float(np.percentile(gap[mask], 95)),
    }


def lag_scan_for_demo(demo, max_lag):
    action_xyz = demo["action_xyz"]
    state_xyz = demo["state_xyz"]
    rows = []
    for lag in range(max_lag + 1):
        count = len(state_xyz) - lag
        if count <= 1:
            continue
        error = action_xyz[:count] - state_xyz[lag: lag + count]
        pos = np.linalg.norm(error, axis=1)
        rows.append(
            {
                "lag": lag,
                "lag_time_s": lag * demo["dt"],
                "mean_error_m": float(np.mean(pos)),
                "p95_error_m": float(np.percentile(pos, 95)),
                "max_error_m": float(np.max(pos)),
                "samples": int(count),
            }
        )
    return rows


def combined_lag_scan(demos, max_lag):
    rows = []
    dt = float(np.mean([demo["dt"] for demo in demos]))
    for lag in range(max_lag + 1):
        parts = []
        count = 0
        for demo in demos:
            action_xyz = demo["action_xyz"]
            state_xyz = demo["state_xyz"]
            n = len(state_xyz) - lag
            if n <= 1:
                continue
            error = action_xyz[:n] - state_xyz[lag: lag + n]
            parts.append(np.linalg.norm(error, axis=1))
            count += n
        if not parts:
            continue
        pos = np.concatenate(parts)
        rows.append(
            {
                "lag": lag,
                "lag_time_s": lag * dt,
                "mean_error_m": float(np.mean(pos)),
                "p95_error_m": float(np.percentile(pos, 95)),
                "max_error_m": float(np.max(pos)),
                "samples": int(count),
            }
        )
    return rows


def dt_stats(deltas):
    if deltas is None or len(deltas) == 0:
        return None
    return {
        "mean_s": float(np.mean(deltas)),
        "median_s": float(np.median(deltas)),
        "p05_s": float(np.percentile(deltas, 5)),
        "p95_s": float(np.percentile(deltas, 95)),
        "min_s": float(np.min(deltas)),
        "max_s": float(np.max(deltas)),
    }


def fmt_ms(seconds):
    return f"{seconds * 1000.0:.3f}"


def fmt_mm(meters):
    return f"{meters * 1000.0:.3f}"


def print_axis(axis_rows, indent="    "):
    for row in axis_rows:
        print(
            f"{indent}{row['axis']}: "
            f"beta={row['beta']:.6f}, "
            f"a={row['a']:.6f}, "
            f"tau={row['tau_s']:.6f}s, "
            f"rmse={fmt_mm(row['rmse_m'])}mm"
        )


def write_summary_csv(path, file_results, combined):
    if not path:
        return
    path = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fields = [
        "name",
        "frames",
        "dt_ms",
        "hz",
        "used_samples",
        "beta",
        "a",
        "tau_s",
        "rmse_xyz_mm",
        "tau_x_s",
        "tau_y_s",
        "tau_z_s",
        "best_lag",
        "best_lag_ms",
        "best_lag_mean_error_mm",
        "best_lag_p95_error_mm",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for result in file_results:
            fit = result["fit"]
            best_lag = result["best_lag"]
            axis = {row["axis"]: row for row in fit["axis"]}
            writer.writerow(
                {
                    "name": result["name"],
                    "frames": result["frames"],
                    "dt_ms": fit["dt_s"] * 1000.0,
                    "hz": 1.0 / fit["dt_s"],
                    "used_samples": fit["used_samples"],
                    "beta": fit["beta"],
                    "a": fit["a"],
                    "tau_s": fit["tau_s"],
                    "rmse_xyz_mm": fit["rmse_xyz_m"] * 1000.0,
                    "tau_x_s": axis["x"]["tau_s"],
                    "tau_y_s": axis["y"]["tau_s"],
                    "tau_z_s": axis["z"]["tau_s"],
                    "best_lag": best_lag["lag"],
                    "best_lag_ms": best_lag["lag_time_s"] * 1000.0,
                    "best_lag_mean_error_mm": best_lag["mean_error_m"] * 1000.0,
                    "best_lag_p95_error_mm": best_lag["p95_error_m"] * 1000.0,
                }
            )

        fit = combined["fit"]
        best_lag = combined["best_lag"]
        axis = {row["axis"]: row for row in fit["axis"]}
        writer.writerow(
            {
                "name": "combined",
                "frames": combined["frames"],
                "dt_ms": fit["dt_s"] * 1000.0,
                "hz": 1.0 / fit["dt_s"],
                "used_samples": fit["used_samples"],
                "beta": fit["beta"],
                "a": fit["a"],
                "tau_s": fit["tau_s"],
                "rmse_xyz_mm": fit["rmse_xyz_m"] * 1000.0,
                "tau_x_s": axis["x"]["tau_s"],
                "tau_y_s": axis["y"]["tau_s"],
                "tau_z_s": axis["z"]["tau_s"],
                "best_lag": best_lag["lag"],
                "best_lag_ms": best_lag["lag_time_s"] * 1000.0,
                "best_lag_mean_error_mm": best_lag["mean_error_m"] * 1000.0,
                "best_lag_p95_error_mm": best_lag["p95_error_m"] * 1000.0,
            }
        )
    print(f"\nSaved summary CSV: {path}")


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate q[k+1] = q[k] + beta * (action_xyz[k] - q[k]) "
            "and tau for task-space MPC."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="H5 file(s) or folder(s) containing demos.",
    )
    parser.add_argument(
        "--pattern",
        default="demo_*.h5",
        help="File glob used when a path is a folder.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search folders recursively.",
    )
    parser.add_argument(
        "--dt",
        type=positive_float,
        default=None,
        help="Fallback dt in seconds if timestamps are missing.",
    )
    parser.add_argument(
        "--min_gap",
        type=nonnegative_float,
        default=0.0,
        help="Ignore samples where ||action_xyz - state_xyz|| is below this value.",
    )
    parser.add_argument(
        "--max_lag",
        type=positive_int,
        default=30,
        help="Maximum lag, in frames, for action/state alignment scan.",
    )
    parser.add_argument(
        "--preview_time",
        type=positive_float,
        default=0.25,
        help="Preview time used to suggest mpc_horizon.",
    )
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Optional path to save per-file and combined fit summary.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Optional path to save detailed results as JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first invalid file instead of skipping it.",
    )
    return parser


def main(args):
    files = collect_h5_files(args.paths, args.pattern, args.recursive)
    if not files:
        raise SystemExit("No H5 files found.")

    demos = []
    skipped = []
    for path in files:
        try:
            demos.append(load_demo(path, fallback_dt=args.dt))
        except Exception as exc:
            if args.strict:
                raise
            skipped.append((str(path), str(exc)))

    if not demos:
        raise SystemExit("No valid demo files found.")

    file_results = []
    print("Per-file estimates:")
    for demo in demos:
        fit = fit_first_order(
            demo["action_xyz"],
            demo["state_xyz"],
            demo["dt"],
            min_gap=args.min_gap,
        )
        fit["dt_s"] = demo["dt"]
        lags = lag_scan_for_demo(demo, args.max_lag)
        best_lag = min(lags, key=lambda row: row["mean_error_m"])
        stats = dt_stats(demo["dt_deltas"])
        name = os.path.basename(demo["path"])
        file_results.append(
            {
                "name": name,
                "path": demo["path"],
                "frames": demo["frames"],
                "dt_source": demo["dt_source"],
                "dt_stats": stats,
                "fit": fit,
                "lag_scan": lags,
                "best_lag": best_lag,
            }
        )

        hz = 1.0 / demo["dt"]
        print(
            f"\n{name}: frames={demo['frames']}, "
            f"dt={fmt_ms(demo['dt'])}ms ({hz:.2f}Hz), "
            f"dt_source={demo['dt_source']}"
        )
        if stats is not None:
            print(
                "  dt stats: "
                f"mean={fmt_ms(stats['mean_s'])}ms, "
                f"p50={fmt_ms(stats['median_s'])}ms, "
                f"p05={fmt_ms(stats['p05_s'])}ms, "
                f"p95={fmt_ms(stats['p95_s'])}ms"
            )
        print(
            "  fit: "
            f"beta={fit['beta']:.6f}, "
            f"a={fit['a']:.6f}, "
            f"tau={fit['tau_s']:.6f}s, "
            f"rmse_xyz={fmt_mm(fit['rmse_xyz_m'])}mm, "
            f"used={fit['used_samples']}/{fit['samples']}"
        )
        print_axis(fit["axis"])
        print(
            "  best lag: "
            f"{best_lag['lag']} frame(s), "
            f"{fmt_ms(best_lag['lag_time_s'])}ms, "
            f"mean={fmt_mm(best_lag['mean_error_m'])}mm, "
            f"p95={fmt_mm(best_lag['p95_error_m'])}mm"
        )

    combined_fit = fit_combined(demos, min_gap=args.min_gap)
    combined_lags = combined_lag_scan(demos, args.max_lag)
    combined_best_lag = min(combined_lags, key=lambda row: row["mean_error_m"])
    combined = {
        "frames": int(sum(demo["frames"] for demo in demos)),
        "fit": combined_fit,
        "lag_scan": combined_lags,
        "best_lag": combined_best_lag,
    }

    print("\nCombined estimate:")
    print(
        f"  files={len(demos)}, frames={combined['frames']}, "
        f"mean_dt={fmt_ms(combined_fit['dt_s'])}ms "
        f"({1.0 / combined_fit['dt_s']:.2f}Hz)"
    )
    print(
        "  fit: "
        f"beta={combined_fit['beta']:.6f}, "
        f"a={combined_fit['a']:.6f}, "
        f"tau={combined_fit['tau_s']:.6f}s, "
        f"rmse_xyz={fmt_mm(combined_fit['rmse_xyz_m'])}mm, "
        f"used={combined_fit['used_samples']}/{combined_fit['samples']}"
    )
    print_axis(combined_fit["axis"])
    print(
        "  best lag: "
        f"{combined_best_lag['lag']} frame(s), "
        f"{fmt_ms(combined_best_lag['lag_time_s'])}ms, "
        f"mean={fmt_mm(combined_best_lag['mean_error_m'])}mm, "
        f"p95={fmt_mm(combined_best_lag['p95_error_m'])}mm"
    )

    suggested_horizon = max(1, int(round(args.preview_time / combined_fit["dt_s"])))
    print("\nSuggested MPC settings:")
    print(f"  dt=\"{combined_fit['dt_s']:.6f}\"")
    print(f"  mpc_tau=\"{combined_fit['tau_s']:.3f}\"")
    print(
        f"  mpc_horizon=\"{suggested_horizon}\"  "
        f"# about {suggested_horizon * combined_fit['dt_s'] * 1000.0:.0f}ms preview"
    )

    if skipped:
        print("\nSkipped files:")
        for path, reason in skipped:
            print(f"  {path}: {reason}")

    write_summary_csv(args.summary_csv, file_results, combined)
    if args.json_path:
        json_path = os.path.abspath(os.path.expanduser(args.json_path))
        json_dir = os.path.dirname(json_path)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        payload = {
            "files": file_results,
            "combined": combined,
            "skipped": skipped,
            "settings": {
                "pattern": args.pattern,
                "recursive": args.recursive,
                "min_gap": args.min_gap,
                "max_lag": args.max_lag,
                "preview_time": args.preview_time,
            },
        }
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(to_jsonable(payload), json_file, indent=2)
        print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
