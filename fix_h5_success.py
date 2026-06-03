import argparse
import glob
import os
from typing import Iterable, List

import h5py
import numpy as np


TRUE_WORDS = {"1", "true", "t", "yes", "y", "success", "ok"}
FALSE_WORDS = {"0", "false", "f", "no", "n", "fail", "failed"}


def parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in TRUE_WORDS:
        return True
    if value in FALSE_WORDS:
        return False
    raise ValueError(f"无法识别的success值: {text}")


def normalize_attr_value(raw):
    if raw is None:
        return None

    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return None
        if raw.size == 1:
            raw = raw.reshape(-1)[0]
        else:
            return raw.tolist()

    if isinstance(raw, (tuple, list)):
        if len(raw) == 0:
            return None
        if len(raw) == 1:
            raw = raw[0]
        else:
            return list(raw)

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")

    return raw


def success_to_text(raw) -> str:
    val = normalize_attr_value(raw)
    if val is None:
        return "MISSING"

    if isinstance(val, str):
        lower = val.strip().lower()
        if lower in TRUE_WORDS:
            return f"True (raw='{val}')"
        if lower in FALSE_WORDS:
            return f"False (raw='{val}')"
        return f"Unknown string (raw='{val}')"

    if isinstance(val, (list, tuple)):
        return f"Composite {val}"

    try:
        return f"{bool(val)} (raw={val})"
    except Exception:
        return f"Unknown (raw={val})"


def collect_files(file_args: List[str], glob_pattern: str) -> List[str]:
    files = []
    if file_args:
        files.extend(file_args)
    if glob_pattern:
        files.extend(glob.glob(glob_pattern, recursive=True))

    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for path in files:
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            seen.add(abs_path)
            unique_files.append(abs_path)

    return unique_files


def update_success(paths: Iterable[str], new_success: bool, dry_run: bool) -> None:
    total = 0
    changed = 0
    skipped = 0

    for path in paths:
        total += 1
        if not os.path.exists(path):
            print(f"[跳过] 文件不存在: {path}")
            skipped += 1
            continue

        try:
            with h5py.File(path, "r+") as f:
                old_raw = f.attrs.get("success")
                old_text = success_to_text(old_raw)

                print(f"\n文件: {path}")
                print(f"  旧 success: {old_text}")
                print(f"  新 success: {new_success}")

                if dry_run:
                    print("  dry-run: 不写入")
                    continue

                f.attrs["success"] = np.bool_(new_success)
                new_raw = f.attrs.get("success")
                print(f"  写入后: {success_to_text(new_raw)}")
                changed += 1

        except Exception as exc:
            print(f"[失败] {path}: {exc}")
            skipped += 1

    print("\n=== 完成 ===")
    print(f"总文件数: {total}")
    print(f"成功写入: {changed}")
    print(f"跳过/失败: {skipped}")

#demo_20260423_120045.h5
def main() -> None:
    parser = argparse.ArgumentParser(description="批量修改H5文件的 success 属性")
    parser.add_argument("--file", nargs="*", default=[], help="一个或多个h5文件路径")
    parser.add_argument("--glob", default="", help="通配符匹配路径，例如 '/path/demo_*.h5'")
    parser.add_argument("--success", required=True, help="目标success值: true/false/1/0")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不实际写入")

    args = parser.parse_args()

    if not args.file and not args.glob:
        raise SystemExit("请至少提供 --file 或 --glob")

    try:
        new_success = parse_bool(args.success)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    files = collect_files(args.file, args.glob)
    if not files:
        raise SystemExit("没有找到任何匹配文件")

    print(f"匹配到 {len(files)} 个文件")
    update_success(files, new_success, args.dry_run)

#有影响的错误轨迹，demo_20260507_114630.h5
if __name__ == "__main__":
    main()
## 1) 先预览（不写盘）
#python fix_h5_success.py --glob '/home/lrz/dp_data/offlineRL_data/task1/demo_*.h5' --success true --dry-run

# 2) 确认后真正写入
#python fix_h5_success.py --glob '/home/lrz/dp_data/offlineRL_data/task1/demo_*.h5' --success true
#demo_20260507_114351.h5
# 3) 只改单个文件
#python fix_h5_success.py --file /home/lrz/dp_data/offlineRL_data/task1/demo_20260322_111958.h5 --success true
#demo_20260327_104534.h5,demo_20260327_110454.h5,demo_20260331_110750.h5
#demo_20260331_110948.h5,demo_20260331_111839.h5,demo_20260331_111504.h5,demo_20260331_113206.h5