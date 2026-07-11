#!/usr/bin/env bash
# Run the RealSense screen-flash latency measurement.
#
# Usage:
#   bash run_realsense_screen_flash_latency.sh
#   bash run_realsense_screen_flash_latency.sh --duration_s 30 --windowed
#   PYTHON_BIN=/path/to/python bash run_realsense_screen_flash_latency.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
measure_script="${script_dir}/measure_realsense_screen_flash_latency.py"

# Measurement settings. Leave serial empty to use the first RealSense device.
serial=""
width="640"
height="480"
fps="30"
duration_s="20"
camera_warmup_s="1"
countdown_s="3"
flash_hz="2"
roi_fraction="0.6"

# Leave manual_exposure_us empty to keep automatic exposure enabled.
# If brightness detection is unstable, try a value such as 5000 or 10000.
manual_exposure_us=""
disable_auto_white_balance="false"

# Each run writes four CSV files with this timestamped name prefix.
output_dir="${script_dir}/realsense_latency_results"
timestamp="$(date +%Y%m%d_%H%M%S)"
csv_base="${output_dir}/screen_flash_latency_${timestamp}.csv"

# Use Python from the currently active environment. Override it with
# PYTHON_BIN when a different interpreter is needed.
python_bin="${PYTHON_BIN:-python}"

if [[ ! -f "${measure_script}" ]]; then
  echo "ERROR: measurement script not found: ${measure_script}" >&2
  exit 1
fi

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter was not found: ${python_bin}" >&2
  echo "Set PYTHON_BIN to an environment containing cv2, numpy, and pyrealsense2." >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "ERROR: no graphical display was detected." >&2
  echo "Run this script from a graphical desktop session." >&2
  exit 1
fi

mkdir -p "${output_dir}"

args=(
  --width "${width}"
  --height "${height}"
  --fps "${fps}"
  --duration_s "${duration_s}"
  --camera_warmup_s "${camera_warmup_s}"
  --countdown_s "${countdown_s}"
  --flash_hz "${flash_hz}"
  --roi_fraction "${roi_fraction}"
  --csv "${csv_base}"
)

if [[ -n "${serial}" ]]; then
  args+=(--serial "${serial}")
fi

if [[ -n "${manual_exposure_us}" ]]; then
  args+=(--manual_exposure_us "${manual_exposure_us}")
fi

if [[ "${disable_auto_white_balance}" == "true" ]]; then
  args+=(--disable_auto_white_balance)
fi

echo "Point the RealSense color camera at this monitor."
echo "A black/white flashing window will open after the countdown; press Esc to stop."
echo "CSV output prefix: ${csv_base}"
echo

# Extra command-line arguments are appended last, so argparse options supplied
# by the caller can override the defaults above.
exec "${python_bin}" "${measure_script}" "${args[@]}" "$@"
