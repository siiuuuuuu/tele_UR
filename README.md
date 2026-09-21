# tele_UR

[![Project Page](https://img.shields.io/badge/Project-Page-2f80c1?style=flat-square)](https://siiuuuuuu.github.io/DexPIE/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.09615-b31b1b?style=flat-square)](https://arxiv.org/abs/2606.09615)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

Teleoperation data collection code based on UR RTDE `servoL`. The main entry point is `servoL.py`: it uses an OpenVR tracker to control the UR5 TCP pose, maps MANUS glove data to Inspire hand commands, records RealSense images, and saves each demonstration as an HDF5 file.

## 1. Install the `tele` Conda Environment

Python 3.8 is recommended. The existing `tele` environment on this machine uses Python 3.8.20, and the commands below use the dependency versions verified in that environment.

```bash
conda create -n tele python=3.8 -y
conda activate tele

python -m pip install --upgrade pip
python -m pip install \
  numpy==1.23.5 \
  opencv-python==4.6.0.66 \
  pyrealsense2==2.55.1.6486 \
  ur-rtde==1.6.2 \
  openvr==2.5.102 \
  h5py==3.11.0 \
  zarr==2.16.1 \
  pynput==1.8.1 \
  pyserial==3.5 \
  termcolor==2.4.0 \
  tqdm==4.62.3 \
  imageio==2.35.1 \
  pyzmq==27.1.0
```

After installation, run a basic import check:

```bash
conda activate tele

python - <<'PY'
import numpy, cv2, pyrealsense2, rtde_control, rtde_receive
import openvr, h5py, zarr, serial, termcolor, tqdm, imageio, zmq
print("tele core imports ok")
PY
```

`pynput` needs access to the local desktop/X session. It may fail over SSH or in a headless session. Before collection, check it separately:

```bash
python - <<'PY'
from pynput import keyboard
print("pynput ok")
PY
```

## 2. Hardware and System Setup

### Hardware Setup

![Hardware installation and alignment](docs/assets/hardware-installation.png)

As shown above, install the Inspire dexterous hand so that its palm normal points along the negative y-axis.

Before collecting data, make sure the following hardware and services are ready:

- UR5 robot: the code connects to `192.168.3.6` by default. Configure this in `collect_data.sh` and `replay_trajectory.sh`. The current collection frequencies and alignment settings are described below.
- UR controller: Connect the computer to the robot controller with an Ethernet cable, either directly or through a LAN switch. Configure both devices on the same IP subnet, and enable Remote Control/RTDE on the robot.
- Inspire hand: default serial port is `/dev/ttyUSB0`, baud rate `115200`.
- RealSense cameras: First install the [Intel RealSense SDK (librealsense)](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md), including its udev rules. Then connect the two cameras; the code uses `front_cam_idx=0` and `right_cam_idx=1` by default, with device serial numbers sorted before indexing.
- SteamVR: Set up the Vive Tracker following [Software Setup Tutorial - VIVE Tracker setup](https://docs.google.com/document/d/1ANxSA_PctkqFf3xqAkyktgBgDWEbrFK7b1OnJe54ltw/edit?tab=t.0#heading=h.yxlxo67jgfyx).
- MANUS glove: First follow the ManusTele ZMQ/protobuf publisher setup, then start the publisher on `tcp://127.0.0.1:2044`. `servoL.py` subscribes through `SM_inspire_manus_zmq.py` by default.

Set temporary serial permission:

```bash
sudo chmod 666 /dev/ttyUSB0
```

If RealSense devices cannot be enumerated, install librealsense/udev rules first, then reconnect the cameras.

## 3. Runtime Architecture and Time Alignment

Robot control and data recording run at different rates. The current
`collect_data.sh` configuration is:

| Component | Current setting |
| --- | ---: |
| Front camera and HDF5 recording pace | 30 Hz |
| Wrist camera | 60 Hz |
| Vive Tracker sampling | 80 Hz |
| UR `servoL` control | 120 Hz |
| Robot-state sampling | 125 Hz |
| MANUS glove ZMQ receive | Publisher-driven; not rate-limited locally |
| Inspire hand control | 120 Hz |
| Tracker interpolation delay | 12.5 ms |
| Cross-stream alignment tolerance | 25 ms |
| History wait timeout | 3 ms |
| Hand smoother `(omega, damping, input alpha)` | `(30, 0.85, 0.84)` |

The tracker, arm servo, robot-state reader, MANUS receiver, and hand-control
loops run independently from recording. Recording is paced by each new front
camera frame. The `dt` argument is retained for compatibility but no longer
sets the HDF5 sampling frequency.

The MANUS receive rate is determined by the external ZMQ/protobuf publisher,
not by `hand_frequency`. The receiver consumes packets as they arrive and uses
`RCVHWM=1` with `CONFLATE` enabled to retain the latest frame instead of
building a stale queue. `hand_frequency=120` is the separate rate at which the
latest smoothed command is sent to the Inspire hand.

Each recorded sample uses the front-camera image timestamp as `t_anchor`:

1. RealSense global time is enabled, and the color frame's `get_timestamp()`
   value is mapped into the host monotonic clock domain.
2. The front image, nearest wrist image, and nearest robot state are aligned to
   `t_anchor`.
3. Arm and hand actions are aligned to
   `t_action_anchor = t_anchor + action_alignment_offset_frames / front_camera_fps`.
4. A sample is skipped when the wrist image, arm action, robot state, or hand
   action exceeds `alignment_tolerance_ms` from its intended anchor.

The default collection-time `action_alignment_offset_frames` is `0`, so raw
HDF5 files retain actions nearest to the same front-camera anchor. The
additional one-frame compensation used for training is applied during HDF5 to
Zarr+memmap conversion, as described in Section 6.

## 4. Collect Demonstrations

After the UR robot, RealSense cameras, SteamVR, MANUS client, and Inspire hand are ready, run:

```bash
conda activate tele
```

Edit the robot address, workspace limits, initial pose, output directory, hand serial port, and other collection settings at the top of `collect_data.sh`. Then run:

```bash
bash collect_data.sh
```

Keyboard controls:

- Press `s`: start recording the current episode.
- Press `s` again: stop recording the current episode.
- After recording stops, the terminal asks whether to save the data. Enter `y` to save or `n` to discard.
- Press `a`: stop the whole collection loop.

Saved file names use this format:

```text
demo_YYYYMMDD_HHMMSS.h5
```

HDF5 fields:

- `color`: front camera RGB image, default shape `[T, 256, 256, 3]`.
- `wrist_color`: wrist camera RGB image, present when `use_wrist_img` is enabled.
- `env_qpos_proprioception`: robot state, `[6 joint + 6 TCP pose]`, shape `[T, 12]`.
- `action`: action vector, `[absolute target xyz + 6D rotation + 6 hand]`, shape `[T, 15]`.
- `timestamps/`: raw timing and alignment diagnostics, with one value per frame.

The `timestamps/` group contains these categories:

- front/wrist camera reported image time, receive time, frame number, and sequence;
- tracker target, interpolation, arm action, and UR servo times;
- robot observation, MANUS sample, and Inspire hand command times;
- synchronization deltas relative to the observation and action anchors.

Each HDF5 file also records the attributes `action_alignment_policy`,
`action_alignment_offset_frames`, `action_alignment_offset_seconds`,
`front_camera_fps`, and `wrist_camera_fps`.

## 5. Measure RealSense Image-Time Offset

`measure_realsense_screen_flash_latency.py` estimates the offset between the
RealSense-reported timestamp and the image content by filming a black/white
flashing window. Run it from a graphical desktop session:

```bash
conda activate tele
bash run_realsense_screen_flash_latency.sh

# Optional: longer measurement in a window
bash run_realsense_screen_flash_latency.sh --duration_s 30 --windowed
```

Point the RealSense color camera at the flashing window. The launcher writes
frame, display-event, camera-edge, and matched-edge CSV files under
`realsense_latency_results/`. The report separates:

- reported image timestamp minus display-toggle application time;
- Python receive time minus display-toggle application time;
- Python receive time minus the reported image timestamp.

For the current cameras and 30 Hz configuration, the measured image-content
time indicates that the true exposure corresponding to an image is about one
frame earlier than the timestamp currently used for alignment:

```text
t_exposure ~= t_reported - 1 / camera_fps
```

At 30 Hz this is approximately 33.3 ms. The measured exposure duration itself
is small relative to a frame, but that estimate is less reliable because the
screen method also includes monitor scanout, pixel response, threshold/ROI
selection, and other measurement errors. Therefore the repository treats the
one-frame offset as an empirical training-label compensation, not a precise
hardware exposure calibration. Re-run the measurement after changing the
camera model, resolution, frame rate, exposure mode, or display setup.

## 6. Convert HDF5 to Zarr+memmap

For regular teleoperation data, edit the input directory, output directory, and saved observation types at the top of `convert_data.sh`. Then run:

```bash
bash convert_data.sh
```

For data with `success` / `intervention` metadata, or when merging multiple directories, edit the directory list and output settings at the top of `convert_rollout_data.sh`. Then run:

```bash
bash convert_rollout_data.sh
```

Both converters write the large-dataset layout used by DexPIE directly; they
do not create the legacy all-Zarr dataset first:

```text
save_dir/
├── data.zarr/
│   ├── data/        # state, action, optional intervention/cloud
│   └── meta/        # episode_ends, success
├── img.npy          # present when save_img=1
├── wrist_img.npy    # present when save_wrist_img=1
└── depth.npy        # present when save_depth=1
```

The visual `.npy` files are NumPy memmaps, while smaller/nonvisual arrays stay
in `data.zarr`. Conversion scans H5 metadata first and then copies frames in
batches, so it does not accumulate the complete image dataset in RAM. Use
`--batch_size` to change the default batch size of 64 frames.

Training conversion defaults to a one-frame future action label:

```text
observation[i] -> action[i + 1]
```

For rollout data, `intervention[i + 1]` stays aligned with the shifted action.
The final frame of each episode is dropped because it has no `action[i + 1]`.
This compensates the approximately one-frame image-time offset measured in
Section 5. Set `action_offset_frames=0` in the shell script, or pass
`--action_offset_frames 0` directly, to disable the compensation.

There are two distinct offsets:

| Offset | Applied at | Default | Meaning |
| --- | --- | ---: | --- |
| `action_alignment_offset_frames` | HDF5 collection | 0 | Select control history relative to the front-camera anchor |
| `action_offset_frames` | Dataset conversion | 1 | Shift the training action/intervention index into the future |

The `data.zarr` root stores `recorded_action_offset_frames`,
`action_index_offset_frames`, and their sum as `action_offset_frames`. Source
HDF5 files with different recorded offsets cannot be merged without an
explicit override.

To compare other future-label offsets, use:

```bash
# Generate +1, +2, and +3 regular datasets
bash convert_data_action_offset.sh

# Generate only +2
bash convert_data_action_offset.sh 2

# Rollout/expert equivalents
bash convert_rollout_data_action_offset.sh
bash convert_rollout_data_action_offset.sh 2
```

The raw HDF5 `timestamps/` diagnostics are intentionally not copied into the
training dataset. Regular demonstration conversion marks all episodes
successful; rollout conversion reads `success` and `intervention` when present.

The provided shell scripts pass `--overwrite` and replace an existing
`save_dir`. Direct Python invocation refuses to replace an existing output
unless `--overwrite` is supplied.

## 7. Inspect Data and Replay Trajectories

Play the `color` and `wrist_color` camera streams:

```bash
python read.py ~/dp_data/new_task1_expertdata
```

Play rollout data with cameras and `intervention` / `success` status overlays:

```bash
python read_with_intervention.py ~/dp_data/offlineRL_data/new_task1_iter1
```

`read.py` displays both cameras side by side with playback status, frame information, and a progress bar. `read_with_intervention.py` additionally uses green borders for autonomous steps and red borders for intervention steps, displays the current mode and episode success/failure at the top, and shows all intervention segments on the bottom timeline.

Both scripts accept either an individual HDF5 file or a directory. Use `--fps` to change the playback frame rate and `--pattern` to change the file glob used for a directory:

```bash
python read.py ~/dp_data/new_task1_expertdata/demo_YYYYMMDD_HHMMSS.h5 --fps 30
python read_with_intervention.py ~/dp_data/offlineRL_data/new_task1_iter1 --pattern 'demo_*.h5'
```

Playback controls:

- `Space`: pause or resume.
- `f` / `b`: move forward or backward by 10 frames.
- `r`: restart the current file.
- `n`: play the next file.
- `q`: quit playback.

After reviewing an episode, correct an incorrect `success` attribute by passing the HDF5 file's absolute path and the correct value:

```bash
# Mark one episode as failed
python fix_h5_success.py /home/lrz/dp_data/offlineRL_data/task1/demo_20260507_114630.h5 false

# Mark one episode as successful
python fix_h5_success.py /home/lrz/dp_data/offlineRL_data/task1/demo_20260507_114630.h5 true
```

To replay one trajectory on the UR robot and Inspire hand, edit the trajectory path and hardware settings at the top of `replay_trajectory.sh`. Keep them consistent with the settings used during collection, including `servo_frequency`, then run:

```bash
bash replay_trajectory.sh
```

Replay sends real commands to the robot and hand and asks for confirmation before starting. Verify the workspace, emergency stop, initial pose, and surrounding environment. The script rejects trajectories whose first recorded target exceeds the configured `max_initial_distance`.

Playback supports a speed multiplier, per-frame TCP tracking errors, summary
statistics, and optional CSV output. `replay_trajectory.sh` currently writes a
`*_replay_tracking_error.csv` file by default. The summary reports mean, p95,
and maximum position and rotation-vector errors. Set
`print_tracking_error=true` in the shell script to print every frame, or leave
`tracking_error_csv` empty to disable CSV output. Playback also stops when a UR
protective stop or emergency stop is detected.

## BibTeX

Please consider citing our work if you find this repository useful:

```bibtex
@article{liao2026dexpie,
  title         = {{DexPIE}: Stable Dexterous Policy Improvement from Real-World Experience},
  author        = {Liao, Ruizhe and Chen, Wenrui and Zeng, Liangji and Lin, Haoran and Yang, Fan and Yang, Kailun and Wang, Yaonan},
  journal       = {arXiv preprint arXiv:2606.09615},
  year          = {2026},
  doi           = {10.48550/arXiv.2606.09615},
  url           = {https://arxiv.org/abs/2606.09615},
  eprint        = {2606.09615},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgement

We thank the authors of [iDP3 / Humanoid-Teleoperation](https://github.com/YanjieZe/Humanoid-Teleoperation) and [RealtimeVLA v2](https://dexmal.github.io/realtime-vla-v2/) for their open-source work, which provided valuable reference and inspiration for this project.
