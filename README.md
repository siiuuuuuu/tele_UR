# tele_UR

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
  imageio==2.35.1
```

After installation, run a basic import check:

```bash
conda activate tele

python - <<'PY'
import numpy, cv2, pyrealsense2, rtde_control, rtde_receive
import openvr, h5py, zarr, serial, termcolor, tqdm, imageio
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

As shown above, align the hand so that the palm normal points along the negative y-axis, with the palm facing the small cylinder.

Before collecting data, make sure the following hardware and services are ready:

- UR5 robot: the code connects to `192.168.3.6` by default. Configure this in the `UR_HOST` constant in `servoL.py` and `traj_valid.py`. If jitter is severe, reduce speed to 50%.
- UR controller: Remote Control/RTDE must be enabled, and the computer must be on the same network as the robot.
- Inspire hand: default serial port is `/dev/ttyUSB0`, baud rate `115200`.
- RealSense cameras: two cameras are used by default, with `front_cam_idx=0` and `right_cam_idx=1`. Device serial numbers are sorted before indexing.
- SteamVR: Set up the Vive Tracker following [Software Setup Tutorial - VIVE Tracker setup](https://docs.google.com/document/d/1ANxSA_PctkqFf3xqAkyktgBgDWEbrFK7b1OnJe54ltw/edit?tab=t.0#heading=h.yxlxo67jgfyx).
- MANUS glove: First follow the [ManusTele repository](https://github.com/siiuuuuuu/ManusTele) to configure and start the MANUS client. Then connect the client to `localhost:8888`, where `SM_inspire_manus_process.py` listens by default, and stream the glove data.

Set temporary serial permission:

```bash
sudo chmod 666 /dev/ttyUSB0
```

For a persistent setup, add the current user to the `dialout` group, then log out and log back in:

```bash
sudo usermod -aG dialout $USER
```

If RealSense devices cannot be enumerated, install librealsense/udev rules first, then reconnect the cameras.

## 4. Collect Demonstrations

After the UR robot, RealSense cameras, SteamVR, MANUS client, and Inspire hand are ready, run:

```bash
conda activate tele
```

Example with explicit robot and safety parameters:

```bash
python servoL.py \
  --demo_dir ~/dp_data/new_task1_expertdata \
  --ur_host 192.168.3.6 \
  --workspace_x -1.0 1.0 \
  --workspace_y -1.0 1.0 \
  --workspace_z 0.0 1.0 \
  --initial_pose 0.248 0.1212 0.3978 1.16 1.25 1.28 \
  --dt 0.04 \
  --max_length 1000 \
  --hand_port /dev/ttyUSB0 \
  --hand_baudrate 115200 \
  --use_wrist_img true
```

You can also use the helper script:

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

## 5. Convert HDF5 to Zarr

Convert regular teleoperation data:

```bash
conda activate tele

python convert_demos.py \
  --demo_dir ~/dp_data/new_task1_expertdata \
  --save_dir ~/dp_data/zarr_task1 \
  --save_img 1 \
  --save_wrist_img 1 \
  --save_depth 0 \
  --save_cloud 0
```

Or edit the paths in `convert_data.sh` and run:

```bash
bash convert_data.sh
```

For data with `success` / `intervention` metadata, or when merging multiple directories:

```bash
python convert_demos_rollout.py \
  --demo_dirs ~/dp_data/task1_expertdata ~/dp_data/task1_rollout \
  --save_dir ~/dp_data/task1_Recap_iter2 \
  --save_img 1 \
  --save_wrist_img 1
```

Note: the conversion scripts overwrite `save_dir` if it already exists.

## 6. Inspect Data and Replay Trajectories

Play the synchronized `color` and `wrist_color` camera streams:

```bash
python read.py ~/dp_data/new_task1_expertdata
```

Play rollout data with synchronized cameras and `intervention` / `success` status overlays:

```bash
python read_with_intervention.py ~/dp_data/offlineRL_data/new_task1_iter1
```

`read.py` displays both cameras side by side with playback status, frame information, and a progress bar. `read_with_intervention.py` additionally uses green borders for autonomous steps and red borders for intervention steps, displays the current mode and episode success/failure at the top, and shows all intervention segments on the bottom timeline.

Both scripts accept either an individual HDF5 file or a directory. Use `--fps` to change the playback frame rate and `--pattern` to change the file glob used for a directory:

```bash
python read.py ~/dp_data/new_task1_expertdata/demo_YYYYMMDD_HHMMSS.h5 --fps 25
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

Replay one trajectory on the UR robot and Inspire hand:

```bash
python traj_valid.py \
  --data_file ~/dp_data/new_task1_expertdata/demo_YYYYMMDD_HHMMSS.h5 \
  --ur_host 192.168.3.6 \
  --workspace_x -1.0 1.0 \
  --workspace_y -1.0 1.0 \
  --workspace_z 0.0 1.0 \
  --initial_pose 0.248 0.1212 0.3978 1.16 1.25 1.28 \
  --dt 0.04 \
  --hand_port /dev/ttyUSB0 \
  --hand_baudrate 115200 \
  --speed 1.0 \
  -y
```

Use the same UR host, workspace, initial pose, control period, and hand serial port used during collection. Replay sends real commands to the robot and hand. Before running it, verify the workspace, emergency stop, initial pose, and surrounding environment. The script rejects trajectories whose first recorded target is more than `0.10 m` from the configured initial pose unless `--max_initial_distance` is changed explicitly.
