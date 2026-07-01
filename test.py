import pyrealsense2 as rs
import time


def get_metadata(frame, key):
    if not frame.supports_frame_metadata(key):
        return None
    return float(frame.get_frame_metadata(key))


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30,
)

profile = pipeline.start(config)

# 启用主机校正后的 Global Timestamp
for sensor in profile.get_device().query_sensors():
    if sensor.supports(rs.option.global_time_enabled):
        sensor.set_option(rs.option.global_time_enabled, 1.0)

# 给 Global Time 映射一些初始化时间
for _ in range(30):
    pipeline.wait_for_frames()

try:
    while True:
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()

        if not frame:
            continue

        sensor_us = get_metadata(
            frame,
            rs.frame_metadata_value.sensor_timestamp,
        )

        frame_device_us = get_metadata(
            frame,
            rs.frame_metadata_value.frame_timestamp,
        )

        exposure_us = get_metadata(
            frame,
            rs.frame_metadata_value.actual_exposure,
        )

        # 通常是 global_time、hardware_clock 或 system_time
        timestamp_domain = frame.get_frame_timestamp_domain()

        # 官方定义：get_timestamp() 返回毫秒
        frame_global_ms = frame.get_timestamp()

        if sensor_us is None or frame_device_us is None:
            print("没有 SENSOR_TIMESTAMP 或 FRAME_TIMESTAMP")
            continue

        # 两个设备时间戳之间的差值
        exposure_to_readout_ms = (
            frame_device_us - sensor_us
        ) / 1000.0

        # 曝光中点转换到 Global/Host 时间域
        exposure_mid_host_ms = (
            frame_global_ms - exposure_to_readout_ms
        )

        result = {
            "frame_number": frame.get_frame_number(),
            "timestamp_domain": str(timestamp_domain),
            "exposure_mid_host_ms": exposure_mid_host_ms,
        }

        if exposure_us is not None:
            result["exposure_start_host_ms"] = (
                exposure_mid_host_ms - exposure_us / 2000.0
            )
            result["exposure_end_host_ms"] = (
                exposure_mid_host_ms + exposure_us / 2000.0
            )
            result["exposure_duration_ms"] = (
                exposure_us / 1000.0
            )

        print(result)

finally:
    pipeline.stop()