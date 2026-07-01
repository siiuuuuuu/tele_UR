import pyrealsense2 as rs

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

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        key = rs.frame_metadata_value.sensor_timestamp

        if color_frame.supports_frame_metadata(key):
            # D400 系列这里通常是微秒级设备时钟
            exposure_mid_device_us = color_frame.get_frame_metadata(key)

            print(
                "曝光中点（相机硬件时钟）:",
                exposure_mid_device_us,
                "us",
            )
        else:
            print("当前系统没有提供 SENSOR_TIMESTAMP 元数据")

finally:
    pipeline.stop()