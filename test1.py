import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    60,
)

profile = pipeline.start(config)

try:
    for _ in range(30):
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue

        required = [
            rs.frame_metadata_value.sensor_timestamp,
            rs.frame_metadata_value.frame_timestamp,
        ]

        if not all(color.supports_frame_metadata(x) for x in required):
            print("metadata unavailable")
            continue

        sensor_us = color.get_frame_metadata(
            rs.frame_metadata_value.sensor_timestamp
        )
        frame_us = color.get_frame_metadata(
            rs.frame_metadata_value.frame_timestamp
        )

        exposure_us = None
        if color.supports_frame_metadata(
            rs.frame_metadata_value.actual_exposure
        ):
            exposure_us = color.get_frame_metadata(
                rs.frame_metadata_value.actual_exposure
            )

        print(
            f"number={color.get_frame_number():6d} "
            f"sensor={sensor_us:12d} us  "
            f"frame={frame_us:12d} us  "
            f"delta={(frame_us - sensor_us) / 1000:8.3f} ms  "
            f"exposure={exposure_us} us  "
            f"sdk={color.get_timestamp():.3f} ms  "
            f"domain={color.get_frame_timestamp_domain()}"
        )

finally:
    pipeline.stop()