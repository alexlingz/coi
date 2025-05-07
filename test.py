import pyrealsense2 as rs
import cv2
import numpy as np

# 配置管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动管道
pipeline.start(config)



frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()


# 将帧转换为 numpy 数组
color_image = np.asanyarray(color_frame.get_data())

# 显示或保存图片
cv2.imwrite('RealSense.png', color_image)


pipeline.stop()