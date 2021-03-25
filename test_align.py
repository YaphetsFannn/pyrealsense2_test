## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
pc = rs.pointcloud()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

device_product_line = str(device.get_info(rs.camera_info.product_line))
# device_product_line is SR300

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# get scale of depth sensor
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("depth_scale is")
print(depth_scale)

# clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 # 1 meter
clipping_distance = clipping_distance_in_meters/depth_scale

# creat align map
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frame = align.process(frames)

        depth_frame = aligned_frame.get_depth_frame()
        color_frame = aligned_frame.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # remove background
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image))   # depth img is 1 channel, color is 3 channels
        bg_rmvd = np.where((depth_image_3d > clipping_distance)|(depth_image_3d<=0),grey_color,color_image)

        # get final img
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.03),cv2.COLORMAP_JET)
        img = np.hstack((bg_rmvd, depth_colormap))
        cv2.namedWindow("Align Example",cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example',img)
        cv2.waitKey(1)
        # break

finally:

    # Stop streaming
    pipeline.stop()