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

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # Intrinsics & Extrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        # print(depth_intrin)
        # [ 640x480  p[314.914 245.854]  f[476.37 476.37]  
        # Inverse Brown Conrady [0.140162 0.0737665 0.00409275 0.00307178 0.0778658] ]

        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        # print(color_intrin)
        # [ 640x480  p[314.696 243.657]  f[615.932 615.932]  None [0 0 0 0 0] 

        depth2color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        # print(depth2color_extrin)

        # get scale of depth sensor
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("depth_scale is")
        print(depth_scale)
        
        # clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1 # 1 meter
        clipping_distance = clipping_distance_in_meters/depth_scale


        #Map depth to  color
        depth_pixel = [320,240] # Radom pixel
        dist2center = depth_frame.get_distance(320,240)        
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dist2center)
        print(depth_point)

        color_point = rs.rs2_transform_point_to_point(depth2color_extrin, depth_point)
        print(color_point)
        color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
        print(color_pixel)
        break

finally:

    # Stop streaming
    pipeline.stop()