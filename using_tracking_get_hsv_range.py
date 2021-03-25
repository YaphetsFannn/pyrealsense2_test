## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

def nothing(x):
    pass

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
    cv2.namedWindow("Tracking")
    cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
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
        
        hsv_map = cv2.cvtColor(bg_rmvd,cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")
        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")
        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv_map, l_b, u_b)
        res = cv2.bitwise_and(bg_rmvd, bg_rmvd, mask=mask)
        
        img = np.hstack((bg_rmvd, depth_colormap))
        cv2.imshow('Align Example',img)
        cv2.imshow("mask", mask)
        cv2.imshow("res", res)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()

finally:

    # Stop streaming
    pipeline.stop()