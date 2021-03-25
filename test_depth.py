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
        print(depth_intrin)
        # [ 640x480  p[314.914 245.854]  f[476.37 476.37]  
        # Inverse Brown Conrady [0.140162 0.0737665 0.00409275 0.00307178 0.0778658] ]

        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        print(color_intrin)
        # [ 640x480  p[314.696 243.657]  f[615.932 615.932]  None [0 0 0 0 0] 

        depth2color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        print(depth2color_extrin)

        dist2center = depth_frame.getdistance(320,240)
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(depth_scale)
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        print(depth_image.shape, depth_image.dtype)
        print(depth_image[250][259])
        # Convert images to numpy arrays

        

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        dimg_gray = cv2.convertScaleAbs(depth_image, alpha=255/4000)    # alpha = 0.03
        print(dimg_gray.shape,dimg_gray.dtype)
        print(dimg_gray[250][259])
        
        depth_colormap = cv2.applyColorMap(dimg_gray, cv2.COLORMAP_JET)
        print(depth_colormap.shape,depth_colormap.dtype)
        print(depth_colormap[250][259])
        
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        #"depth map and color map 's shape is ((480, 640, 3), (480, 640, 3))"

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # get gray img and concate with depth
        gray_img = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        # gray_depth = np.concatenate((depth_colormap,gray_img), axis=1)
        cv2.imshow('gray_depth',gray_img)
        # print(depth_colormap)
        
        cv2.waitKey(0)
        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', images)
        # cv2.waitKey(1)
        pipeline.stop()

finally:

    # Stop streaming
    pipeline.stop()