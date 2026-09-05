import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os

# Where to save outputs
OUTPUT_DIR = "captures"
CONFIG_DIR = "config"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# 1. Set up the pipeline — this is how our code talks to the camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

pipeline.start(config)

# 2. Align depth to color — the two sensors sit a few mm apart on the camera,
#    so without this, pixel (100,100) in RGB and pixel (100,100) in depth
#    would NOT be looking at the same physical point.
align = rs.align(rs.stream.color)

# Post-processing filters to reduce depth holes on tricky surfaces
# (transparent, shiny, or thin objects can scatter the IR poorly)
spatial_filter = rs.spatial_filter()
hole_filling_filter = rs.hole_filling_filter()

print("Warming up the camera (letting auto-exposure settle)...")
for _ in range(30):
    pipeline.wait_for_frames()

# 3. Grab one aligned RGB-D pair
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)

color_frame = aligned_frames.get_color_frame()
depth_frame = aligned_frames.get_depth_frame()

if not color_frame or not depth_frame:
    print("ERROR: Did not receive both frames. Check the USB connection.")
    pipeline.stop()
    exit()

# 3b. Clean up the depth frame with the filters
depth_frame = spatial_filter.process(depth_frame)
depth_frame = hole_filling_filter.process(depth_frame)

# 4. Convert to images we can actually look at
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

cv2.imwrite(os.path.join(OUTPUT_DIR, "rgb_test.png"), color_image)

# Depth values are distance in millimeters (16-bit) — not visible as-is,
# so we map them onto a color scale purely so a human can inspect it.
# alpha=0.25 is tuned for close-range scenes (roughly under 1 meter away).
depth_colormap = cv2.applyColorMap(
    cv2.convertScaleAbs(depth_image, alpha=0.25), cv2.COLORMAP_JET
)
cv2.imwrite(os.path.join(OUTPUT_DIR, "depth_test.png"), depth_colormap)

# 5. Pull the camera's intrinsics (its internal lens parameters) — we need
#    these later to convert a pixel + depth value into a real 3D point.
intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()

intrinsics_dict = {
    "width": intrinsics.width,
    "height": intrinsics.height,
    "fx": intrinsics.fx,
    "fy": intrinsics.fy,
    "ppx": intrinsics.ppx,
    "ppy": intrinsics.ppy,
    "distortion_model": str(intrinsics.model),
    "coeffs": intrinsics.coeffs,
}

with open(os.path.join(CONFIG_DIR, "camera_intrinsics.json"), "w") as f:
    json.dump(intrinsics_dict, f, indent=2)

print("Saved rgb_test.png and depth_test.png to the 'captures' folder.")
print("Saved camera_intrinsics.json to the 'config' folder.")
print(intrinsics_dict)

pipeline.stop()