from moviepy.editor import VideoFileClip
import os

base_dir = "D:\Works\my-nerf\\results"
video_name = "kanade_test_spiral_200000_rgb_no_fine"
# 加载视频
clip = VideoFileClip(os.path.join(base_dir, video_name + ".mp4"))

# 将视频保存为 GIF
gif = clip.speedx(factor=2.0).write_gif(os.path.join(base_dir, video_name + ".gif"), fps=30)