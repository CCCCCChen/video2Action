import cv2
import os

# 创建保存帧的文件夹
frames_dir = 'frames_sample'
os.makedirs(frames_dir, exist_ok=True)

# 打开视频文件
video_path = 'clip.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

# 每隔 15 帧保存一张图（假设 30fps，大约0.5秒一张）
interval = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % interval == 0:
        frame_filename = os.path.join(frames_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
    frame_count += 1

cap.release()

saved_count, frames_dir
