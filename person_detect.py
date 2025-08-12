import cv2
import math

# 视频路径
video_path = 'clip.mp4'

# 加载 Haar 级联分类器
cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
body_cascade = cv2.CascadeClassifier(cascade_path)

# 打开视频文件
# 打开视频文件并获取大小
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度

# 计算中央区域阈值（20% 容忍度）
center_threshold_x = frame_width * 0.2
center_threshold_y = frame_height * 0.2

print(f"Video size: {frame_width}x{frame_height}")


frame_count = 0
detected_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用 Haar 级联分类器检测人体
    bodies = body_cascade.detectMultiScale(gray, 1.1, 2)

 
    # 只在人物靠近画面中心时绘制框
    for (x, y, w, h) in bodies:
        center_x = x + w / 2
        center_y = y + h / 2
        if (frame_width * 0.5 - center_threshold_x < center_x < frame_width * 0.5 + center_threshold_x) and \
           (frame_height * 0.5 - center_threshold_y < center_y < frame_height * 0.5 + center_threshold_y):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected_frames.append(frame)
    
    frame_count += 1

    # 记录前一帧的人物位置
    previous_position = None

    # 检测人物运动
    for (x, y, w, h) in bodies:
        center_x = x + w / 2
        center_y = y + h / 2

        if previous_position:
            # 计算当前位置和前一帧的位移
            delta_x = center_x - previous_position[0]
            delta_y = center_y - previous_position[1]

            # 检测前进（水平位移大于一定阈值）
            if abs(delta_x) > 10:  # 10 可以根据实际情况调整
                print(f"检测到前进，水平位移: {delta_x}")

            # 检测跳跃（垂直位移大于一定阈值）
            if abs(delta_y) > 10:  # 10 可以根据实际情况调整
                if delta_y > 0:
                    print("检测到跳跃，人物上升")
                else:
                    print("检测到跳跃，人物下降")

        # 更新前一帧的人物位置
        previous_position = (center_x, center_y)

    cv2.imshow("Detected Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

cap.release()

# 检测到的人物帧数量
print(f"Total frames with detected bodies: {len(detected_frames)}")
