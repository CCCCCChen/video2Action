import cv2
import numpy as np

# 视频路径
video_path = "clip.mp4"

video_path = "test.mp4"

# 加载 Haar 级联分类器（人体检测）
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# 打开视频文件
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 中央区域阈值
center_threshold_x = frame_width * 0.2
center_threshold_y = frame_height * 0.2

# 存储上一次人物中心位置用于动作判断
prev_center_y = None

print(f"Video size: {frame_width}x{frame_height}, FPS: {fps}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人物
    bodies = body_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in bodies:
        center_x = x + w / 2
        center_y = y + h / 2

        # 只检测中央区域的人物
        if (frame_width * 0.5 - center_threshold_x < center_x < frame_width * 0.5 + center_threshold_x) and \
           (frame_height * 0.5 - center_threshold_y < center_y < frame_height * 0.5 + center_threshold_y):

            # 画出人物框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 简单动作识别（检测Y方向位置变化来判断跳跃）
            if prev_center_y is not None:
                if prev_center_y - center_y > 15:  # 阈值可调
                    action = "Jump"
                elif center_y - prev_center_y > 15:
                    action = "Fall"
                else:
                    action = "Move/Idle"
            else:
                action = "Idle"

            prev_center_y = center_y

            # 显示动作
            cv2.putText(frame, f"Action: {action}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示视频
    cv2.imshow("Character Detection", frame)

    # 按下 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
