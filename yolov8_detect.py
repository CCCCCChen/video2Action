import cv2
from ultralytics import YOLO

# 1. 加载预训练 YOLOv8 模型（用 COCO 数据集的 'person' 类）
model = YOLO("yolov8n.pt")  # 轻量版，速度快。可换成 yolov8s.pt 提升精度

# 2. 打开视频
video_path = "clip.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频大小
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video size: {frame_width}x{frame_height}, FPS: {fps}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. YOLOv8 推理
    results = model(frame, verbose=False)

    # 4. 遍历检测结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])  # 类别ID
            conf = float(box.conf[0])  # 置信度
            if cls_id == 0 and conf > 0.5:  # 0 表示 'person' 类
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 5. 显示
    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
