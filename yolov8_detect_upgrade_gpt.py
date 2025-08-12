import cv2
import numpy as np
from ultralytics import YOLO

def compute_thresholds(video_path, model, sample_seconds=2, factor=3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(sample_seconds * fps)

    last_bbox = None
    dx_list, dy_list, dh_list = [], [], []

    frame_idx = 0
    while frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = detect_person_bbox(frame, model)

        if bbox and last_bbox:
            dx = bbox[0] - last_bbox[0]
            dy = bbox[1] - last_bbox[1]
            dh = bbox[3] - last_bbox[3]
            dx_list.append(dx)
            dy_list.append(dy)
            dh_list.append(dh)

        last_bbox = bbox
        frame_idx += 1

    cap.release()

    def calc_thresh(data):
        if len(data) == 0:
            return 0
        median = np.median(data)
        std_dev = np.std(data)
        return abs(median) + factor * std_dev

    return calc_thresh(dx_list), calc_thresh(dy_list), calc_thresh(dh_list)

def detect_person_bbox(frame, model):
    results = model(frame)
    for box in results[0].boxes:
        cls = int(box.cls)
        if cls == 0:  # YOLO 类别 0 = person
            x1, y1, x2, y2 = box.xyxy[0]
            w, h = x2 - x1, y2 - y1
            return (float(x1 + w / 2), float(y1 + h / 2), float(w), float(h))
    return None

def detect_actions(video_path, model, thresholds):
    cap = cv2.VideoCapture(video_path)
    dx_thresh, dy_thresh, dh_thresh = thresholds
    print(f"阈值 -> dx:{dx_thresh:.2f}, dy:{dy_thresh:.2f}, dh:{dh_thresh:.2f}")

    last_bbox = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = detect_person_bbox(frame, model)

        action_text = ""
        if bbox and last_bbox:
            dx = bbox[0] - last_bbox[0]
            dy = bbox[1] - last_bbox[1]
            dh = bbox[3] - last_bbox[3]

            if abs(dx) > dx_thresh:
                action_text += "右移 " if dx > 0 else "左移 "
            if abs(dy) > dy_thresh:
                action_text += "前进 " if dy < 0 else "后退 "
            if abs(dh) > dh_thresh:
                action_text += "跳跃/下蹲 "

        if bbox:
            cx, cy, w, h = bbox
            x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if action_text:
                cv2.putText(frame, action_text.strip(), (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("YOLO Action Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break

        last_bbox = bbox

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "test.mp4"
    model = YOLO("yolov8s.pt")  # 你可以换成 yolov8s.pt 提高精度

    thresholds = compute_thresholds(video_path, model, sample_seconds=2, factor=3)
    detect_actions(video_path, model, thresholds)
