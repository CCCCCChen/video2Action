#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + MediaPipe Pose + 光流背景检测 + 原神特化规则的动作识别系统

流程设计:
1. YOLO检测人物 -> 选取中心ROI
2. ROI平滑化（卡尔曼滤波）
3. 光流检测背景移动
4. MediaPipe Pose识别骨骼关键点
5. 基于骨骼关键点位移判定动作
6. 游戏特定规则过滤
7. 输出动作日志

作者: AI Assistant
版本: 3.0
"""

import cv2
import numpy as np
import time
import json
import math
import os
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from scipy.spatial.distance import euclidean
from filterpy.kalman import KalmanFilter

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy数据类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
# from pynput.keyboard import Key, Controller  # 取消注释以启用按键模拟
# keyboard = Controller()  # 取消注释以启用按键模拟

# MediaPipe初始化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def load_config(config_path="pose_config.json"):
    """加载配置文件"""
    default_config = {
        "model_settings": {
            "yolo_model_path": "yolov8n.pt",
            "video_path": "clip.mp4",
            "confidence_threshold": 0.5
        },
        "roi_settings": {
            "center_region_ratio": 0.6,
            "roi_expand_ratio": 1.5,
            "kalman_q": 0.001,
            "kalman_r": 0.01
        },
        "optical_flow_settings": {
            "optical_flow_points": 100,
            "camera_move_threshold": 5.0
        },
        "action_detection": {
            "jump_threshold": 15,
            "move_threshold": 8,
            "idle_threshold": 3,
            "action_smooth_frames": 3
        },
        "key_mapping": {
            "Jump": "space",
            "Landing": "space",
            "Move Right": "d",
            "Move Left": "a",
            "Move Forward": "w",
            "Move Backward": "s",
            "Idle": None
        }
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"配置文件加载成功: {config_path}")
            return config
        else:
            print(f"配置文件不存在，使用默认配置: {config_path}")
            return default_config
    except Exception as e:
        print(f"配置文件加载失败，使用默认配置: {e}")
        return default_config

# 全局配置
config = load_config()

# 配置参数类
class Config:
    # 关键点索引（MediaPipe Pose）
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    NOSE = 0

class SimpleROIFilter:
    """简单的ROI平滑滤波器（指数移动平均）"""
    def __init__(self, alpha=0.7):
        self.alpha = alpha  # 平滑系数
        self.prev_roi = None
        self.initialized = False
    
    def update(self, bbox):
        """更新ROI滤波器"""
        bbox = np.array(bbox, dtype=np.float32)
        
        if not self.initialized or self.prev_roi is None:
            self.prev_roi = bbox.copy()
            self.initialized = True
            return bbox.astype(int)
        
        # 指数移动平均平滑
        smoothed_roi = self.alpha * bbox + (1 - self.alpha) * self.prev_roi
        self.prev_roi = smoothed_roi.copy()
        
        return smoothed_roi.astype(int)

class OpticalFlowDetector:
    """光流检测器用于背景移动检测"""
    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        
        # Lucas-Kanade光流参数
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 特征点检测参数
        self.feature_params = dict(
            maxCorners=config['optical_flow_settings'].get('feature_max_corners', 100),
            qualityLevel=config['optical_flow_settings'].get('feature_quality_level', 0.3),
            minDistance=config['optical_flow_settings'].get('feature_min_distance', 7),
            blockSize=7
        )
    
    def detect_camera_movement(self, frame):
        """检测相机移动"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        camera_movement = np.array([0.0, 0.0])
        
        if self.prev_gray is not None:
            if self.prev_points is not None and len(self.prev_points) > 10:
                # 计算光流
                next_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_points, None, **self.lk_params
                )
                
                # 选择好的点
                good_new = next_points[status == 1]
                good_old = self.prev_points[status == 1]
                
                if len(good_new) > 10:
                    # 计算平均移动向量
                    movements = good_new - good_old
                    camera_movement = np.mean(movements, axis=0)
            
            # 重新检测特征点
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        else:
            # 初始化特征点
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        
        self.prev_gray = gray.copy()
        
        # 判断是否为相机移动
        movement_magnitude = np.linalg.norm(camera_movement)
        is_camera_moving = movement_magnitude > config['optical_flow_settings']['camera_move_threshold']
        
        return camera_movement, is_camera_moving

class PoseActionDetector:
    """基于MediaPipe Pose的动作检测器"""
    def __init__(self):
        self.pose_history = deque(maxlen=config['action_detection'].get('pose_history_length', 10))
        self.action_history = deque(maxlen=config['action_detection']['action_smooth_frames'])
        self.last_action_time = time.time()
        self.key_mapping = config.get('key_mapping', {})
        self.last_simulated_action = None
    
    def extract_key_points(self, image):
        """提取关键点"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.visibility])
            return np.array(landmarks), results.pose_landmarks
        
        return None, None
    
    def calculate_pose_features(self, landmarks, image_shape):
        """计算姿态特征"""
        h, w = image_shape[:2]
        features = {}
        
        # 脚踝位置（像素坐标）
        left_ankle = landmarks[Config.LEFT_ANKLE] * [w, h, 1]
        right_ankle = landmarks[Config.RIGHT_ANKLE] * [w, h, 1]
        
        # 膝盖位置
        left_knee = landmarks[Config.LEFT_KNEE] * [w, h, 1]
        right_knee = landmarks[Config.RIGHT_KNEE] * [w, h, 1]
        
        # 臀部位置
        left_hip = landmarks[Config.LEFT_HIP] * [w, h, 1]
        right_hip = landmarks[Config.RIGHT_HIP] * [w, h, 1]
        
        # 肩膀位置
        left_shoulder = landmarks[Config.LEFT_SHOULDER] * [w, h, 1]
        right_shoulder = landmarks[Config.RIGHT_SHOULDER] * [w, h, 1]
        
        # 鼻子位置
        nose = landmarks[Config.NOSE] * [w, h, 1]
        
        # 计算特征
        features['foot_center'] = (left_ankle[:2] + right_ankle[:2]) / 2
        features['knee_center'] = (left_knee[:2] + right_knee[:2]) / 2
        features['hip_center'] = (left_hip[:2] + right_hip[:2]) / 2
        features['shoulder_center'] = (left_shoulder[:2] + right_shoulder[:2]) / 2
        features['body_center'] = (features['hip_center'] + features['shoulder_center']) / 2
        features['head_position'] = nose[:2]
        
        # 身体高度
        features['body_height'] = abs(features['head_position'][1] - features['foot_center'][1])
        
        # 脚部高度（用于跳跃检测）
        features['foot_height'] = features['foot_center'][1]
        
        # 身体宽度
        features['body_width'] = abs(left_shoulder[0] - right_shoulder[0])
        
        return features
    
    def detect_action(self, landmarks, image_shape, camera_movement, is_camera_moving):
        """检测动作"""
        if landmarks is None:
            return None
        
        features = self.calculate_pose_features(landmarks, image_shape)
        self.pose_history.append(features)
        
        if len(self.pose_history) < 3:
            return None
        
        # 获取历史特征
        current = self.pose_history[-1]
        previous = self.pose_history[-3]
        
        # 计算位移（补偿相机移动）
        foot_displacement = current['foot_center'] - previous['foot_center']
        body_displacement = current['body_center'] - previous['body_center']
        
        # 如果检测到相机移动，补偿位移
        if is_camera_moving:
            foot_displacement -= camera_movement * 2  # 乘以帧数差
            body_displacement -= camera_movement * 2
        
        # 动作检测逻辑
        action = self._classify_action(current, previous, foot_displacement, body_displacement)
        
        # 动作平滑
        self.action_history.append(action)
        smoothed_action = self._smooth_action()
        
        return smoothed_action
    
    def _classify_action(self, current, previous, foot_displacement, body_displacement):
        """分类动作"""
        # 计算各种位移量
        foot_dx, foot_dy = foot_displacement
        body_dx, body_dy = body_displacement
        
        # 高度变化（跳跃检测）
        height_change = previous['foot_height'] - current['foot_height']  # Y轴向上为负
        
        # 获取阈值
        jump_threshold = config['action_detection']['jump_threshold']
        move_threshold = config['action_detection']['move_threshold']
        idle_threshold = config['action_detection']['idle_threshold']
        
        # 跳跃检测（优先级最高）
        if height_change > jump_threshold:
            return "Jump"
        
        # 下降/着陆检测
        if height_change < -jump_threshold:
            return "Landing"
        
        # 水平移动检测
        horizontal_movement = abs(foot_dx)
        if horizontal_movement > move_threshold:
            if foot_dx > 0:
                return "Move Right"
            else:
                return "Move Left"
        
        # 前后移动检测（基于身体中心的Y轴移动）
        vertical_movement = abs(body_dy)
        if vertical_movement > move_threshold:
            if body_dy > 0:
                return "Move Backward"
            else:
                return "Move Forward"
        
        # 待机检测
        total_movement = np.linalg.norm([foot_dx, foot_dy])
        if total_movement < idle_threshold:
            return "Idle"
        
        return "Unknown"
    
    def _smooth_action(self):
        """动作平滑"""
        smooth_frames = config['action_detection']['action_smooth_frames']
        if len(self.action_history) < smooth_frames:
            return None
        
        # 统计最近几帧的动作
        action_counts = {}
        for action in self.action_history:
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
        
        if not action_counts:
            return None
        
        # 返回出现次数最多的动作
        most_common_action = max(action_counts, key=action_counts.get)
        
        # 如果动作一致性不够，返回None
        if action_counts[most_common_action] < smooth_frames // 2:
            return None
        
        return most_common_action
    
    def simulate_key_press(self, action):
        """模拟按键操作"""
        if action and action in self.key_mapping and self.key_mapping[action]:
            key = self.key_mapping[action]
            if action != self.last_simulated_action:
                try:
                    # 这里可以添加实际的按键模拟代码
                    # keyboard.press(key)
                    # keyboard.release(key)
                    print(f"模拟按键: {key} (动作: {action})")
                    self.last_simulated_action = action
                except Exception as e:
                    print(f"按键模拟失败: {e}")
            return key
        return None

class GameActionRecognizer:
    """游戏动作识别主类"""
    def __init__(self):
        # 初始化组件
        self.yolo_model = YOLO(config['model_settings']['yolo_model_path'])
        self.roi_filter = SimpleROIFilter()
        self.optical_flow = OpticalFlowDetector()
        self.pose_detector = PoseActionDetector()
        
        # 状态变量
        self.frame_count = 0
        self.actions_log = []
        self.last_roi = None
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'person_detected_frames': 0,
            'actions_detected': 0,
            'camera_movement_frames': 0
        }
    
    def detect_person_roi(self, frame):
        """检测人物ROI"""
        results = self.yolo_model(frame, verbose=False)
        
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        best_person = None
        min_distance = float('inf')
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls_id == 0 and conf > config['model_settings']['confidence_threshold']:  # 人物类别
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 计算检测框中心
                        bbox_center_x = (x1 + x2) // 2
                        bbox_center_y = (y1 + y2) // 2
                        
                        # 计算到画面中心的距离
                        distance = math.sqrt(
                            (bbox_center_x - center_x) ** 2 + 
                            (bbox_center_y - center_y) ** 2
                        )
                        
                        # 选择最靠近中心的人物
                        if distance < min_distance:
                            min_distance = distance
                            best_person = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
        
        if best_person is not None:
            # 使用卡尔曼滤波平滑ROI
            smoothed_roi = self.roi_filter.update(best_person)
            
            # 扩展ROI以包含更多上下文
            x, y, w, h = smoothed_roi
            expand_w = int(w * config['roi_settings']['roi_expand_ratio'])
            expand_h = int(h * config['roi_settings']['roi_expand_ratio'])
            
            # 确保ROI在图像范围内
            x = max(0, x - (expand_w - w) // 2)
            y = max(0, y - (expand_h - h) // 2)
            expand_w = min(expand_w, frame.shape[1] - x)
            expand_h = min(expand_h, frame.shape[0] - y)
            
            self.last_roi = [x, y, expand_w, expand_h]
            return self.last_roi
        
        return self.last_roi  # 返回上一帧的ROI
    
    def process_frame(self, frame):
        """处理单帧"""
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        # 1. 检测人物ROI
        roi = self.detect_person_roi(frame)
        
        if roi is None:
            return frame, None, None, False
        
        self.stats['person_detected_frames'] += 1
        
        # 2. 光流检测背景移动
        camera_movement, is_camera_moving = self.optical_flow.detect_camera_movement(frame)
        
        if is_camera_moving:
            self.stats['camera_movement_frames'] += 1
        
        # 3. 提取ROI区域
        x, y, w, h = roi
        roi_image = frame[y:y+h, x:x+w]
        
        if roi_image.size == 0:
            return frame, None, None, is_camera_moving
        
        # 4. MediaPipe Pose检测
        landmarks, pose_landmarks = self.pose_detector.extract_key_points(roi_image)
        
        # 5. 动作检测
        action = None
        if landmarks is not None:
            action = self.pose_detector.detect_action(
                landmarks, roi_image.shape, camera_movement, is_camera_moving
            )
            
            if action and action != "Unknown":
                self.stats['actions_detected'] += 1
                
                # 模拟按键
                simulated_key = self.pose_detector.simulate_key_press(action)
                
                # 记录动作日志
                log_entry = {
                    'timestamp': time.time(),
                    'frame': self.frame_count,
                    'action': action,
                    'simulated_key': simulated_key,
                    'roi': roi,
                    'camera_moving': is_camera_moving,
                    'camera_movement': camera_movement.tolist()
                }
                self.actions_log.append(log_entry)
        
        # 6. 绘制可视化信息
        annotated_frame = self.draw_annotations(
            frame, roi, pose_landmarks, action, is_camera_moving, camera_movement
        )
        
        return annotated_frame, action, roi, is_camera_moving
    
    def draw_annotations(self, frame, roi, pose_landmarks, action, is_camera_moving, camera_movement):
        """绘制标注信息"""
        annotated_frame = frame.copy()
        
        # 绘制ROI
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "ROI", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制骨骼关键点
        if pose_landmarks is not None and roi is not None:
            x, y, w, h = roi
            
            # 将ROI内的关键点坐标转换为全图坐标
            for landmark in pose_landmarks.landmark:
                px = int(landmark.x * w + x)
                py = int(landmark.y * h + y)
                cv2.circle(annotated_frame, (px, py), 3, (255, 0, 0), -1)
        
        # 显示动作信息
        if action:
            action_color = (0, 0, 255) if action == "Idle" else (255, 0, 0)
            cv2.putText(annotated_frame, f"Action: {action}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, action_color, 2)
        
        # 显示相机移动状态
        camera_status = "Camera Moving" if is_camera_moving else "Camera Static"
        camera_color = (0, 165, 255) if is_camera_moving else (0, 255, 0)
        cv2.putText(annotated_frame, camera_status, (50, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, camera_color, 2)
        
        # 显示统计信息
        stats_text = [
            f"Frame: {self.frame_count}",
            f"Actions: {self.stats['actions_detected']}",
            f"Person Rate: {self.stats['person_detected_frames']/self.stats['total_frames']*100:.1f}%"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(annotated_frame, text, (frame.shape[1] - 250, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def save_log(self, filename="pose_actions_log.json"):
        """保存动作日志"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'stats': self.stats,
                    'actions': self.actions_log
                }, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            print(f"动作日志已保存到 {filename}")
        except Exception as e:
            print(f"保存日志失败: {e}")

def main():
    """主函数"""
    print("=== YOLO + MediaPipe Pose 动作识别系统 ===")
    print(f"模型: {config['model_settings']['yolo_model_path']}")
    print(f"视频: {config['model_settings']['video_path']}")
    print("\n功能特性:")
    print("- YOLO人物检测 + ROI裁剪")
    print("- 卡尔曼滤波ROI平滑")
    print("- 光流背景移动检测")
    print("- MediaPipe Pose骨骼关键点")
    print("- 原神特化动作识别")
    print("\n控制说明:")
    print("- 按 'q' 退出")
    print("- 按 's' 保存日志")
    print("- 按 'r' 重置统计")
    print("- 按 'c' 清空动作历史")
    print("\n开始处理...\n")
    
    # 初始化识别器
    recognizer = GameActionRecognizer()
    
    # 打开视频
    video_path = config['model_settings']['video_path']
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            annotated_frame, action, roi, is_camera_moving = recognizer.process_frame(frame)
            
            # 显示结果
            window_name = config.get('display_settings', {}).get('window_name', "YOLO + MediaPipe Pose Action Recognition")
            cv2.imshow(window_name, annotated_frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                recognizer.save_log()
            elif key == ord('r'):
                recognizer.stats = {
                    'total_frames': 0,
                    'person_detected_frames': 0,
                    'actions_detected': 0,
                    'camera_movement_frames': 0
                }
                recognizer.actions_log.clear()
                print("统计信息已重置")
            elif key == ord('c'):
                recognizer.pose_detector.pose_history.clear()
                recognizer.pose_detector.action_history.clear()
                print("动作历史已清空")
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        # 保存最终日志
        recognizer.save_log()
        
        # 显示最终统计
        stats = recognizer.stats
        print(f"\n=== 最终统计 ===")
        print(f"总帧数: {stats['total_frames']}")
        print(f"检测到人物的帧数: {stats['person_detected_frames']}")
        print(f"检测到动作的次数: {stats['actions_detected']}")
        print(f"相机移动的帧数: {stats['camera_movement_frames']}")
        print(f"人物检测率: {stats['person_detected_frames']/stats['total_frames']*100:.2f}%")
        print(f"相机移动率: {stats['camera_movement_frames']/stats['total_frames']*100:.2f}%")
        
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    main()