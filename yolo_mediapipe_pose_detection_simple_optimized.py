#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + MediaPipe Pose 游戏动作识别系统 - 简化优化版本

主要优化:
1. 降低YOLO检测置信度阈值
2. 改进ROI稳定化算法
3. 增强动作检测逻辑
4. 添加多帧验证机制
5. 简化数组操作避免错误

作者: AI Assistant
版本: 2.1 (简化优化版)
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

# MediaPipe初始化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def load_config(config_path="pose_config_optimized.json"):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return {
            "model_settings": {
                "yolo_model_path": "yolov8s.pt",
                "video_path": "test.mp4",
                "confidence_threshold": 0.4,  # 降低阈值
                "iou_threshold": 0.5,
                "max_detections": 1
            },
            "roi_settings": {
                "center_region_ratio": 0.8,
                "roi_expand_ratio": 1.3,
                "roi_stability_threshold": 30,  # 降低稳定性阈值
                "roi_min_size": 60,  # 降低最小尺寸
                "roi_max_size": 500
            },
            "optical_flow_settings": {
                "camera_move_threshold": 2.5,  # 降低阈值
                "lk_win_size": 15,
                "lk_max_level": 2,
                "feature_max_corners": 100,
                "feature_quality_level": 0.01,
                "feature_min_distance": 7
            },
            "action_detection": {
                "jump_threshold": 15,  # 降低阈值
                "move_threshold": 8,   # 降低阈值
                "idle_threshold": 3,
                "action_smooth_frames": 3,  # 减少平滑帧数
                "pose_history_length": 10,
                "velocity_smooth_frames": 2
            },
            "mediapipe_pose": {
                "model_complexity": 1,  # 降低复杂度提高速度
                "min_detection_confidence": 0.4,  # 降低阈值
                "min_tracking_confidence": 0.4,   # 降低阈值
                "enable_segmentation": False,
                "smooth_landmarks": True
            },
            "key_mapping": {
                "Jump": "space",
                "Landing": "space",
                "Move Right": "d",
                "Move Left": "a",
                "Move Forward": "w",
                "Move Backward": "s",
                "Idle": None
            },
            "display_settings": {
                "window_name": "YOLO + MediaPipe Pose Detection (Simple Optimized)",
                "show_roi": True,
                "show_pose": True,
                "show_action_text": True,
                "show_confidence": True
            }
        }
    except Exception as e:
        print(f"加载配置文件失败: {e}，使用默认配置")
        return load_config()  # 递归调用返回默认配置

# 加载配置
config = load_config()

# MediaPipe Pose初始化（使用配置参数）
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=config['mediapipe_pose']['model_complexity'],
    enable_segmentation=config['mediapipe_pose']['enable_segmentation'],
    min_detection_confidence=config['mediapipe_pose']['min_detection_confidence'],
    min_tracking_confidence=config['mediapipe_pose']['min_tracking_confidence']
)

class Config:
    """MediaPipe关键点索引"""
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    NOSE = 0
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14

class SimpleROIFilter:
    """简化的ROI滤波器"""
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_roi = None
        self.roi_history = deque(maxlen=5)
        
    def update(self, bbox, confidence=1.0):
        """更新ROI滤波器"""
        if bbox is None:
            return None
            
        x, y, w, h = bbox
        current_roi = [float(x), float(y), float(w), float(h)]
        
        if self.prev_roi is None:
            self.prev_roi = current_roi
            self.roi_history.append(current_roi)
            return [int(x) for x in current_roi]
        
        # 简单的指数移动平均
        smoothed_roi = []
        for i in range(4):
            smoothed_roi.append(self.alpha * current_roi[i] + (1 - self.alpha) * self.prev_roi[i])
        
        self.prev_roi = smoothed_roi
        self.roi_history.append(smoothed_roi)
        
        return [int(x) for x in smoothed_roi]

class SimpleOpticalFlowDetector:
    """简化的光流检测器"""
    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        self.camera_move_threshold = config['optical_flow_settings']['camera_move_threshold']
        
        # Lucas-Kanade光流参数
        self.lk_params = dict(
            winSize=(config['optical_flow_settings']['lk_win_size'], 
                    config['optical_flow_settings']['lk_win_size']),
            maxLevel=config['optical_flow_settings']['lk_max_level'],
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 特征点检测参数
        self.feature_params = dict(
            maxCorners=config['optical_flow_settings']['feature_max_corners'],
            qualityLevel=config['optical_flow_settings']['feature_quality_level'],
            minDistance=config['optical_flow_settings']['feature_min_distance'],
            blockSize=7
        )
    
    def detect_camera_movement(self, frame):
        """检测相机移动"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False, (0.0, 0.0)
        
        # 检测特征点
        if self.prev_points is None or len(self.prev_points) < 20:
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)
        
        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_gray = gray
            return False, (0.0, 0.0)
        
        # 计算光流
        try:
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )
            
            # 筛选好的点
            good_new = next_points[status == 1]
            good_old = self.prev_points[status == 1]
            
            if len(good_new) < 5:
                self.prev_gray = gray
                self.prev_points = None
                return False, (0.0, 0.0)
            
            # 计算平均光流
            flow_vectors = good_new - good_old
            avg_flow = np.mean(flow_vectors, axis=0)
            
            # 计算运动幅度
            movement_magnitude = np.linalg.norm(avg_flow)
            
            # 判断是否为相机移动
            is_camera_moving = movement_magnitude > self.camera_move_threshold
            
            # 更新状态
            self.prev_gray = gray
            self.prev_points = good_new.reshape(-1, 1, 2)
            
            return is_camera_moving, (float(avg_flow[0]), float(avg_flow[1]))
            
        except Exception as e:
            self.prev_gray = gray
            self.prev_points = None
            return False, (0.0, 0.0)

class SimplePoseActionDetector:
    """简化的姿态动作检测器"""
    def __init__(self):
        self.pose_history = deque(maxlen=config['action_detection']['pose_history_length'])
        self.action_history = deque(maxlen=config['action_detection']['action_smooth_frames'])
        self.velocity_history = deque(maxlen=config['action_detection']['velocity_smooth_frames'])
        
        self.key_mapping = config['key_mapping']
        self.last_simulated_action = None
        
        # 动作检测阈值
        self.jump_threshold = config['action_detection']['jump_threshold']
        self.move_threshold = config['action_detection']['move_threshold']
        
    def extract_key_points(self, image):
        """提取关键点"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)
            
            if results.pose_landmarks:
                landmarks = []
                confidences = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.visibility])
                    confidences.append(landmark.visibility)
                return np.array(landmarks), results.pose_landmarks, np.mean(confidences)
            
            return None, None, 0.0
        except Exception as e:
            return None, None, 0.0
    
    def calculate_pose_features(self, landmarks, image_shape):
        """计算姿态特征"""
        try:
            h, w = image_shape[:2]
            features = {}
            
            # 关键点像素坐标
            key_points = {
                'left_ankle': landmarks[Config.LEFT_ANKLE] * [w, h, 1],
                'right_ankle': landmarks[Config.RIGHT_ANKLE] * [w, h, 1],
                'left_knee': landmarks[Config.LEFT_KNEE] * [w, h, 1],
                'right_knee': landmarks[Config.RIGHT_KNEE] * [w, h, 1],
                'left_hip': landmarks[Config.LEFT_HIP] * [w, h, 1],
                'right_hip': landmarks[Config.RIGHT_HIP] * [w, h, 1],
                'left_shoulder': landmarks[Config.LEFT_SHOULDER] * [w, h, 1],
                'right_shoulder': landmarks[Config.RIGHT_SHOULDER] * [w, h, 1],
                'nose': landmarks[Config.NOSE] * [w, h, 1]
            }
            
            # 计算中心点
            hip_center = (key_points['left_hip'][:2] + key_points['right_hip'][:2]) / 2
            shoulder_center = (key_points['left_shoulder'][:2] + key_points['right_shoulder'][:2]) / 2
            ankle_center = (key_points['left_ankle'][:2] + key_points['right_ankle'][:2]) / 2
            
            # 身体高度和宽度
            body_height = np.linalg.norm(shoulder_center - ankle_center)
            body_width = np.linalg.norm(key_points['left_shoulder'][:2] - key_points['right_shoulder'][:2])
            
            features.update({
                'hip_center': hip_center,
                'shoulder_center': shoulder_center,
                'ankle_center': ankle_center,
                'body_height': body_height,
                'body_width': body_width,
                'key_points': key_points
            })
            
            return features
        except Exception as e:
            return None
    
    def detect_action(self, landmarks, image_shape, camera_movement, is_camera_moving, confidence):
        """检测动作"""
        if landmarks is None:
            return "idle"
        
        try:
            # 计算姿态特征
            current_features = self.calculate_pose_features(landmarks, image_shape)
            
            if current_features is None:
                return "idle"
            
            # 更新历史记录
            self.pose_history.append(current_features)
            
            if len(self.pose_history) < 2:
                return "idle"
            
            # 相机移动时降低检测敏感度
            if is_camera_moving:
                return "idle"  # 简化处理：相机移动时不检测动作
            
            # 正常动作检测
            action = self._classify_action_simple(current_features)
            
            # 动作平滑
            smoothed_action = self._smooth_action_simple(action)
            
            # 模拟按键
            if smoothed_action and smoothed_action != "idle":
                self.simulate_key_press(smoothed_action)
            
            return smoothed_action or "idle"
            
        except Exception as e:
            return "idle"
    
    def _classify_action_simple(self, current_features):
        """简化的动作分类"""
        if len(self.pose_history) < 2:
            return "idle"
        
        try:
            prev_features = self.pose_history[-2]
            
            # 计算运动特征
            ankle_velocity = current_features['ankle_center'] - prev_features['ankle_center']
            hip_velocity = current_features['hip_center'] - prev_features['hip_center']
            
            # 更新速度历史
            self.velocity_history.append(ankle_velocity)
            
            # 垂直运动检测（跳跃/着陆）
            vertical_movement = -ankle_velocity[1]  # Y轴向上为负
            
            # 水平运动检测
            horizontal_movement = ankle_velocity[0]
            
            # 动作判定逻辑
            if vertical_movement > self.jump_threshold:
                return "Jump"
            elif vertical_movement < -self.jump_threshold:
                return "Landing"
            elif abs(horizontal_movement) > self.move_threshold:
                if horizontal_movement > 0:
                    return "Move Right"
                else:
                    return "Move Left"
            elif len(self.pose_history) >= 3:
                # 前后移动检测（基于身体宽度变化）
                body_width_change = current_features['body_width'] - prev_features['body_width']
                if abs(body_width_change) > 3:
                    if body_width_change > 0:
                        return "Move Backward"
                    else:
                        return "Move Forward"
            
            return "idle"
            
        except Exception as e:
            return "idle"
    
    def _smooth_action_simple(self, action):
        """简化的动作平滑"""
        self.action_history.append(action)
        
        smooth_frames = config['action_detection']['action_smooth_frames']
        if len(self.action_history) < smooth_frames:
            return "idle"
        
        # 简单投票机制
        action_counts = {}
        for a in self.action_history:
            action_counts[a] = action_counts.get(a, 0) + 1
        
        if action_counts:
            most_common_action = max(action_counts, key=action_counts.get)
            if action_counts[most_common_action] >= smooth_frames * 0.5:  # 降低阈值
                return most_common_action
        
        return "idle"
    
    def simulate_key_press(self, action):
        """模拟按键按下"""
        if action in self.key_mapping and self.key_mapping[action]:
            key = self.key_mapping[action]
            print(f"模拟按键: {key} (动作: {action})")
            self.last_simulated_action = action

class SimpleOptimizedGameActionRecognizer:
    """简化优化的游戏动作识别器"""
    def __init__(self):
        # 初始化组件
        self.yolo_model = YOLO(config['model_settings']['yolo_model_path'])
        self.roi_filter = SimpleROIFilter()
        self.optical_flow = SimpleOpticalFlowDetector()
        self.pose_detector = SimplePoseActionDetector()
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'person_detected_frames': 0,
            'actions_detected': 0,
            'camera_movement_frames': 0,
            'average_confidence': 0.0,
            'processing_times': []
        }
        
        # 动作日志
        self.actions_log = []
        
        # 性能监控
        self.frame_times = deque(maxlen=30)
        
    def detect_person_roi(self, frame):
        """检测人物ROI"""
        try:
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # YOLO检测
            results = self.yolo_model(
                frame,
                conf=config['model_settings']['confidence_threshold'],
                iou=config['model_settings'].get('iou_threshold', 0.5),
                max_det=config['model_settings'].get('max_detections', 1),
                verbose=False
            )
            
            best_roi = None
            best_confidence = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 只检测人物（class 0）
                        if int(box.cls) == 0:
                            confidence = float(box.conf)
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # 计算与中心区域的距离
                            roi_center_x = (x1 + x2) / 2
                            roi_center_y = (y1 + y2) / 2
                            
                            center_distance = math.sqrt(
                                (roi_center_x - center_x) ** 2 + 
                                (roi_center_y - center_y) ** 2
                            )
                            
                            # 综合评分：置信度 - 中心距离权重
                            score = confidence - center_distance / (w + h) * 0.3
                            
                            if score > best_confidence:
                                best_confidence = score
                                
                                # 扩展ROI
                                roi_w = x2 - x1
                                roi_h = y2 - y1
                                expand_ratio = config['roi_settings']['roi_expand_ratio']
                                
                                expand_w = roi_w * (expand_ratio - 1) / 2
                                expand_h = roi_h * (expand_ratio - 1) / 2
                                
                                expanded_x1 = max(0, int(x1 - expand_w))
                                expanded_y1 = max(0, int(y1 - expand_h))
                                expanded_x2 = min(w, int(x2 + expand_w))
                                expanded_y2 = min(h, int(y2 + expand_h))
                                
                                best_roi = (expanded_x1, expanded_y1, 
                                          expanded_x2 - expanded_x1, 
                                          expanded_y2 - expanded_y1)
            
            if best_roi:
                # ROI大小验证
                roi_w, roi_h = best_roi[2], best_roi[3]
                min_size = config['roi_settings'].get('roi_min_size', 60)
                max_size = config['roi_settings'].get('roi_max_size', 500)
                
                if roi_w < min_size or roi_h < min_size or roi_w > max_size or roi_h > max_size:
                    return None, 0
                
                # 应用ROI滤波
                filtered_roi = self.roi_filter.update(best_roi, best_confidence)
                return filtered_roi, best_confidence
            
            return None, 0
            
        except Exception as e:
            return None, 0
    
    def process_frame(self, frame):
        """处理单帧"""
        start_time = time.time()
        
        self.stats['total_frames'] += 1
        
        # 检测人物ROI
        roi, roi_confidence = self.detect_person_roi(frame)
        
        if roi is None:
            return frame, "idle", False, (0.0, 0.0)
        
        self.stats['person_detected_frames'] += 1
        
        # 提取ROI区域
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        
        if roi_frame.size == 0:
            return frame, "idle", False, (0.0, 0.0)
        
        # 检测相机移动
        is_camera_moving, camera_movement = self.optical_flow.detect_camera_movement(frame)
        
        if is_camera_moving:
            self.stats['camera_movement_frames'] += 1
        
        # 提取姿态关键点
        landmarks, pose_landmarks, pose_confidence = self.pose_detector.extract_key_points(roi_frame)
        
        # 检测动作
        action = "idle"
        if landmarks is not None:
            # 综合置信度
            combined_confidence = (roi_confidence + pose_confidence) / 2
            self.stats['average_confidence'] = (
                self.stats['average_confidence'] * (self.stats['person_detected_frames'] - 1) + 
                combined_confidence
            ) / self.stats['person_detected_frames']
            
            action = self.pose_detector.detect_action(
                landmarks, roi_frame.shape, camera_movement, 
                is_camera_moving, combined_confidence
            )
            
            if action != "idle":
                self.stats['actions_detected'] += 1
                
                # 记录动作日志
                self.actions_log.append({
                    'timestamp': time.time(),
                    'frame': self.stats['total_frames'],
                    'action': action,
                    'simulated_key': self.pose_detector.key_mapping.get(action),
                    'roi': roi,
                    'roi_confidence': float(roi_confidence),
                    'pose_confidence': float(pose_confidence),
                    'combined_confidence': float(combined_confidence),
                    'camera_moving': is_camera_moving,
                    'camera_movement': camera_movement
                })
        
        # 性能监控
        processing_time = time.time() - start_time
        self.frame_times.append(processing_time)
        self.stats['processing_times'].append(processing_time)
        
        # 绘制注释
        annotated_frame = self.draw_annotations(
            frame, roi, pose_landmarks, action, is_camera_moving, 
            camera_movement, roi_confidence, pose_confidence
        )
        
        return annotated_frame, action, is_camera_moving, camera_movement
    
    def draw_annotations(self, frame, roi, pose_landmarks, action, is_camera_moving, 
                        camera_movement, roi_confidence=0, pose_confidence=0):
        """绘制注释信息"""
        annotated_frame = frame.copy()
        
        # 绘制ROI
        if roi is not None and config['display_settings']['show_roi']:
            x, y, w, h = roi
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 显示置信度
            if config['display_settings']['show_confidence']:
                cv2.putText(annotated_frame, f'ROI: {roi_confidence:.2f}', 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制姿态关键点
        if pose_landmarks is not None and config['display_settings']['show_pose']:
            x, y, w, h = roi if roi else (0, 0, frame.shape[1], frame.shape[0])
            
            # 调整关键点坐标到原图
            for landmark in pose_landmarks.landmark:
                px = int(landmark.x * w + x)
                py = int(landmark.y * h + y)
                cv2.circle(annotated_frame, (px, py), 3, (255, 0, 0), -1)
        
        # 显示动作信息
        if config['display_settings']['show_action_text']:
            action_text = f"Action: {action}"
            cv2.putText(annotated_frame, action_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # 显示姿态置信度
            if config['display_settings']['show_confidence']:
                conf_text = f"Pose Conf: {pose_confidence:.2f}"
                cv2.putText(annotated_frame, conf_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # 显示相机移动信息
        if is_camera_moving:
            camera_text = f"Camera Moving: {camera_movement[0]:.1f}, {camera_movement[1]:.1f}"
            cv2.putText(annotated_frame, camera_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # 显示性能信息
        if len(self.frame_times) > 0:
            avg_time = np.mean(list(self.frame_times))
            fps = 1.0 / avg_time if avg_time > 0 else 0
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated_frame
    
    def save_log(self, filename="pose_actions_log_simple_optimized.json"):
        """保存动作日志"""
        try:
            # 计算最终统计
            final_stats = self.stats.copy()
            if final_stats['processing_times']:
                final_stats['average_processing_time'] = np.mean(final_stats['processing_times'])
                final_stats['max_processing_time'] = np.max(final_stats['processing_times'])
                final_stats['min_processing_time'] = np.min(final_stats['processing_times'])
                # 移除详细时间记录以减少文件大小
                del final_stats['processing_times']
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'stats': final_stats,
                    'actions': self.actions_log,
                    'config_used': config
                }, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            print(f"简化优化版动作日志已保存到 {filename}")
        except Exception as e:
            print(f"保存日志失败: {e}")

def main():
    """主函数"""
    print("=== YOLO + MediaPipe Pose 游戏动作识别系统 (简化优化版) ===")
    print(f"配置文件: {config.get('model_settings', {}).get('yolo_model_path', 'N/A')}")
    print(f"视频文件: {config.get('model_settings', {}).get('video_path', 'N/A')}")
    
    # 初始化识别器
    recognizer = SimpleOptimizedGameActionRecognizer()
    
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
    
    # 创建窗口
    window_name = config['display_settings']['window_name']
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    print("\n控制说明:")
    print("- 按 'q' 退出")
    print("- 按 'p' 暂停/继续")
    print("- 按 'r' 重置统计")
    print("- 按 's' 保存当前日志")
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                annotated_frame, action, is_camera_moving, camera_movement = recognizer.process_frame(frame)
                
                # 显示结果
                cv2.imshow(window_name, annotated_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'暂停' if paused else '继续'}播放")
            elif key == ord('r'):
                recognizer.stats = {
                    'total_frames': 0,
                    'person_detected_frames': 0,
                    'actions_detected': 0,
                    'camera_movement_frames': 0,
                    'average_confidence': 0.0,
                    'processing_times': []
                }
                recognizer.actions_log = []
                print("统计信息已重置")
            elif key == ord('s'):
                recognizer.save_log()
                print("日志已保存")
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    finally:
        # 保存最终日志
        recognizer.save_log()
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 显示最终统计
        stats = recognizer.stats
        print("\n=== 最终统计 (简化优化版) ===")
        print(f"总帧数: {stats['total_frames']}")
        print(f"检测到人物的帧数: {stats['person_detected_frames']}")
        print(f"检测到动作的次数: {stats['actions_detected']}")
        print(f"相机移动的帧数: {stats['camera_movement_frames']}")
        
        if stats['total_frames'] > 0:
            print(f"人物检测率: {stats['person_detected_frames']/stats['total_frames']*100:.2f}%")
            print(f"相机移动率: {stats['camera_movement_frames']/stats['total_frames']*100:.2f}%")
            print(f"平均置信度: {stats['average_confidence']:.3f}")
        
        if stats['processing_times']:
            avg_time = np.mean(stats['processing_times'])
            print(f"平均处理时间: {avg_time*1000:.2f}ms")
            print(f"平均FPS: {1.0/avg_time:.1f}")

if __name__ == "__main__":
    main()