#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + MediaPipe Pose 游戏动作识别系统 - 优化版本

主要优化:
1. 更精确的人物检测和ROI稳定化
2. 改进的动作识别算法
3. 更准确的镜头移动检测
4. 多帧验证和时序平滑
5. 自适应阈值调整
6. 置信度加权和异常值检测

作者: AI Assistant
版本: 2.0 (优化版)
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
from scipy import signal
from sklearn.cluster import DBSCAN

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
                "confidence_threshold": 0.7,
                "iou_threshold": 0.5,
                "max_detections": 1
            },
            "roi_settings": {
                "center_region_ratio": 0.8,
                "roi_expand_ratio": 1.2,
                "roi_stability_threshold": 50,
                "roi_min_size": 80,
                "roi_max_size": 400
            },
            "optical_flow_settings": {
                "optical_flow_points": 200,
                "camera_move_threshold": 3.0,
                "lk_win_size": 21,
                "lk_max_level": 3,
                "feature_max_corners": 200,
                "feature_quality_level": 0.01,
                "feature_min_distance": 10,
                "flow_magnitude_threshold": 2.0,
                "flow_consistency_threshold": 0.8
            },
            "action_detection": {
                "jump_threshold": 20,
                "move_threshold": 12,
                "idle_threshold": 5,
                "action_smooth_frames": 5,
                "pose_history_length": 15,
                "velocity_smooth_frames": 3,
                "acceleration_threshold": 8,
                "position_change_threshold": 10
            },
            "mediapipe_pose": {
                "model_complexity": 2,
                "min_detection_confidence": 0.7,
                "min_tracking_confidence": 0.7,
                "enable_segmentation": False,
                "smooth_landmarks": True
            },
            "genshin_optimization": {
                "fixed_center_detection": True,
                "breathing_filter_threshold": 1.5,
                "camera_rotation_filter": True,
                "action_consistency_frames": 4,
                "multi_frame_validation": True,
                "pose_confidence_threshold": 0.6,
                "landmark_stability_check": True
            },
            "advanced_filtering": {
                "temporal_smoothing": True,
                "outlier_detection": True,
                "confidence_weighting": True,
                "adaptive_thresholds": True,
                "noise_reduction": True
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
                "window_name": "YOLO + MediaPipe Pose Detection (Optimized)",
                "show_roi": True,
                "show_pose": True,
                "show_optical_flow": True,
                "show_action_text": True,
                "show_confidence": True
            },
            "logging": {
                "enable_logging": True,
                "log_level": "INFO",
                "save_detailed_logs": True,
                "performance_monitoring": True
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

class AdvancedROIFilter:
    """高级ROI滤波器，结合多种滤波技术"""
    def __init__(self, alpha=0.7, stability_threshold=50):
        self.alpha = alpha
        self.stability_threshold = stability_threshold
        self.roi_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        self.prev_roi = None
        self.stable_roi = None
        self.initialized = False
        
    def update(self, bbox, confidence=1.0):
        """更新ROI滤波器"""
        bbox = np.array(bbox, dtype=np.float32)
        
        if not self.initialized or self.prev_roi is None:
            self.prev_roi = bbox.copy()
            self.stable_roi = bbox.copy()
            self.initialized = True
            self.roi_history.append(bbox)
            self.confidence_history.append(confidence)
            return bbox.astype(int)
        
        # 计算移动距离
        movement = np.linalg.norm(bbox[:2] - self.prev_roi[:2])
        
        # 异常值检测
        if movement > self.stability_threshold:
            # 可能是异常值，使用历史数据预测
            if len(self.roi_history) >= 3:
                predicted_roi = self._predict_roi()
                bbox = predicted_roi.copy()
        
        # 置信度加权平滑
        weight = confidence * self.alpha
        smoothed_roi = weight * bbox + (1 - weight) * self.prev_roi
        
        # 更新历史记录
        self.roi_history.append(smoothed_roi)
        self.confidence_history.append(confidence)
        
        # 计算速度
        if len(self.roi_history) >= 2:
            velocity = self.roi_history[-1][:2] - self.roi_history[-2][:2]
            self.velocity_history.append(velocity)
        
        self.prev_roi = smoothed_roi.copy()
        
        # 更新稳定ROI
        if movement < self.stability_threshold * 0.3:
            self.stable_roi = 0.9 * self.stable_roi + 0.1 * smoothed_roi
        
        return smoothed_roi.astype(int)
    
    def _predict_roi(self):
        """基于历史数据预测ROI位置"""
        if len(self.roi_history) < 3:
            return self.prev_roi
        
        # 使用线性预测
        recent_rois = np.array(list(self.roi_history)[-3:])
        if len(self.velocity_history) > 0:
            avg_velocity = np.mean(self.velocity_history, axis=0)
            predicted_pos = self.prev_roi[:2] + avg_velocity
            predicted_roi = self.prev_roi.copy()
            predicted_roi[:2] = predicted_pos
            return predicted_roi
        
        return self.prev_roi

class EnhancedOpticalFlowDetector:
    """增强的光流检测器"""
    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        self.flow_history = deque(maxlen=10)
        self.camera_movement_history = deque(maxlen=5)
        
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
        
        self.camera_move_threshold = config['optical_flow_settings']['camera_move_threshold']
        self.flow_magnitude_threshold = config['optical_flow_settings']['flow_magnitude_threshold']
        self.flow_consistency_threshold = config['optical_flow_settings']['flow_consistency_threshold']
    
    def detect_camera_movement(self, frame):
        """检测相机移动"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False, (0, 0)
        
        # 检测特征点
        if self.prev_points is None or len(self.prev_points) < 50:
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)
        
        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_gray = gray
            return False, (0, 0)
        
        # 计算光流
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        # 筛选好的点
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]
        
        if len(good_new) < 10:
            self.prev_gray = gray
            self.prev_points = None
            return False, (0, 0)
        
        # 计算光流向量
        flow_vectors = good_new - good_old
        
        # 异常值检测和过滤
        flow_magnitudes = np.linalg.norm(flow_vectors, axis=1)
        valid_flows = flow_vectors[flow_magnitudes < 50]  # 过滤过大的流向量
        
        if len(valid_flows) < 5:
            self.prev_gray = gray
            self.prev_points = good_new.reshape(-1, 1, 2)
            return False, (0, 0)
        
        # 使用DBSCAN聚类检测一致性运动
        if len(valid_flows) > 10:
            clustering = DBSCAN(eps=3, min_samples=3).fit(valid_flows)
            labels = clustering.labels_
            
            # 找到最大的聚类
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:  # 有聚类结果
                largest_cluster = -1
                largest_size = 0
                for label in unique_labels:
                    if label != -1:  # 不是噪声点
                        cluster_size = np.sum(labels == label)
                        if cluster_size > largest_size:
                            largest_size = cluster_size
                            largest_cluster = label
                
                if largest_cluster != -1 and largest_size >= len(valid_flows) * 0.3:
                    # 使用最大聚类的平均运动
                    cluster_flows = valid_flows[labels == largest_cluster]
                    avg_flow = np.mean(cluster_flows, axis=0)
                else:
                    avg_flow = np.mean(valid_flows, axis=0)
            else:
                avg_flow = np.mean(valid_flows, axis=0)
        else:
            avg_flow = np.mean(valid_flows, axis=0)
        
        # 计算运动幅度
        movement_magnitude = np.linalg.norm(avg_flow)
        
        # 更新历史记录
        self.flow_history.append(avg_flow)
        self.camera_movement_history.append(movement_magnitude)
        
        # 时序平滑
        if len(self.camera_movement_history) >= 3:
            smoothed_magnitude = np.median(list(self.camera_movement_history))
        else:
            smoothed_magnitude = movement_magnitude
        
        # 判断是否为相机移动
        is_camera_moving = smoothed_magnitude > self.camera_move_threshold
        
        # 更新状态
        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)
        
        return is_camera_moving, tuple(avg_flow)

class AdvancedPoseActionDetector:
    """高级姿态动作检测器"""
    def __init__(self):
        self.pose_history = deque(maxlen=config['action_detection']['pose_history_length'])
        self.action_history = deque(maxlen=config['action_detection']['action_smooth_frames'])
        self.velocity_history = deque(maxlen=config['action_detection']['velocity_smooth_frames'])
        self.confidence_history = deque(maxlen=10)
        
        self.key_mapping = config['key_mapping']
        self.last_simulated_action = None
        
        # 动态阈值
        self.adaptive_jump_threshold = config['action_detection']['jump_threshold']
        self.adaptive_move_threshold = config['action_detection']['move_threshold']
        
        # 多帧验证
        self.action_candidates = deque(maxlen=5)
        
    def extract_key_points(self, image):
        """提取关键点"""
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
    
    def calculate_pose_features(self, landmarks, image_shape):
        """计算姿态特征"""
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
        
        # 腿部弯曲度
        left_leg_angle = self._calculate_angle(
            key_points['left_hip'][:2], 
            key_points['left_knee'][:2], 
            key_points['left_ankle'][:2]
        )
        right_leg_angle = self._calculate_angle(
            key_points['right_hip'][:2], 
            key_points['right_knee'][:2], 
            key_points['right_ankle'][:2]
        )
        
        features.update({
            'hip_center': hip_center,
            'shoulder_center': shoulder_center,
            'ankle_center': ankle_center,
            'body_height': body_height,
            'body_width': body_width,
            'left_leg_angle': left_leg_angle,
            'right_leg_angle': right_leg_angle,
            'key_points': key_points
        })
        
        return features
    
    def _calculate_angle(self, p1, p2, p3):
        """计算三点之间的角度"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def detect_action(self, landmarks, image_shape, camera_movement, is_camera_moving, confidence):
        """检测动作"""
        if landmarks is None:
            return "idle"
        
        # 计算姿态特征
        current_features = self.calculate_pose_features(landmarks, image_shape)
        
        # 更新历史记录
        self.pose_history.append(current_features)
        self.confidence_history.append(confidence)
        
        if len(self.pose_history) < 3:
            return "idle"
        
        # 相机移动时降低检测敏感度
        if is_camera_moving:
            return self._detect_action_camera_moving(current_features)
        
        # 正常动作检测
        action = self._classify_action_advanced(current_features)
        
        # 多帧验证
        if config['genshin_optimization']['multi_frame_validation']:
            action = self._validate_action_multi_frame(action)
        
        # 动作平滑
        smoothed_action = self._smooth_action_advanced(action)
        
        # 模拟按键
        if smoothed_action and smoothed_action != "idle":
            self.simulate_key_press(smoothed_action)
        
        return smoothed_action or "idle"
    
    def _detect_action_camera_moving(self, current_features):
        """相机移动时的动作检测"""
        # 提高阈值，降低误检
        if len(self.pose_history) < 5:
            return "idle"
        
        prev_features = self.pose_history[-2]
        
        # 只检测明显的动作
        ankle_movement = np.linalg.norm(
            current_features['ankle_center'] - prev_features['ankle_center']
        )
        
        if ankle_movement > self.adaptive_jump_threshold * 1.5:
            return "Jump"
        
        return "idle"
    
    def _classify_action_advanced(self, current_features):
        """高级动作分类"""
        if len(self.pose_history) < 3:
            return "idle"
        
        prev_features = self.pose_history[-2]
        prev2_features = self.pose_history[-3]
        
        # 计算运动特征
        ankle_velocity = current_features['ankle_center'] - prev_features['ankle_center']
        ankle_acceleration = ankle_velocity - (prev_features['ankle_center'] - prev2_features['ankle_center'])
        
        hip_velocity = current_features['hip_center'] - prev_features['hip_center']
        
        # 更新速度历史
        self.velocity_history.append(ankle_velocity)
        
        # 计算平均速度（时序平滑）
        if len(self.velocity_history) >= 3:
            smoothed_velocity = np.mean(list(self.velocity_history)[-3:], axis=0)
        else:
            smoothed_velocity = ankle_velocity
        
        # 自适应阈值调整
        self._update_adaptive_thresholds()
        
        # 垂直运动检测（跳跃/着陆）
        vertical_movement = -smoothed_velocity[1]  # Y轴向上为负
        vertical_acceleration = -ankle_acceleration[1]
        
        # 水平运动检测
        horizontal_movement = smoothed_velocity[0]
        
        # 腿部角度变化
        leg_angle_change = abs(
            (current_features['left_leg_angle'] + current_features['right_leg_angle']) / 2 -
            (prev_features['left_leg_angle'] + prev_features['right_leg_angle']) / 2
        )
        
        # 动作判定逻辑
        if vertical_movement > self.adaptive_jump_threshold and vertical_acceleration > 5:
            return "Jump"
        elif vertical_movement < -self.adaptive_jump_threshold and vertical_acceleration < -5:
            return "Landing"
        elif abs(horizontal_movement) > self.adaptive_move_threshold:
            if horizontal_movement > 0:
                return "Move Right"
            else:
                return "Move Left"
        elif len(self.pose_history) >= 5:
            # 前后移动检测（基于身体宽度变化）
            body_width_change = current_features['body_width'] - prev_features['body_width']
            if abs(body_width_change) > 5:
                if body_width_change > 0:
                    return "Move Backward"
                else:
                    return "Move Forward"
        
        return "idle"
    
    def _update_adaptive_thresholds(self):
        """更新自适应阈值"""
        if not config['advanced_filtering']['adaptive_thresholds']:
            return
        
        if len(self.velocity_history) >= 10:
            velocities = np.array(list(self.velocity_history))
            velocity_std = np.std(velocities, axis=0)
            
            # 根据运动变化调整阈值
            base_jump = config['action_detection']['jump_threshold']
            base_move = config['action_detection']['move_threshold']
            
            self.adaptive_jump_threshold = base_jump + velocity_std[1] * 0.5
            self.adaptive_move_threshold = base_move + velocity_std[0] * 0.3
            
            # 限制阈值范围
            self.adaptive_jump_threshold = np.clip(self.adaptive_jump_threshold, base_jump * 0.5, base_jump * 2)
            self.adaptive_move_threshold = np.clip(self.adaptive_move_threshold, base_move * 0.5, base_move * 2)
    
    def _validate_action_multi_frame(self, action):
        """多帧验证动作"""
        self.action_candidates.append(action)
        
        if len(self.action_candidates) < 3:
            return action
        
        # 统计最近几帧的动作
        recent_actions = list(self.action_candidates)[-3:]
        action_counts = {}
        for a in recent_actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        
        # 需要至少2/3的一致性
        for a, count in action_counts.items():
            if count >= 2 and a != "idle":
                return a
        
        return "idle"
    
    def _smooth_action_advanced(self, action):
        """高级动作平滑"""
        self.action_history.append(action)
        
        smooth_frames = config['action_detection']['action_smooth_frames']
        if len(self.action_history) < smooth_frames:
            return "idle"
        
        # 置信度加权
        if config['advanced_filtering']['confidence_weighting'] and len(self.confidence_history) >= smooth_frames:
            recent_confidences = list(self.confidence_history)[-smooth_frames:]
            recent_actions = list(self.action_history)[-smooth_frames:]
            
            # 计算加权投票
            weighted_votes = {}
            for i, (a, conf) in enumerate(zip(recent_actions, recent_confidences)):
                if a not in weighted_votes:
                    weighted_votes[a] = 0
                weighted_votes[a] += conf
            
            if weighted_votes:
                best_action = max(weighted_votes, key=weighted_votes.get)
                if weighted_votes[best_action] > smooth_frames * 0.4:
                    return best_action
        else:
            # 传统投票
            action_counts = {}
            for a in self.action_history:
                action_counts[a] = action_counts.get(a, 0) + 1
            
            if action_counts:
                most_common_action = max(action_counts, key=action_counts.get)
                if action_counts[most_common_action] >= smooth_frames * 0.6:
                    return most_common_action
        
        return "idle"
    
    def simulate_key_press(self, action):
        """模拟按键按下"""
        if action in self.key_mapping and self.key_mapping[action]:
            key = self.key_mapping[action]
            print(f"模拟按键: {key} (动作: {action})")
            self.last_simulated_action = action
            
            # 这里可以添加实际的按键模拟代码
            # keyboard.press(key)
            # keyboard.release(key)

class OptimizedGameActionRecognizer:
    """优化的游戏动作识别器"""
    def __init__(self):
        # 初始化组件
        self.yolo_model = YOLO(config['model_settings']['yolo_model_path'])
        self.roi_filter = AdvancedROIFilter()
        self.optical_flow = EnhancedOpticalFlowDetector()
        self.pose_detector = AdvancedPoseActionDetector()
        
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
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 定义中心区域
        center_ratio = config['roi_settings']['center_region_ratio']
        center_w = int(w * center_ratio)
        center_h = int(h * center_ratio)
        
        center_region = (
            center_x - center_w // 2,
            center_y - center_h // 2,
            center_w,
            center_h
        )
        
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
                        
                        # 计算与中心区域的重叠
                        roi_center_x = (x1 + x2) / 2
                        roi_center_y = (y1 + y2) / 2
                        
                        # 优先选择中心区域的检测结果
                        center_distance = np.sqrt(
                            (roi_center_x - center_x) ** 2 + 
                            (roi_center_y - center_y) ** 2
                        )
                        
                        # 综合评分：置信度 + 中心位置
                        score = confidence - center_distance / (w + h) * 0.5
                        
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
            roi_w, roi_h = int(best_roi[2]), int(best_roi[3])
            min_size = config['roi_settings'].get('roi_min_size', 80)
            max_size = config['roi_settings'].get('roi_max_size', 400)
            
            if roi_w < min_size or roi_h < min_size or roi_w > max_size or roi_h > max_size:
                return None, 0
            
            # 应用ROI滤波
            filtered_roi = self.roi_filter.update(best_roi, best_confidence)
            return filtered_roi, best_confidence
        
        return None, 0
    
    def process_frame(self, frame):
        """处理单帧"""
        start_time = time.time()
        
        self.stats['total_frames'] += 1
        
        # 检测人物ROI
        roi, roi_confidence = self.detect_person_roi(frame)
        
        if roi is None:
            return frame, "idle", False, (0, 0)
        
        self.stats['person_detected_frames'] += 1
        
        # 提取ROI区域
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        roi_frame = frame[y:y+h, x:x+w]
        
        if roi_frame.size == 0:
            return frame, "idle", False, (0, 0)
        
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
    
    def save_log(self, filename="pose_actions_log_optimized.json"):
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
            print(f"优化版动作日志已保存到 {filename}")
        except Exception as e:
            print(f"保存日志失败: {e}")

def main():
    """主函数"""
    print("=== YOLO + MediaPipe Pose 游戏动作识别系统 (优化版) ===")
    print(f"配置文件: {config.get('model_settings', {}).get('yolo_model_path', 'N/A')}")
    print(f"视频文件: {config.get('model_settings', {}).get('video_path', 'N/A')}")
    
    # 初始化识别器
    recognizer = OptimizedGameActionRecognizer()
    
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
    print("- 按 'c' 清空动作历史")
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
            elif key == ord('c'):
                recognizer.pose_detector.pose_history.clear()
                recognizer.pose_detector.action_history.clear()
                recognizer.pose_detector.velocity_history.clear()
                print("动作历史已清空")
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
        print("\n=== 最终统计 (优化版) ===")
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