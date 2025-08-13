#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + MediaPipe Pose 动作识别系统 - 最终优化版
结合YOLO人物检测和MediaPipe姿态估计进行动作识别
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import json
import time
import keyboard
from collections import deque
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import threading
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """配置类"""
    def __init__(self, config_path: str = "pose_config_final.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 模型设置
        model_settings = config['model_settings']
        self.yolo_model_path = model_settings['yolo_model_path']
        self.confidence_threshold = model_settings['confidence_threshold']
        self.iou_threshold = model_settings['iou_threshold']
        self.max_detections = model_settings['max_detections']
        self.device = model_settings['device']
        
        # ROI设置
        roi_settings = config['roi_settings']
        self.roi_expansion_factor = roi_settings['roi_expansion_factor']
        self.roi_smoothing_factor = roi_settings['roi_smoothing_factor']
        self.min_roi_size = tuple(roi_settings['min_roi_size'])
        self.max_roi_size = tuple(roi_settings['max_roi_size'])
        self.roi_stability_threshold = roi_settings['roi_stability_threshold']
        self.roi_update_threshold = roi_settings['roi_update_threshold']
        
        # 光流设置
        optical_flow = config['optical_flow_settings']
        self.feature_params = optical_flow['feature_params']
        self.lk_params = optical_flow['lk_params']
        self.movement_threshold = optical_flow['movement_threshold']
        self.min_features = optical_flow['min_features']
        
        # 动作检测
        action_detection = config['action_detection']
        self.jump_threshold = action_detection['jump_threshold']
        self.landing_threshold = action_detection['landing_threshold']
        self.move_threshold = action_detection['move_threshold']
        self.action_smoothing_frames = action_detection['action_smoothing_frames']
        self.min_action_confidence = action_detection['min_action_confidence']
        self.action_cooldown_frames = action_detection['action_cooldown_frames']
        self.velocity_smoothing_factor = action_detection['velocity_smoothing_factor']
        
        # MediaPipe设置
        mp_pose = config['mediapipe_pose']
        self.mp_static_image_mode = mp_pose['static_image_mode']
        self.mp_model_complexity = mp_pose['model_complexity']
        self.mp_smooth_landmarks = mp_pose['smooth_landmarks']
        self.mp_enable_segmentation = mp_pose['enable_segmentation']
        self.mp_smooth_segmentation = mp_pose['smooth_segmentation']
        self.mp_min_detection_confidence = mp_pose['min_detection_confidence']
        self.mp_min_tracking_confidence = mp_pose['min_tracking_confidence']
        
        # 原神优化
        genshin_opt = config['genshin_optimization']
        self.character_height_ratio = genshin_opt['character_height_ratio']
        self.jump_detection_sensitivity = genshin_opt['jump_detection_sensitivity']
        self.movement_detection_sensitivity = genshin_opt['movement_detection_sensitivity']
        self.landing_detection_sensitivity = genshin_opt['landing_detection_sensitivity']
        
        # 按键映射
        self.key_mapping = config['key_mapping']
        
        # 高级滤波
        advanced_filtering = config['advanced_filtering']
        self.kalman_process_noise = advanced_filtering['kalman_filter']['process_noise']
        self.kalman_measurement_noise = advanced_filtering['kalman_filter']['measurement_noise']
        self.kalman_estimation_error = advanced_filtering['kalman_filter']['estimation_error']
        
        temporal_smoothing = advanced_filtering['temporal_smoothing']
        self.position_smoothing = temporal_smoothing['position_smoothing']
        self.velocity_smoothing = temporal_smoothing['velocity_smoothing']
        self.confidence_smoothing = temporal_smoothing['confidence_smoothing']
        
        outlier_detection = advanced_filtering['outlier_detection']
        self.outlier_detection_enable = outlier_detection['enable']
        self.z_score_threshold = outlier_detection['z_score_threshold']
        self.min_samples = outlier_detection['min_samples']
        
        # 显示设置
        display_settings = config['display_settings']
        self.show_video = display_settings['show_video']
        self.show_roi = display_settings['show_roi']
        self.show_pose = display_settings['show_pose']
        self.show_optical_flow = display_settings['show_optical_flow']
        self.show_stats = display_settings['show_stats']
        self.video_scale = display_settings['video_scale']

class SimpleKalmanFilter:
    """简化的卡尔曼滤波器"""
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimation_error = 1.0
        self.last_estimate = None
        
    def update(self, measurement):
        if self.last_estimate is None:
            self.last_estimate = measurement
            return measurement
            
        # 预测
        prediction = self.last_estimate
        prediction_error = self.estimation_error + self.process_noise
        
        # 更新
        kalman_gain = prediction_error / (prediction_error + self.measurement_noise)
        estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimation_error = (1 - kalman_gain) * prediction_error
        
        self.last_estimate = estimate
        return estimate

class EnhancedROIFilter:
    """增强的ROI滤波器"""
    def __init__(self, config: Config):
        self.config = config
        self.roi_history = deque(maxlen=10)
        self.kalman_filters = [SimpleKalmanFilter() for _ in range(4)]  # x, y, w, h
        self.last_stable_roi = None
        self.stability_counter = 0
        
    def update(self, new_roi, confidence):
        if new_roi is None:
            return self.last_stable_roi
            
        # 应用卡尔曼滤波
        filtered_roi = np.array([
            self.kalman_filters[i].update(new_roi[i]) for i in range(4)
        ])
        
        # 尺寸约束
        filtered_roi[2] = np.clip(filtered_roi[2], self.config.min_roi_size[0], self.config.max_roi_size[0])
        filtered_roi[3] = np.clip(filtered_roi[3], self.config.min_roi_size[1], self.config.max_roi_size[1])
        
        # 稳定性检查
        if self.last_stable_roi is not None:
            distance = np.linalg.norm(filtered_roi[:2] - self.last_stable_roi[:2])
            if distance < self.config.roi_stability_threshold:
                self.stability_counter += 1
            else:
                self.stability_counter = 0
                
        # 更新稳定ROI
        if self.stability_counter > 3 or self.last_stable_roi is None:
            self.last_stable_roi = filtered_roi.copy()
            
        self.roi_history.append(filtered_roi)
        return filtered_roi.astype(int)

class EnhancedOpticalFlowDetector:
    """增强的光流检测器"""
    def __init__(self, config: Config):
        self.config = config
        self.prev_gray = None
        self.prev_points = None
        self.movement_history = deque(maxlen=10)
        
    def detect_camera_movement(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False, np.array([0.0, 0.0])
            
        # 检测特征点
        if self.prev_points is None or len(self.prev_points) < self.config.min_features:
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, **self.config.feature_params)
            
        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_gray = gray
            return False, np.array([0.0, 0.0])
            
        # 光流跟踪
        lk_params = self.config.lk_params.copy()
        lk_params['criteria'] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **lk_params
        )
        
        # 过滤好的点
        good_prev = self.prev_points[status == 1]
        good_next = next_points[status == 1]
        
        if len(good_prev) < 10:
            self.prev_gray = gray
            self.prev_points = None
            return False, np.array([0.0, 0.0])
            
        # 计算平均移动
        movement = np.mean(good_next - good_prev, axis=0)
        movement_magnitude = np.linalg.norm(movement)
        
        # 更新历史
        self.movement_history.append(movement)
        
        # 判断是否为相机移动
        is_moving = movement_magnitude > self.config.movement_threshold
        
        # 更新
        self.prev_gray = gray
        self.prev_points = good_next.reshape(-1, 1, 2)
        
        return is_moving, movement

class EnhancedPoseActionDetector:
    """增强的姿态动作检测器"""
    def __init__(self, config: Config):
        self.config = config
        self.pose_history = deque(maxlen=15)
        self.action_history = deque(maxlen=10)
        self.last_action_frame = -999
        self.velocity_history = deque(maxlen=8)
        
    def detect_action(self, landmarks, frame_count):
        if landmarks is None:
            return None, 0.0
            
        # 提取关键点
        key_points = self._extract_key_points(landmarks)
        self.pose_history.append(key_points)
        
        if len(self.pose_history) < 5:
            return None, 0.0
            
        # 计算速度
        velocity = self._calculate_velocity()
        self.velocity_history.append(velocity)
        
        # 动作冷却
        if frame_count - self.last_action_frame < self.config.action_cooldown_frames:
            return None, 0.0
            
        # 检测各种动作
        actions = [
            self._detect_jump(velocity),
            self._detect_landing(velocity),
            self._detect_horizontal_movement(velocity)
        ]
        
        # 选择最可能的动作
        best_action = None
        best_confidence = 0.0
        
        for action, confidence in actions:
            if action and confidence > best_confidence and confidence > self.config.min_action_confidence:
                best_action = action
                best_confidence = confidence
                
        if best_action:
            self.last_action_frame = frame_count
            self.action_history.append((best_action, best_confidence))
            
        return best_action, best_confidence
        
    def _extract_key_points(self, landmarks):
        """提取关键点"""
        key_indices = [11, 12, 23, 24, 25, 26, 27, 28]  # 肩膀、臀部、膝盖、脚踝
        points = []
        for idx in key_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                points.extend([lm.x, lm.y, lm.z])
            else:
                points.extend([0.0, 0.0, 0.0])
        return np.array(points)
        
    def _calculate_velocity(self):
        """计算速度"""
        if len(self.pose_history) < 3:
            return np.zeros(len(self.pose_history[-1]))
            
        # 使用Savitzky-Golay滤波器平滑数据
        poses = np.array(list(self.pose_history))
        if len(poses) >= 5:
            smoothed = savgol_filter(poses, 5, 2, axis=0)
            velocity = smoothed[-1] - smoothed[-3]
        else:
            velocity = poses[-1] - poses[-2]
            
        return velocity
        
    def _detect_jump(self, velocity):
        """检测跳跃"""
        # 检查脚踝和膝盖的向上运动
        ankle_velocity = np.mean([velocity[18], velocity[21]])  # 左右脚踝y速度
        knee_velocity = np.mean([velocity[15], velocity[18]])   # 左右膝盖y速度
        
        jump_score = -(ankle_velocity + knee_velocity) * self.config.jump_detection_sensitivity
        
        if jump_score > self.config.jump_threshold:
            return "Jump", min(jump_score / self.config.jump_threshold, 1.0)
        return None, 0.0
        
    def _detect_landing(self, velocity):
        """检测着陆"""
        # 检查脚踝的向下运动和减速
        ankle_velocity = np.mean([velocity[18], velocity[21]])  # 左右脚踝y速度
        
        landing_score = ankle_velocity * self.config.landing_detection_sensitivity
        
        if landing_score > self.config.landing_threshold:
            return "Landing", min(landing_score / self.config.landing_threshold, 1.0)
        return None, 0.0
        
    def _detect_horizontal_movement(self, velocity):
        """检测水平移动"""
        # 检查整体水平移动
        hip_velocity_x = np.mean([velocity[6], velocity[9]])    # 左右臀部x速度
        shoulder_velocity_x = np.mean([velocity[0], velocity[3]]) # 左右肩膀x速度
        
        horizontal_velocity = (hip_velocity_x + shoulder_velocity_x) / 2
        move_score = abs(horizontal_velocity) * self.config.movement_detection_sensitivity
        
        if move_score > self.config.move_threshold:
            if horizontal_velocity > 0:
                return "Move Right", min(move_score / self.config.move_threshold, 1.0)
            else:
                return "Move Left", min(move_score / self.config.move_threshold, 1.0)
                
        # 检查前后移动
        hip_velocity_z = np.mean([velocity[8], velocity[11]])    # 左右臀部z速度
        if abs(hip_velocity_z) * self.config.movement_detection_sensitivity > self.config.move_threshold:
            if hip_velocity_z > 0:
                return "Move Forward", min(abs(hip_velocity_z) / self.config.move_threshold, 1.0)
            else:
                return "Move Backward", min(abs(hip_velocity_z) / self.config.move_threshold, 1.0)
                
        return None, 0.0

class YOLOMediaPipePoseDetector:
    """YOLO + MediaPipe 姿态检测器"""
    def __init__(self, config_path: str = "pose_config_final.json"):
        self.config = Config(config_path)
        
        # 初始化模型
        self.yolo_model = YOLO(self.config.yolo_model_path)
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=self.config.mp_static_image_mode,
            model_complexity=self.config.mp_model_complexity,
            smooth_landmarks=self.config.mp_smooth_landmarks,
            enable_segmentation=self.config.mp_enable_segmentation,
            smooth_segmentation=self.config.mp_smooth_segmentation,
            min_detection_confidence=self.config.mp_min_detection_confidence,
            min_tracking_confidence=self.config.mp_min_tracking_confidence
        )
        
        # 初始化组件
        self.roi_filter = EnhancedROIFilter(self.config)
        self.optical_flow_detector = EnhancedOpticalFlowDetector(self.config)
        self.pose_action_detector = EnhancedPoseActionDetector(self.config)
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'person_detected_frames': 0,
            'actions_detected': 0,
            'camera_movement_frames': 0,
            'processing_times': [],
            'confidences': []
        }
        
        # 动作日志
        self.action_log = {'stats': {}, 'actions': []}
        
        # 控制变量
        self.paused = False
        self.running = True
        
    def simulate_key_press(self, action: str):
        """模拟按键"""
        if action in self.config.key_mapping:
            key = self.config.key_mapping[action]
            print(f"模拟按键: {key} (动作: {action})")
            # keyboard.press_and_release(key)  # 取消注释以启用实际按键模拟
            
    def process_video(self, video_path: str):
        """处理视频"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return
            
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}")
        print("\n控制说明:")
        print("- 按 'q' 退出")
        print("- 按 'p' 暂停/继续")
        print("- 按 'r' 重置统计")
        print("- 按 's' 保存当前日志")
        
        frame_count = 0
        
        try:
            while self.running and cap.isOpened():
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    start_time = time.time()
                    
                    # 处理帧
                    self._process_frame(frame, frame_count)
                    
                    processing_time = time.time() - start_time
                    self.stats['processing_times'].append(processing_time)
                    
                    frame_count += 1
                    self.stats['total_frames'] = frame_count
                    
                    # 显示进度
                    if frame_count % 100 == 0:
                        progress = (frame_count / total_frames) * 100
                        avg_fps = 1.0 / np.mean(self.stats['processing_times'][-100:]) if self.stats['processing_times'] else 0
                        print(f"\r处理进度: {progress:.1f}% ({frame_count}/{total_frames}), 平均FPS: {avg_fps:.1f}", end="")
                        
                # 检查键盘输入
                self._handle_keyboard_input()
                
                # 控制帧率
                time.sleep(1.0 / 60.0)  # 限制到60FPS
                
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._save_final_log()
            self._print_final_stats()
            
    def _process_frame(self, frame, frame_count):
        """处理单帧"""
        # YOLO人物检测
        results = self.yolo_model(frame, conf=self.config.confidence_threshold, iou=self.config.iou_threshold, verbose=False)
        
        person_detected = False
        best_roi = None
        best_confidence = 0.0
        
        # 处理检测结果
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # person类别
                        confidence = float(box.conf[0])
                        if confidence > best_confidence:
                            best_confidence = confidence
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            best_roi = np.array([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                            person_detected = True
                            
        if person_detected:
            self.stats['person_detected_frames'] += 1
            self.stats['confidences'].append(best_confidence)
            
        # 更新ROI
        filtered_roi = self.roi_filter.update(best_roi, best_confidence)
        
        # 光流检测
        camera_moving, camera_movement = self.optical_flow_detector.detect_camera_movement(frame)
        if camera_moving:
            self.stats['camera_movement_frames'] += 1
            
        # 姿态检测和动作识别
        action = None
        action_confidence = 0.0
        pose_confidence = 0.0
        
        if filtered_roi is not None and person_detected:
            # 提取ROI
            x, y, w, h = filtered_roi
            x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            
            if x2 > x and y2 > y:
                roi_frame = frame[y:y2, x:x2]
                
                # MediaPipe姿态检测
                rgb_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                pose_results = self.mp_pose.process(rgb_roi)
                
                if pose_results.pose_landmarks:
                    # 计算姿态置信度
                    pose_confidence = np.mean([lm.visibility for lm in pose_results.pose_landmarks.landmark])
                    
                    # 动作检测
                    action, action_confidence = self.pose_action_detector.detect_action(
                        pose_results.pose_landmarks, frame_count
                    )
                    
                    if action:
                        self.stats['actions_detected'] += 1
                        self.simulate_key_press(action)
                        
                        # 记录动作
                        combined_confidence = (best_confidence + pose_confidence + action_confidence) / 3
                        self.action_log['actions'].append({
                            'timestamp': time.time(),
                            'frame': frame_count,
                            'action': action,
                            'simulated_key': self.config.key_mapping.get(action, ''),
                            'roi': filtered_roi.tolist(),
                            'roi_confidence': float(best_confidence),
                            'pose_confidence': float(pose_confidence),
                            'action_confidence': float(action_confidence),
                            'combined_confidence': float(combined_confidence),
                            'camera_moving': camera_moving,
                            'camera_movement': camera_movement.tolist()
                        })
                        
    def _handle_keyboard_input(self):
        """处理键盘输入"""
        if keyboard.is_pressed('q'):
            self.running = False
        elif keyboard.is_pressed('p'):
            self.paused = not self.paused
            time.sleep(0.2)  # 防止重复触发
        elif keyboard.is_pressed('r'):
            self._reset_stats()
            time.sleep(0.2)
        elif keyboard.is_pressed('s'):
            self._save_current_log()
            time.sleep(0.2)
            
    def _reset_stats(self):
        """重置统计"""
        self.stats = {
            'total_frames': 0,
            'person_detected_frames': 0,
            'actions_detected': 0,
            'camera_movement_frames': 0,
            'processing_times': [],
            'confidences': []
        }
        self.action_log = {'stats': {}, 'actions': []}
        print("\n统计已重置")
        
    def _save_current_log(self):
        """保存当前日志"""
        self._update_log_stats()
        with open('pose_actions_log_final_current.json', 'w', encoding='utf-8') as f:
            json.dump(self.action_log, f, ensure_ascii=False, indent=2)
        print("\n当前日志已保存")
        
    def _save_final_log(self):
        """保存最终日志"""
        self._update_log_stats()
        with open('pose_actions_log_final.json', 'w', encoding='utf-8') as f:
            json.dump(self.action_log, f, ensure_ascii=False, indent=2)
        print("\n最终优化版动作日志已保存到 pose_actions_log_final.json")
        
    def _update_log_stats(self):
        """更新日志统计"""
        self.action_log['stats'] = {
            'total_frames': self.stats['total_frames'],
            'person_detected_frames': self.stats['person_detected_frames'],
            'actions_detected': self.stats['actions_detected'],
            'camera_movement_frames': self.stats['camera_movement_frames'],
            'average_confidence': np.mean(self.stats['confidences']) if self.stats['confidences'] else 0.0,
            'average_processing_time': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0.0,
            'max_processing_time': np.max(self.stats['processing_times']) if self.stats['processing_times'] else 0.0,
            'min_processing_time': np.min(self.stats['processing_times']) if self.stats['processing_times'] else 0.0
        }
        
    def _print_final_stats(self):
        """打印最终统计"""
        total_frames = self.stats['total_frames']
        person_frames = self.stats['person_detected_frames']
        actions = self.stats['actions_detected']
        camera_frames = self.stats['camera_movement_frames']
        
        person_rate = (person_frames / total_frames * 100) if total_frames > 0 else 0
        camera_rate = (camera_frames / total_frames * 100) if total_frames > 0 else 0
        avg_confidence = np.mean(self.stats['confidences']) if self.stats['confidences'] else 0
        avg_processing_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        print("\n=== 最终统计 (最终优化版) ===")
        print(f"总帧数: {total_frames}")
        print(f"检测到人物的帧数: {person_frames}")
        print(f"检测到动作的次数: {actions}")
        print(f"相机移动的帧数: {camera_frames}")
        print(f"人物检测率: {person_rate:.2f}%")
        print(f"相机移动率: {camera_rate:.2f}%")
        print(f"平均置信度: {avg_confidence:.3f}")
        print(f"平均处理时间: {avg_processing_time*1000:.2f}ms")
        print(f"平均FPS: {avg_fps:.1f}")

def main():
    """主函数"""
    video_path = "test.mp4"
    
    try:
        detector = YOLOMediaPipePoseDetector("pose_config_final.json")
        detector.process_video(video_path)
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()