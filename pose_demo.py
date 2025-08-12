#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + MediaPipe Pose 动作识别系统演示和分析工具

功能:
1. 分析动作日志
2. 可视化统计数据
3. 性能测试
4. 配置验证

作者: AI Assistant
版本: 1.0
"""

import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

def load_pose_log(log_file="pose_actions_log.json"):
    """加载动作日志"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"日志文件不存在: {log_file}")
        return None
    except json.JSONDecodeError:
        print(f"日志文件格式错误: {log_file}")
        return None

def analyze_actions(log_data):
    """分析动作数据"""
    if not log_data or 'actions' not in log_data:
        print("没有可分析的动作数据")
        return
    
    actions = log_data['actions']
    stats = log_data.get('stats', {})
    
    print("=== 动作识别分析报告 ===")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 基本统计
    print("=== 基本统计 ===")
    print(f"总帧数: {stats.get('total_frames', 0)}")
    print(f"检测到人物的帧数: {stats.get('person_detected_frames', 0)}")
    print(f"识别到动作的次数: {stats.get('actions_detected', 0)}")
    print(f"相机移动的帧数: {stats.get('camera_movement_frames', 0)}")
    
    if stats.get('total_frames', 0) > 0:
        person_rate = stats.get('person_detected_frames', 0) / stats['total_frames'] * 100
        camera_rate = stats.get('camera_movement_frames', 0) / stats['total_frames'] * 100
        print(f"人物检测率: {person_rate:.2f}%")
        print(f"相机移动率: {camera_rate:.2f}%")
    print()
    
    if not actions:
        print("没有动作数据可分析")
        return
    
    # 动作类型统计
    print("=== 动作类型统计 ===")
    action_counts = Counter([action['action'] for action in actions])
    for action, count in action_counts.most_common():
        percentage = count / len(actions) * 100
        print(f"{action}: {count} 次 ({percentage:.1f}%)")
    print()
    
    # 按键统计
    print("=== 按键模拟统计 ===")
    key_counts = Counter([action.get('simulated_key') for action in actions if action.get('simulated_key')])
    for key, count in key_counts.most_common():
        print(f"{key}: {count} 次")
    print()
    
    # 时间分析
    print("=== 时间分析 ===")
    timestamps = [action['timestamp'] for action in actions]
    if len(timestamps) > 1:
        duration = timestamps[-1] - timestamps[0]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_interval = np.mean(intervals)
        action_rate = len(actions) / duration if duration > 0 else 0
        
        print(f"总时长: {duration:.2f} 秒")
        print(f"平均动作间隔: {avg_interval:.3f} 秒")
        print(f"动作频率: {action_rate:.2f} 动作/秒")
    print()
    
    # 相机移动分析
    print("=== 相机移动分析 ===")
    camera_moving_actions = [action for action in actions if action.get('camera_moving', False)]
    camera_static_actions = [action for action in actions if not action.get('camera_moving', False)]
    
    print(f"相机移动时的动作: {len(camera_moving_actions)} 次")
    print(f"相机静止时的动作: {len(camera_static_actions)} 次")
    
    if camera_moving_actions:
        moving_action_types = Counter([action['action'] for action in camera_moving_actions])
        print("相机移动时的动作类型:")
        for action, count in moving_action_types.most_common():
            print(f"  {action}: {count} 次")
    print()

def visualize_actions(log_data):
    """可视化动作数据"""
    if not log_data or 'actions' not in log_data:
        print("没有可可视化的数据")
        return
    
    actions = log_data['actions']
    if not actions:
        print("没有动作数据可可视化")
        return
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLO + MediaPipe Pose 动作识别分析', fontsize=16)
    
    # 1. 动作类型分布饼图
    action_counts = Counter([action['action'] for action in actions])
    ax1.pie(action_counts.values(), labels=action_counts.keys(), autopct='%1.1f%%')
    ax1.set_title('动作类型分布')
    
    # 2. 时间序列图
    timestamps = [action['timestamp'] for action in actions]
    action_names = [action['action'] for action in actions]
    
    # 将动作转换为数值
    unique_actions = list(set(action_names))
    action_to_num = {action: i for i, action in enumerate(unique_actions)}
    action_nums = [action_to_num[action] for action in action_names]
    
    # 转换时间戳为相对时间
    if timestamps:
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]
        
        ax2.scatter(relative_times, action_nums, alpha=0.6)
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('动作类型')
        ax2.set_title('动作时间序列')
        ax2.set_yticks(range(len(unique_actions)))
        ax2.set_yticklabels(unique_actions)
    
    # 3. 按键统计柱状图
    key_counts = Counter([action.get('simulated_key') for action in actions if action.get('simulated_key')])
    if key_counts:
        ax3.bar(key_counts.keys(), key_counts.values())
        ax3.set_xlabel('按键')
        ax3.set_ylabel('次数')
        ax3.set_title('按键模拟统计')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. 相机移动vs动作类型
    camera_data = defaultdict(lambda: {'moving': 0, 'static': 0})
    for action in actions:
        action_type = action['action']
        if action.get('camera_moving', False):
            camera_data[action_type]['moving'] += 1
        else:
            camera_data[action_type]['static'] += 1
    
    if camera_data:
        action_types = list(camera_data.keys())
        moving_counts = [camera_data[action]['moving'] for action in action_types]
        static_counts = [camera_data[action]['static'] for action in action_types]
        
        x = np.arange(len(action_types))
        width = 0.35
        
        ax4.bar(x - width/2, moving_counts, width, label='相机移动', alpha=0.8)
        ax4.bar(x + width/2, static_counts, width, label='相机静止', alpha=0.8)
        ax4.set_xlabel('动作类型')
        ax4.set_ylabel('次数')
        ax4.set_title('相机状态vs动作类型')
        ax4.set_xticks(x)
        ax4.set_xticklabels(action_types, rotation=45)
        ax4.legend()
    
    plt.tight_layout()
    plt.show()

def validate_config(config_file="pose_config.json"):
    """验证配置文件"""
    print("=== 配置文件验证 ===")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✓ 配置文件加载成功: {config_file}")
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False
    
    # 检查必需的配置项
    required_sections = [
        'model_settings',
        'roi_settings', 
        'optical_flow_settings',
        'action_detection',
        'key_mapping'
    ]
    
    for section in required_sections:
        if section in config:
            print(f"✓ 配置节存在: {section}")
        else:
            print(f"✗ 缺少配置节: {section}")
    
    # 检查文件路径
    model_path = config.get('model_settings', {}).get('yolo_model_path')
    video_path = config.get('model_settings', {}).get('video_path')
    
    if model_path and os.path.exists(model_path):
        print(f"✓ YOLO模型文件存在: {model_path}")
    else:
        print(f"✗ YOLO模型文件不存在: {model_path}")
    
    if video_path and os.path.exists(video_path):
        print(f"✓ 视频文件存在: {video_path}")
    else:
        print(f"✗ 视频文件不存在: {video_path}")
    
    print()
    return True

def performance_test():
    """性能测试"""
    print("=== 性能测试 ===")
    
    try:
        # 测试导入时间
        start_time = time.time()
        import cv2
        import mediapipe as mp
        from ultralytics import YOLO
        import_time = time.time() - start_time
        print(f"库导入时间: {import_time:.3f} 秒")
        
        # 测试YOLO模型加载
        start_time = time.time()
        model = YOLO("yolov8n.pt")
        model_load_time = time.time() - start_time
        print(f"YOLO模型加载时间: {model_load_time:.3f} 秒")
        
        # 测试MediaPipe初始化
        start_time = time.time()
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mediapipe_init_time = time.time() - start_time
        print(f"MediaPipe初始化时间: {mediapipe_init_time:.3f} 秒")
        
        # 测试OpenCV
        start_time = time.time()
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        opencv_time = time.time() - start_time
        print(f"OpenCV处理时间: {opencv_time:.6f} 秒")
        
        pose.close()
        
    except Exception as e:
        print(f"性能测试失败: {e}")
    
    print()

def create_test_config():
    """创建测试配置"""
    test_config = {
        "model_settings": {
            "yolo_model_path": "yolov8n.pt",
            "video_path": "test_video.mp4",
            "confidence_threshold": 0.3
        },
        "roi_settings": {
            "center_region_ratio": 0.8,
            "roi_expand_ratio": 2.0,
            "kalman_q": 0.01,
            "kalman_r": 0.1
        },
        "optical_flow_settings": {
            "optical_flow_points": 50,
            "camera_move_threshold": 3.0
        },
        "action_detection": {
            "jump_threshold": 10,
            "move_threshold": 5,
            "idle_threshold": 2,
            "action_smooth_frames": 2
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
    
    filename = "pose_config_test.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(test_config, f, ensure_ascii=False, indent=2)
    
    print(f"测试配置文件已创建: {filename}")
    return filename

def main():
    """主函数"""
    print("=== YOLO + MediaPipe Pose 系统演示工具 ===")
    print()
    
    while True:
        print("请选择功能:")
        print("1. 分析动作日志")
        print("2. 可视化动作数据")
        print("3. 验证配置文件")
        print("4. 性能测试")
        print("5. 创建测试配置")
        print("6. 退出")
        
        choice = input("\n请输入选择 (1-6): ").strip()
        
        if choice == '1':
            log_file = input("请输入日志文件名 (默认: pose_actions_log.json): ").strip()
            if not log_file:
                log_file = "pose_actions_log.json"
            
            log_data = load_pose_log(log_file)
            if log_data:
                analyze_actions(log_data)
        
        elif choice == '2':
            log_file = input("请输入日志文件名 (默认: pose_actions_log.json): ").strip()
            if not log_file:
                log_file = "pose_actions_log.json"
            
            log_data = load_pose_log(log_file)
            if log_data:
                visualize_actions(log_data)
        
        elif choice == '3':
            config_file = input("请输入配置文件名 (默认: pose_config.json): ").strip()
            if not config_file:
                config_file = "pose_config.json"
            
            validate_config(config_file)
        
        elif choice == '4':
            performance_test()
        
        elif choice == '5':
            create_test_config()
        
        elif choice == '6':
            print("再见！")
            break
        
        else:
            print("无效选择，请重试")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()