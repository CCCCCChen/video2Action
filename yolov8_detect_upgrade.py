#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版游戏动作识别系统

功能特性:
1. 基于YOLOv8的人物检测
2. 多种动作识别（移动、跳跃、闲置等）
3. 游戏状态检测（游戏中/菜单页面）
4. 按键映射和模拟
5. 动作日志记录
6. 实时状态显示

使用说明:
- 按 'q' 退出程序
- 按 's' 保存动作日志
- 按 'c' 清空动作日志
- 按 'r' 重置检测状态

作者: AI Assistant
版本: 2.0
"""

import cv2
import time
import json
from ultralytics import YOLO
from collections import deque
# import pynput  # 如需实际按键模拟，取消注释
# from pynput import keyboard

# 加载配置文件
def load_config():
    """加载配置文件"""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("配置文件 config.json 未找到，使用默认配置")
        return {
            "key_mapping": {
                "Left Turn": "a", "Right Turn": "d", "Forward": "w", "Backward": "s",
                "Jump": "space", "Glide": "shift", "Run Left": "a", "Run Right": "d",
                "Run Forward": "w", "Run Backward": "s", "Move Up": "w", "Move Down": "s"
            },
            "detection_settings": {
                "confidence_threshold": 0.5,
                "center_region": {"x_min_ratio": 0.3, "x_max_ratio": 0.7, "y_min_ratio": 0.1, "y_max_ratio": 0.9},
                "movement_thresholds": {"horizontal_movement": 15, "vertical_movement": 10, "size_change": 8, "jump_threshold": 25, "speed_threshold": 8},
                "idle_detection": {"idle_frames": 60, "no_person_frames": 30}
            },
            "video_settings": {"video_path": "clip.mp4", "model_path": "yolov8n.pt"}
        }
    except Exception as e:
        print(f"配置文件加载失败: {e}，使用默认配置")
        return load_config()  # 递归调用返回默认配置

# 加载配置
config = load_config()
key_mapping = config["key_mapping"]
detection_settings = config["detection_settings"]
video_settings = config["video_settings"]

# 1. 加载 YOLOv8
model = YOLO(video_settings["model_path"])  # 从配置文件读取模型路径

video_path = video_settings["video_path"]
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"无法打开视频文件: {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video size: {frame_width}x{frame_height}, FPS: {fps}")
print(f"Model: {video_settings['model_path']}, Video: {video_path}")

# 中央区域范围（从配置文件读取）
center_region = detection_settings["center_region"]
center_x_min = frame_width * center_region["x_min_ratio"]
center_x_max = frame_width * center_region["x_max_ratio"]
center_y_min = frame_height * center_region["y_min_ratio"]
center_y_max = frame_height * center_region["y_max_ratio"]

# 用队列存最近的中心点位置
history = deque(maxlen=10)  # 最近10帧位置
idle_counter = 0  # 闲置计数器
last_action_time = time.time()
no_person_counter = 0  # 无人物计数器

# 游戏状态
class GameState:
    PLAYING = "playing"  # 游戏中
    MENU = "menu"       # 菜单页面
    LOADING = "loading" # 加载页面
    IDLE = "idle"       # 人物闲置

current_state = GameState.PLAYING
last_key_pressed = None
last_key_time = 0

# 动作日志
actions_log = []

def detect_game_state(main_person_detected, frame):
    """检测游戏状态"""
    global no_person_counter, current_state
    
    no_person_threshold = detection_settings["idle_detection"]["no_person_frames"]
    
    if not main_person_detected:
        no_person_counter += 1
        if no_person_counter > no_person_threshold:
            current_state = GameState.MENU
            return GameState.MENU
    else:
        no_person_counter = 0
        if current_state == GameState.MENU:
            current_state = GameState.PLAYING
    
    return current_state

def detect_action():
    """检测人物动作"""
    global idle_counter, last_action_time
    
    if len(history) < 3:
        return None
    
    # 从配置文件获取阈值
    thresholds = detection_settings["movement_thresholds"]
    idle_frames = detection_settings["idle_detection"]["idle_frames"]
    
    # 获取最近几帧的位置信息
    recent_positions = list(history)[-5:] if len(history) >= 5 else list(history)
    
    if len(recent_positions) < 2:
        return None
    
    # 计算位移和变化
    dx = recent_positions[-1][0] - recent_positions[0][0]
    dy = recent_positions[-1][1] - recent_positions[0][1]
    dh = recent_positions[-1][2] - recent_positions[0][2]
    
    # 计算移动速度
    frames_span = len(recent_positions) - 1
    speed_x = abs(dx) / frames_span if frames_span > 0 else 0
    speed_y = abs(dy) / frames_span if frames_span > 0 else 0
    
    # 检测闲置状态
    if speed_x < 2 and speed_y < 2 and abs(dh) < 3:
        idle_counter += 1
        if idle_counter > idle_frames:
            return "Idle"
    else:
        idle_counter = 0
        last_action_time = time.time()
    
    # 跳跃检测（优先级最高）
    if dy < -thresholds["jump_threshold"] and speed_y > 3:
        return "Jump"
    
    # 下降/滑翔检测
    if dy > thresholds["jump_threshold"] and speed_y > 3:
        return "Glide"
    
    # 快速移动检测（跑步）
    if speed_x > thresholds["speed_threshold"] or speed_y > thresholds["speed_threshold"]:
        if abs(dx) > abs(dy):
            return "Run Right" if dx > 0 else "Run Left"
        else:
            return "Run Forward" if dh > 0 else "Run Backward"
    
    # 左右转（水平移动）
    if abs(dx) > thresholds["horizontal_movement"] and abs(dy) < thresholds["vertical_movement"]:
        return "Right Turn" if dx > 0 else "Left Turn"
    
    # 前进/后退（基于框大小变化）
    if abs(dh) > thresholds["size_change"] and abs(dx) < thresholds["vertical_movement"]:
        return "Forward" if dh > 0 else "Backward"
    
    # 垂直移动
    if abs(dy) > thresholds["vertical_movement"] and abs(dx) < thresholds["size_change"]:
        return "Move Up" if dy < 0 else "Move Down"
    
    return None

def simulate_key_press(action):
    """模拟按键操作"""
    global last_key_pressed, last_key_time
    
    if action in key_mapping:
        key = key_mapping[action]
        current_time = time.time()
        
        # 避免重复按键（防抖）
        if last_key_pressed != key or (current_time - last_key_time) > 0.1:
            try:
                # 这里可以添加实际的按键模拟代码
                print(f"模拟按键: {key} (动作: {action})")
                last_key_pressed = key
                last_key_time = current_time
                
                # 记录动作日志
                log_entry = {
                    "timestamp": current_time,
                    "action": action,
                    "key": key,
                    "state": current_state
                }
                actions_log.append(log_entry)
                
            except Exception as e:
                print(f"按键模拟失败: {e}")

def save_actions_log():
    """保存动作日志"""
    try:
        with open("actions_log.json", "w", encoding="utf-8") as f:
            json.dump(actions_log, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存日志失败: {e}")

try:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        results = model(frame, verbose=False)
        
        main_person = None
        person_detected = False
        
        # 检测人物
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                     cls_id = int(box.cls[0])
                     conf = float(box.conf[0])
                     confidence_threshold = detection_settings["confidence_threshold"]
                     if cls_id == 0 and conf > confidence_threshold:  # 人物类别
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        w, h = x2 - x1, y2 - y1
                        
                        # 只保留中央区域的角色
                        if center_x_min < cx < center_x_max and center_y_min < cy < center_y_max:
                            main_person = (cx, cy, h)
                            person_detected = True
                            
                            # 绘制检测框
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Player {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 检测游戏状态
        game_state = detect_game_state(person_detected, frame)
        
        # 显示游戏状态
        state_color = (0, 255, 0) if game_state == GameState.PLAYING else (0, 0, 255)
        cv2.putText(frame, f"State: {game_state}", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        
        current_action = None
        
        # 只在游戏状态下进行动作检测
        if game_state == GameState.PLAYING and main_person:
            history.append(main_person)
            current_action = detect_action()
            
            if current_action:
                # 显示当前动作
                action_color = (0, 0, 255) if current_action == "Idle" else (255, 0, 0)
                cv2.putText(frame, f"Action: {current_action}", (50, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, action_color, 2)
                
                # 模拟按键（除了闲置状态）
                if current_action != "Idle":
                    simulate_key_press(current_action)
        
        elif game_state == GameState.MENU:
            # 清空历史记录
            history.clear()
            idle_counter = 0
            cv2.putText(frame, "No player detected - Menu/Loading", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 显示统计信息
        cv2.putText(frame, f"Frame: {frame_count}", (frame_width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"History: {len(history)}", (frame_width - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Actions logged: {len(actions_log)}", (frame_width - 200, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 绘制中央检测区域
        cv2.rectangle(frame, (int(center_x_min), int(center_y_min)), 
                     (int(center_x_max), int(center_y_max)), (255, 255, 0), 1)
        
        # 显示
        cv2.imshow("Enhanced Game Action Detection", frame)
        
        # 按键控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):  # 保存日志
            save_actions_log()
            print("动作日志已保存")
        elif key == ord("c"):  # 清空日志
            actions_log.clear()
            print("动作日志已清空")
        elif key == ord("r"):  # 重置状态
            history.clear()
            idle_counter = 0
            no_person_counter = 0
            current_state = GameState.PLAYING
            print("状态已重置")

except KeyboardInterrupt:
    print("\n程序被用户中断")
except Exception as e:
    print(f"程序运行出错: {e}")
finally:
    # 保存最终日志
    save_actions_log()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n程序结束，共处理 {frame_count} 帧")
    print(f"记录了 {len(actions_log)} 个动作")
    if actions_log:
        print(f"动作日志已保存到 actions_log.json")
