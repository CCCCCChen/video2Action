#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
游戏动作识别系统演示脚本

这个脚本展示了如何使用增强版动作识别系统的基本功能
"""

import json
import time
from datetime import datetime

def analyze_actions_log(log_file="actions_log.json"):
    """分析动作日志文件"""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            actions = json.load(f)
        
        if not actions:
            print("动作日志为空")
            return
        
        print(f"\n=== 动作日志分析 ===")
        print(f"总动作数量: {len(actions)}")
        
        # 统计各种动作的频率
        action_counts = {}
        key_counts = {}
        
        for action in actions:
            action_type = action["action"]
            key = action["key"]
            
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
            key_counts[key] = key_counts.get(key, 0) + 1
        
        print("\n动作频率统计:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(actions)) * 100
            print(f"  {action}: {count} 次 ({percentage:.1f}%)")
        
        print("\n按键频率统计:")
        for key, count in sorted(key_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(actions)) * 100
            print(f"  {key}: {count} 次 ({percentage:.1f}%)")
        
        # 时间分析
        if len(actions) > 1:
            start_time = actions[0]["timestamp"]
            end_time = actions[-1]["timestamp"]
            duration = end_time - start_time
            
            print(f"\n时间分析:")
            print(f"  总时长: {duration:.2f} 秒")
            print(f"  平均动作间隔: {duration/len(actions):.3f} 秒")
            print(f"  动作频率: {len(actions)/duration:.2f} 动作/秒")
        
    except FileNotFoundError:
        print(f"未找到日志文件: {log_file}")
        print("请先运行主程序生成动作日志")
    except Exception as e:
        print(f"分析日志时出错: {e}")

def show_config_info():
    """显示配置信息"""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        print("\n=== 当前配置信息 ===")
        
        print("\n按键映射:")
        for action, key in config["key_mapping"].items():
            print(f"  {action} -> {key}")
        
        print("\n检测设置:")
        settings = config["detection_settings"]
        print(f"  置信度阈值: {settings['confidence_threshold']}")
        print(f"  中央区域: {settings['center_region']}")
        print(f"  移动阈值: {settings['movement_thresholds']}")
        print(f"  闲置检测: {settings['idle_detection']}")
        
        print("\n视频设置:")
        video = config["video_settings"]
        print(f"  视频路径: {video['video_path']}")
        print(f"  模型路径: {video['model_path']}")
        
    except FileNotFoundError:
        print("未找到配置文件 config.json")
    except Exception as e:
        print(f"读取配置时出错: {e}")

def create_custom_config():
    """创建自定义配置示例"""
    custom_config = {
        "key_mapping": {
            "Left Turn": "left",
            "Right Turn": "right",
            "Forward": "up",
            "Backward": "down",
            "Jump": "space",
            "Glide": "shift",
            "Run Left": "left",
            "Run Right": "right",
            "Run Forward": "up",
            "Run Backward": "down"
        },
        "detection_settings": {
            "confidence_threshold": 0.6,
            "center_region": {
                "x_min_ratio": 0.25,
                "x_max_ratio": 0.75,
                "y_min_ratio": 0.1,
                "y_max_ratio": 0.9
            },
            "movement_thresholds": {
                "horizontal_movement": 20,
                "vertical_movement": 12,
                "size_change": 10,
                "jump_threshold": 30,
                "speed_threshold": 10
            },
            "idle_detection": {
                "idle_frames": 90,
                "no_person_frames": 45
            }
        },
        "video_settings": {
            "video_path": "test.mp4",
            "model_path": "yolov8s.pt"
        }
    }
    
    try:
        with open("config_custom.json", "w", encoding="utf-8") as f:
            json.dump(custom_config, f, ensure_ascii=False, indent=2)
        print("\n自定义配置文件已创建: config_custom.json")
        print("可以将其重命名为 config.json 来使用")
    except Exception as e:
        print(f"创建配置文件时出错: {e}")

def main():
    """主函数"""
    print("=== 游戏动作识别系统演示 ===")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    while True:
        print("\n请选择操作:")
        print("1. 分析动作日志")
        print("2. 查看当前配置")
        print("3. 创建自定义配置")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            analyze_actions_log()
        elif choice == "2":
            show_config_info()
        elif choice == "3":
            create_custom_config()
        elif choice == "4":
            print("\n再见！")
            break
        else:
            print("\n无效选择，请重试")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()