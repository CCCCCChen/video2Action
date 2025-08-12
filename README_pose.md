# YOLO + MediaPipe Pose 游戏动作识别系统

## 概述

这是一个基于YOLO人物检测、MediaPipe Pose骨骼关键点提取和光流背景检测的高精度游戏动作识别系统，专门针对《原神》等第三人称游戏进行优化。

## 系统架构

### 核心流程

1. **YOLO人物检测** → 在整幅视频中定位人物，减少不必要的全画面关键点检测
2. **ROI裁剪与平滑** → 使用卡尔曼滤波对检测框进行平滑处理，防止抖动
3. **光流背景检测** → 检测相机移动，过滤背景抖动对动作识别的干扰
4. **MediaPipe Pose** → 在裁剪的人物区域提取33个骨骼关键点
5. **动作分类** → 基于骨骼关键点位移进行精确动作判定
6. **游戏特化优化** → 应用原神特定的优化规则和动作平滑
7. **按键模拟** → 根据识别结果模拟相应的游戏按键

### 技术特点

- **高精度检测**: 结合YOLO和MediaPipe的优势，实现更准确的人物动作识别
- **背景抗干扰**: 光流检测相机移动，有效过滤背景变化对动作识别的影响
- **实时性能**: 优化的ROI检测减少计算量，提高处理速度
- **游戏特化**: 针对原神游戏特点进行专门优化
- **可配置性**: 通过JSON配置文件灵活调整各种参数

## 功能特性

### 动作识别类型

- **跳跃 (Jump)**: 基于脚部关键点Y坐标快速上升检测
- **着陆 (Landing)**: 基于脚部关键点Y坐标快速下降检测
- **左右移动 (Move Left/Right)**: 基于脚部中心水平位移检测
- **前后移动 (Move Forward/Backward)**: 基于身体中心垂直位移检测
- **待机 (Idle)**: 基于所有关键点位移方差的低阈值检测

### 原神特化优化

- **固定中心检测**: 原神角色固定在屏幕中心，优化检测范围
- **相机移动过滤**: 使用光流检测相机转动，避免误判为人物移动
- **动作平滑**: 要求动作连续满足条件才输出，避免瞬间误检
- **呼吸动作过滤**: 忽略站立时的微小抖动

### 技术组件

- **卡尔曼滤波器**: 平滑ROI检测框，减少抖动
- **光流检测器**: 检测背景移动和相机转动
- **姿态动作检测器**: 基于MediaPipe Pose的动作分类
- **按键模拟器**: 根据动作自动模拟游戏按键

## 安装要求

### 系统要求

- Python 3.8+
- Windows/Linux/macOS
- 支持CUDA的GPU（推荐，用于YOLO加速）

### 依赖安装

```bash
pip install -r requirements_pose.txt
```

### 主要依赖包

- `ultralytics`: YOLOv8模型
- `mediapipe`: 姿态估计
- `opencv-python`: 计算机视觉
- `filterpy`: 卡尔曼滤波
- `numpy`, `scipy`: 数值计算
- `pynput`: 按键模拟（可选）

## 文件结构

```
video2Action/
├── yolo_mediapipe_pose_detection.py  # 主程序
├── pose_config.json                   # 配置文件
├── requirements_pose.txt              # 依赖列表
├── README_pose.md                     # 说明文档
├── yolov8n.pt                        # YOLO模型文件
├── clip.mp4                          # 测试视频
└── pose_actions_log.json             # 动作日志（运行后生成）
```

## 使用方法

### 基本运行

```bash
python yolo_mediapipe_pose_detection.py
```

### 交互控制

运行时可使用以下按键控制：

- **q**: 退出程序
- **s**: 保存动作日志
- **r**: 重置统计信息
- **c**: 清空动作历史

### 配置自定义

编辑 `pose_config.json` 文件来自定义系统参数：

```json
{
  "model_settings": {
    "yolo_model_path": "yolov8n.pt",
    "video_path": "your_video.mp4",
    "confidence_threshold": 0.5
  },
  "action_detection": {
    "jump_threshold": 15,
    "move_threshold": 8,
    "idle_threshold": 3
  },
  "key_mapping": {
    "Jump": "space",
    "Move Right": "d",
    "Move Left": "a"
  }
}
```

## 界面说明

### 实时显示信息

- **绿色框**: 检测到的人物ROI区域
- **蓝色点**: MediaPipe检测到的骨骼关键点
- **动作标签**: 当前识别的动作（红色=待机，蓝色=其他动作）
- **相机状态**: 显示相机是否在移动
- **统计信息**: 帧数、动作数量、人物检测率等

### 日志格式

动作日志保存为JSON格式，包含以下信息：

```json
{
  "stats": {
    "total_frames": 1000,
    "person_detected_frames": 950,
    "actions_detected": 45,
    "camera_movement_frames": 120
  },
  "actions": [
    {
      "timestamp": 1699123456.789,
      "frame": 100,
      "action": "Jump",
      "simulated_key": "space",
      "roi": [100, 50, 200, 300],
      "camera_moving": false,
      "camera_movement": [0.1, -0.2]
    }
  ]
}
```

## 参数调优

### 动作检测阈值

- `jump_threshold`: 跳跃检测阈值（像素），默认15
- `move_threshold`: 移动检测阈值（像素），默认8
- `idle_threshold`: 待机检测阈值（像素），默认3

### ROI设置

- `roi_expand_ratio`: ROI扩展比例，默认1.5
- `kalman_q`: 卡尔曼滤波过程噪声，默认0.001
- `kalman_r`: 卡尔曼滤波观测噪声，默认0.01

### 光流检测

- `camera_move_threshold`: 相机移动阈值，默认5.0
- `optical_flow_points`: 光流特征点数量，默认100

## 性能优化

### GPU加速

确保安装了CUDA版本的PyTorch以启用GPU加速：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 内存优化

- 调整 `pose_history_length` 减少内存使用
- 降低 `optical_flow_points` 数量
- 使用较小的YOLO模型（如yolov8n.pt）

## 故障排除

### 常见问题

1. **无法检测到人物**
   - 检查视频质量和光照条件
   - 降低 `confidence_threshold`
   - 确认人物在画面中心区域

2. **动作识别不准确**
   - 调整动作检测阈值
   - 增加 `action_smooth_frames` 提高稳定性
   - 检查相机移动过滤是否正常工作

3. **性能问题**
   - 启用GPU加速
   - 降低视频分辨率
   - 减少光流特征点数量

4. **按键模拟不工作**
   - 取消注释pynput相关代码
   - 确保有管理员权限（Windows）
   - 检查按键映射配置

### 调试模式

在代码中设置调试标志以获取更多信息：

```python
# 在主函数开始处添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展开发

### 添加新动作类型

1. 在 `_classify_action` 方法中添加新的检测逻辑
2. 在配置文件的 `key_mapping` 中添加对应按键
3. 调整相关阈值参数

### 适配其他游戏

1. 修改人物检测的中心区域设置
2. 调整动作检测阈值以适应游戏特点
3. 更新按键映射配置
4. 添加游戏特定的优化规则

## 版本信息

- **版本**: 3.0
- **作者**: AI Assistant
- **更新日期**: 2024年
- **兼容性**: Python 3.8+, OpenCV 4.8+, MediaPipe 0.10+

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8实现
- [MediaPipe](https://github.com/google/mediapipe) - 姿态估计
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [FilterPy](https://github.com/rlabbe/filterpy) - 卡尔曼滤波实现