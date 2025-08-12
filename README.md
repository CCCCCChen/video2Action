# 增强版游戏动作识别系统

基于YOLOv8的实时游戏人物动作识别与按键映射系统。

## 功能特性

### 🎯 核心功能
- **人物检测**: 基于YOLOv8模型的高精度人物检测
- **动作识别**: 支持多种游戏动作识别
  - 移动: 左转、右转、前进、后退
  - 跳跃: 跳跃、滑翔/下降
  - 跑步: 快速移动检测
  - 闲置: 人物静止状态检测
- **游戏状态检测**: 自动识别游戏页面和菜单页面
- **按键映射**: 将识别的动作映射到对应按键
- **动作日志**: 记录所有检测到的动作和按键操作

### 🔧 技术特点
- 可配置的检测参数
- 实时状态显示
- 防抖动按键处理
- 异常处理和错误恢复

## 安装要求

```bash
pip install ultralytics opencv-python
# 如需实际按键模拟功能，还需要安装:
# pip install pynput
```

## 文件结构

```
video2Action/
├── yolov8_detect_upgrade.py  # 主程序文件
├── config.json               # 配置文件
├── yolov8n.pt               # YOLOv8模型文件
├── clip.mp4                 # 测试视频文件
├── actions_log.json         # 动作日志文件（运行后生成）
└── README.md                # 说明文档
```

## 使用方法

### 1. 基本运行
```bash
python yolov8_detect_upgrade.py
```

### 2. 交互控制
运行程序后，可以使用以下按键控制：
- **q**: 退出程序
- **s**: 保存动作日志到文件
- **c**: 清空当前动作日志
- **r**: 重置检测状态

### 3. 配置自定义
编辑 `config.json` 文件来自定义：

#### 按键映射
```json
"key_mapping": {
  "Left Turn": "a",
  "Right Turn": "d",
  "Forward": "w",
  "Backward": "s",
  "Jump": "space"
}
```

#### 检测参数
```json
"detection_settings": {
  "confidence_threshold": 0.5,
  "movement_thresholds": {
    "horizontal_movement": 15,
    "vertical_movement": 10,
    "jump_threshold": 25
  }
}
```

#### 视频设置
```json
"video_settings": {
  "video_path": "your_video.mp4",
  "model_path": "yolov8n.pt"
}
```

## 界面说明

程序运行时会显示：
- **绿色框**: 检测到的人物
- **黄色框**: 中央检测区域
- **状态信息**: 游戏状态、当前动作、统计数据
- **颜色含义**:
  - 绿色文字: 正常游戏状态
  - 红色文字: 菜单状态或闲置动作
  - 蓝色文字: 活跃动作

## 动作识别类型

| 动作类型 | 检测条件 | 对应按键 |
|---------|---------|----------|
| Left Turn | 水平向左移动 | a |
| Right Turn | 水平向右移动 | d |
| Forward | 人物框变大 | w |
| Backward | 人物框变小 | s |
| Jump | 快速向上移动 | space |
| Glide | 快速向下移动 | shift |
| Run Left/Right | 快速水平移动 | a/d |
| Idle | 无明显移动 | 无 |

## 游戏状态

- **Playing**: 检测到中央区域有人物，正常游戏状态
- **Menu**: 连续多帧未检测到人物，判断为菜单或加载页面

## 日志格式

动作日志以JSON格式保存：
```json
[
  {
    "timestamp": 1234567890.123,
    "action": "Jump",
    "key": "space",
    "state": "playing"
  }
]
```

## 注意事项

1. **模型文件**: 确保 `yolov8n.pt` 模型文件存在
2. **视频格式**: 支持常见视频格式（mp4, avi等）
3. **性能**: 建议使用GPU加速以获得更好的实时性能
4. **按键模拟**: 当前版本仅打印按键信息，如需实际按键模拟请取消注释pynput相关代码

## 故障排除

### 常见问题
1. **无法检测到人物**: 调整 `confidence_threshold` 参数
2. **动作识别不准确**: 调整 `movement_thresholds` 中的各项阈值
3. **频繁误判**: 增加 `idle_frames` 和 `no_person_frames` 参数

### 性能优化
- 使用更小的模型（yolov8n.pt）以提高速度
- 使用更大的模型（yolov8s.pt, yolov8m.pt）以提高精度
- 调整视频分辨率

## 版本信息

- **版本**: 2.0
- **作者**: AI Assistant
- **更新日期**: 2024

## 许可证

本项目仅供学习和研究使用。