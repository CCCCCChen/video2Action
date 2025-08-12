import cv2
import pyautogui
import glob
import time

# 读取技能图标模板（灰度）
template = cv2.imread('skill_icon_template.png', 0)
w, h = template.shape[::-1]

# 遍历帧文件
for frame_path in sorted(glob.glob('frames/*.jpg')):
    img_gray = cv2.imread(frame_path, 0)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    if res.max() > 0.8:  # 匹配阈值可调
        print(f"检测到技能图标: {frame_path}")
        # 模拟按键（这里用记事本测试，不要直接作用到游戏）
        pyautogui.press('1')
    time.sleep(0.125)  # 对应 8 FPS（1/8 秒间隔）
