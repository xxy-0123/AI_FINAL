from tkinter.constants import TRUE
import torch
import numpy as np
import cv2
import time
import win32api
import win32con
import pandas as pd
from utils.general import cv2
import cupy as cp
import ctypes
import sys
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import gameSelection
from ultralytics import YOLO
from tkinter import Tk, Toplevel
import threading
import keyboard
import time

# 获取屏幕信息
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

def smooth_mouse_move(dx, dy, steps=5):
    """
    Function to move the mouse smoothly 
    You can also use Kalman filter.
    """
    delta_x = dx / steps
    delta_y = dy / steps

    for _ in range(steps):
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(delta_x), int(delta_y), 0, 0)


def release_all_keys():
    for key in range(256):  # 释放所有按键
        if win32api.GetAsyncKeyState(key):
            win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)

templates = [ cv2.imread('scene/拐角.png', cv2.IMREAD_GRAYSCALE), cv2.imread('scene/卡车.png', cv2.IMREAD_GRAYSCALE), cv2.imread('scene/汽车.png', cv2.IMREAD_GRAYSCALE), cv2.imread('scene/中门.png', cv2.IMREAD_GRAYSCALE), cv2.imread('scene/转身点1.png', cv2.IMREAD_GRAYSCALE), cv2.imread('scene/转身点2.png', cv2.IMREAD_GRAYSCALE)]
scene=["拐角",'卡车', '汽车', '中门', '转身点1', '转身点2']

def sift_flann_matching(target_img, min_match_count=50):
    sift = cv2.SIFT_create()
    flann_index_kdtree = 0
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    target_kp, target_des = sift.detectAndCompute(target_img, None)
    best_match = None
    best_match_index = None
    i=-1

    for template in templates:
        i+=1
        template_kp, template_des = sift.detectAndCompute(template, None)
        if template_des is None or target_des is None:
            continue
        matches = flann.knnMatch(template_des, target_des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > min_match_count:
            if best_match is None or len(good) > best_match:
                best_match = len(good)
                best_match_index=i


    return best_match_index, best_match



def press_w_key(m):
    # print("Thread started, pressing 'w' key for 5 seconds.")
    keyboard.press('w')
    time.sleep(m)
    keyboard.release('w')

# Function to start the thread
def start_pressing_w_key(m):
    print(f"前进{m}s")
    press_thread = threading.Thread(target=press_w_key, args=(m,))
    press_thread.start()



def main():
    # 配置参数
    movement_factor = .55
    quit_key = "Q"
    confidence = 0.65
    headshot_mode = True
    cpsDisplay = True
    visuals = True
    character='all'
    screenShotHeight = 640
    screenShotWidth = 640
    match_scene=False
    vision_type='draw_transparent'
    trained_model_type='n_160_64_best.pt'

    # 获取参数的文本表示
    def get_params_text():
        params_text = [
            f'movement_factor: {movement_factor}',
            f'quit_key: {quit_key}',
            f'confidence: {confidence}',
            f'headshot_mode: {headshot_mode}',
            f'cpsDisplay: {cpsDisplay}',
            f'visuals: {visuals}',
            f'f1 character: {character}',
            f'f2 vision_type:{vision_type}',
            f'f3 for match_scene:{match_scene}',
            f"f4: {trained_model_type}"

        ]
        return "\n".join(params_text)

    current_state_index=0
    current_visual_index=0
    current_model_index=0

    # 运行gameSelection.py中的游戏选择菜单
    region, camera, cWidth, cHeight = gameSelection.gameSelection()

    # 用于强制垃圾回收
    count = 0
    sTime = time.time()

    trained_model=['n_160_64_best.pt','n_100_64_best.pt','s_100_64_best.pt','s_160_64_best.pt']
    model = YOLO('trained_model\\'+trained_model[current_model_index], verbose=False) 
    # model = YOLO("yolov9-e1.engine", task='detect', verbose=False) 
    # 用于绘制边界框的颜色
    COLORS = np.random.uniform(0, 255, size=(1500, 3))

    root = tk.Tk()
    root.geometry("400x300+0+0")
    root.attributes("-topmost", 1)  # 窗口置顶
    root.overrideredirect(True) 
    root.attributes("-transparentcolor", "white")  # 设置黑色为透明色

    # 创建Tkinter窗口1 - 显示检测结果
    root1 = Toplevel()
    root1.geometry(f"{screenShotHeight}x{screenShotWidth}+{region[0]}+{region[1]}")  # 窗口大小与位置
    root1.attributes("-topmost", 1)  # 窗口置顶
    root1.overrideredirect(True)  # 无边框窗口
    root1.attributes("-transparentcolor", "black")  # 设置黑色为透明色

    # 创建标签显示参数
    params_label = tk.Label(root, bg="black")  # 标签背景设置为黑色以实现透明效果
    params_label.pack(fill=tk.BOTH, expand=True)
    def update_setting():
        # print(1)
        text = get_params_text()
        image = Image.new('RGBA', (300, 300), (0, 0, 0, 0))  # 创建一个透明背景的图像
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("times.ttf", 25)
        draw.text((10, 10), text, font=font, fill=(255, 0, 0, 255))  # 白色文字

        image_tk = ImageTk.PhotoImage(image)
        params_label.config(image=image_tk)
        params_label.image = image_tk  # 保持引用

        root.after(1000, update_setting)

    params_label1 = tk.Label(root1, bg="black")  # 标签背景设置为黑色以实现透明效果
    params_label1.pack(fill=tk.BOTH, expand=True)

    targets = []

    def update_params_label(targets):
        # 获取当前屏幕截图
        COLORS = [(255, 0, 0)]
        image = Image.new('RGBA', (screenShotHeight, screenShotHeight), (0, 0, 0, 0))  # 创建一个透明背景的图像
        draw = ImageDraw.Draw(image)
        if len(targets) > 0:
            for i in range(len(targets)):
                conf = targets["confidence"].iloc[i]
                if conf >=confidence:
                    halfW = round(targets["width"].iloc[i] / 2)
                    halfH = round(targets["height"].iloc[i] / 2)
                    midX = targets['current_mid_x'].iloc[i]
                    midY = targets['current_mid_y'].iloc[i]
                    startX, startY, endX, endY = int(midX - halfW), int(midY - halfH), int(midX + halfW), int(midY + halfH)
                    cls_id = int(targets['class'].iloc[i])
                    label = "{}: {:.2f}%".format(model.names[cls_id], targets["confidence"].iloc[i] * 100)

                    # 绘制边界框
                    draw.rectangle([startX, startY, endX, endY], outline=COLORS[0], width=2)
                    # 确定标签的位置
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    # 加载字体
                    try:
                        font = ImageFont.truetype("times.ttf", 15)
                    except IOError:
                        font = ImageFont.load_default()
                    # 绘制标签
                    draw.text((startX, y), label, fill=COLORS[0], font=font)

        # print(1)
        image_tk = ImageTk.PhotoImage(image)
        params_label1.config(image=image_tk)
        params_label1.image = image_tk  # 保持引用

    update_setting()
    

    # 主循环，按退出键退出
    last_mid_coord = None
    last_reload_time_r = time.time()  # 初始化最后换弹时间
    last_reload_time_match = time.time()  

    with torch.no_grad():
        while win32api.GetAsyncKeyState(ord(quit_key)) == 0:
            npImg = cp.array([camera.get_latest_frame()])
            if npImg.shape[3] == 4:
                # 如果图像有alpha通道，移除它
                npImg = npImg[:, :, :, :3]

            im = npImg / 255
            im = im.astype(cp.half)
            im = cp.moveaxis(im, 3, 1)
            im = torch.from_numpy(cp.asnumpy(im)).to('cuda')

            results = model(im, verbose=False)

            targets = []

            for result in results:
                for box in result.boxes:
                    xywh = box.xywh.cpu().numpy()[0]
                    conf = box.conf.cpu().numpy()[0]
                    cls = box.cls.cpu().numpy()[0]
                    targets.append(list(xywh) + [conf, cls])

            targets = pd.DataFrame(
                targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence", "class"])
            
            center_screen = [cWidth, cHeight]


            states = ['t','ct','all']
            visualtype=['draw_transparent','show_results', 'null']

            if win32api.GetKeyState(0x70):  # 检查最高位是否为 1
                state = win32api.GetKeyState(0x70)
                if state < 0:  # 检查最高位是否为 1
                    win32api.keybd_event(0x70, 0, win32con.KEYEVENTF_KEYUP, 0)  # 松开
                    current_state_index = (current_state_index + 1) % 3
                    character = states[current_state_index]
                    print(f"Current character: {character}")

            if win32api.GetKeyState(0x71):  # 检查最高位是否为 1
                state = win32api.GetKeyState(0x71)
                if state < 0:  # 检查最高位是否为 1
                    win32api.keybd_event(0x71, 0, win32con.KEYEVENTF_KEYUP, 0)  # 松开
                    current_visual_index = (current_visual_index + 1) % 3
                    vision_type = visualtype[current_visual_index]
                    print(f"Current vision_type: {vision_type}")

            if win32api.GetKeyState(0x72):  # 检查最高位是否为 1
                state = win32api.GetKeyState(0x72)
                if state < 0:  # 检查最高位是否为 1
                    win32api.keybd_event(0x72, 0, win32con.KEYEVENTF_KEYUP, 0)  # 松开
                    match_scene = not match_scene
                    print(f"Current match_scene: {match_scene}")

            if win32api.GetKeyState(0x73):  # 检查最高位是否为 1
                state = win32api.GetKeyState(0x73)
                if state < 0:  # 检查最高位是否为 1
                    current_model_index = (current_model_index + 1) % 4
                    trained_model_type = trained_model[current_model_index]
                    print(f"Current model_type: {trained_model_type}")

            if character=='t':
            # if mode:
                targets = targets[targets['class'].isin([0, 2])]  # 0是ct，2是cthead
            elif character=='ct':
                targets = targets[targets['class'].isin([1, 3])]  # 1是t，3是thead
                

            if len(targets) == 0 and match_scene:
                # x, y, w, h = region
                # 调用 sift_flann_matching 进行图像匹配
                if time.time() - last_reload_time_match > 9:
                    target_img = cp.asnumpy(npImg[0])
                    last_reload_time_match=time.time()  
                    # cv2.imshow('Live Feed', target_img)
                    best_match_index, best_match = sift_flann_matching(cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY), 50)
                    if best_match is not None:
                        print("SIFT/FLANN 匹配成功")
                        print(scene[best_match_index])
                        if best_match_index in [0,1,5,3]:
                            if best_match_index ==1:
                                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 1500, 0, 0, 0)
                            elif best_match_index in [0,3]:
                                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -1500, 0, 0, 0)
                            else:
                                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -3000, 0, 0, 0)
                            start_pressing_w_key(7.5)

                        else:
                            if best_match_index ==4:
                                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -3000, 0, 0, 0)
                            else :
                                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 1500, 0, 0, 0)
                            start_pressing_w_key(4.5)

                    else:
                        print("SIFT/FLANN 匹配失败")

            if visuals and vision_type=='draw_transparent':
                update_params_label(targets)
            elif visuals and vision_type=='show_results':
                update_params_label([])
                for result in results:
                    img = result.plot()
                # img = cv2.resize(img, (screenShotHeight, screenShotWidth))
                cv2.imshow('YOLO', img)
                if cv2.waitKey(1) & 0xFF == ord(quit_key):
                    break
            else:
                cv2.destroyAllWindows()


            # 如果中心边界框内有人
            if len(targets) > 0:
                # 计算距离中心的距离
                targets["dist_from_center"] = np.sqrt((targets.current_mid_x - center_screen[0])**2 + (targets.current_mid_y - center_screen[1])**2)

                # 按距离中心的距离排序
                targets = targets.sort_values("dist_from_center")

                if last_mid_coord:
                    targets['last_mid_x'] = last_mid_coord[0]
                    targets['last_mid_y'] = last_mid_coord[1]
                    # 计算当前人的中间坐标和最后一个人的中间坐标之间的距离
                    targets['dist'] = np.linalg.norm(
                        targets.iloc[:, [0, 1]].values - targets.iloc[:, [4, 5]], axis=1)
                    targets.sort_values(by="dist", ascending=False)

                # 取数据框中第一个人（按欧几里得距离排序）
                xMid = targets.iloc[0].current_mid_x
                yMid = targets.iloc[0].current_mid_y

                box_height = targets.iloc[0].height
                                # 依据目标类型设置headshot_offset
                cls_id = int(targets.iloc[0]["class"])
                if headshot_mode:
                    if cls_id in [2, 3]:  # cthead or thead
                        headshot_offset = box_height * 0.38
                    else:
                        headshot_offset = box_height * 0.2
                else:
                    headshot_offset = box_height * 0.2  # 如果不在headshot模式，直接瞄准身体


                
                if win32api.GetKeyState(0x14):
                    #最近目标
                    dist_to_center = np.sqrt((xMid - cWidth)**2 + ((yMid - headshot_offset) - cHeight)**2)
                    conf = targets.iloc[0]["confidence"]
                    shoot_distance=10
                    if model.names[cls_id] in ['HT','HCT']:
                        shoot_distance=min(5,targets.iloc[0]["height"])

                    if dist_to_center < shoot_distance and conf >=confidence: 
                        print(f"Distance to center: {dist_to_center}")
                        cls_id = targets.iloc[0]["class"]
                        conf = targets.iloc[0]["confidence"]
                        label = model.names[cls_id]
                        print(f"Shooting at target: {label}")
                        # win32api.keybd_event(win32con.VK_SHIFT, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
                        keyboard.press('shift')
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        keyboard.release('shift')
                        # win32api.keybd_event(win32con.VK_SHIFT, 0, win32con.KEYEVENTF_KEYUP, 0)



                # 移动鼠标
                mouseMove = [xMid - cWidth, (yMid - headshot_offset) - cHeight]

                if win32api.GetKeyState(0x14):
                    conf = targets.iloc[0]["confidence"]
                    if abs(xMid - cWidth)+abs((yMid - headshot_offset) - cHeight)<20:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouseMove[0]), int(mouseMove[1]), 0, 0)
                    elif conf >=confidence: 
                        smooth_mouse_move(int(mouseMove[0] * movement_factor), int(mouseMove[1] * movement_factor))
                        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                last_mid_coord = [xMid, yMid]
                last_reload_time_r = time.time() 

            else:
                last_mid_coord = None

                if win32api.GetKeyState(0x14):
                    if time.time() - last_reload_time_r > 5:
                        win32api.keybd_event(0x52, 0, 0, 0)  # 按下R键
                        win32api.keybd_event(0x52, 0, win32con.KEYEVENTF_KEYUP, 0)  # 松开R键
                        last_reload_time_r = time.time()  # 重置最后换弹时间


            # 强制垃圾清理每秒一次
            count += 1
            if (time.time() - sTime) > 1:
                if cpsDisplay:
                    # logger.info("CPS: {}".format(count))
                    print("CPS: {}".format(count))
                count = 0
                sTime = time.time()

            root.update()  # 确保Tkinter窗口更新
            if visuals and vision_type=='draw_transparent':
                root1.update()  # 确保Tkinter窗口更新


               

    # 退出时清理
    cv2.destroyAllWindows()
    camera.release()
    release_all_keys()
    root.destroy()

    


if __name__ == "__main__":
    main()
    sys.exit()

