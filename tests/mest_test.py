import time

import cv2
import numpy as np

from my_insightface.insightface.utils.my_tools import detect_cameras


def get_FOURCC(*resolution):
    cap = cv2.VideoCapture(0)

    #  设置帧数
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    get_fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
    print(f"after setting, The video  codec  is {codec}")
    frames = 0
    start_time = time.time()
    try:
        while True:
            if_true, frame = cap.read()
            if if_true:
                cv2.imshow('test', frame)
                frames += 1
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(e)
    finally:
        # 释放VideoCapture对象
        end_time = time.time()
        print("resolution is ", resolution)
        print(f"measured FPS of the video is {frames / (end_time - start_time)}")
        print(f"get FPS of the video is {get_fps}")
        cap.release()

def main():
    detect_cameras()
if __name__ == '__main__':
    main()
