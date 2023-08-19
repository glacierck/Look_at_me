import time

import cv2
import numpy as np
from line_profiler_pycharm import profile

from my_insightface.insightface.utils.my_tools import detect_cameras

@profile
def get_FOURCC(*resolution):
    cap = cv2.VideoCapture('http://localhost:8080/video')


    #  设置帧数
    cap.set(cv2.CAP_PROP_FPS, 30)
    # # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))


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
                cv2.imshow("test", frame)
                frames += 1
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
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


import re


def replace_emoji_with_entity(html):
    return re.sub(
        r"([^\x00-\x7F]+)",
        lambda c: "".join("&#{};".format(ord(char)) for char in c.group(1)),
        html,
    )


def main():
    # detect_cameras()
    get_FOURCC(2592, 1944)


if __name__ == "__main__":
    main()
