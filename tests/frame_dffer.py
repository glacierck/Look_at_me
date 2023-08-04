import cv2
import numpy as np

from my_insightface.insightface.app.screen import Screen
from my_insightface.insightface.app.camera import Camera

"""
帧差法，背景减除法几个不同的模型，计算效率在500万像素下cpu负荷较高，实时率差
"""
def frame_diff(limit_frames: int):
    cap = Camera('auto_focus', 'usb', resolution=(2592, 1944), fps=30).videoCapture
    start_time = cv2.getTickCount()
    _, frame1 = cap.read()
    _, frame2 = cap.read()
    frames = 0
    while frames < limit_frames:
        _, frame3 = cap.read()
        if not _:
            break
        # Convert the frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        # Get the difference between frames
        diff1 = cv2.absdiff(gray1, gray2)
        diff2 = cv2.absdiff(gray2, gray3)

        # Apply thresholding to get the binary image
        _, thresh1 = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(diff2, 25, 255, cv2.THRESH_BINARY)

        # Combine the two thresholded images
        final = cv2.bitwise_and(thresh1, thresh2)
        resize = Screen.resize_image(final)
        # Display the result
        cv2.imshow('Motion Detection', resize)

        # Update the frames
        frame1 = frame2
        frame2 = frame3
        frames += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end_time = cv2.getTickCount()
    print(f"measured FPS of the video is {frames / ((end_time - start_time) / cv2.getTickFrequency())}")
    cap.release()
    cv2.destroyAllWindows()


def background_subtractor(limit_frames: int):
    cap = Camera('auto_focus', 'usb', resolution=(2592, 1944), fps=30).videoCapture
    # 创建背景减除器对象
    fgbg = cv2.createBackgroundSubtractorKNN()
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 对当前帧应用背景减除器
        fgmask = fgbg.apply(frame)

        # 使用形态学开运算去除噪声
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 寻找前景物体的轮廓
        contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 忽略面积过小的轮廓
            if cv2.contourArea(contour) < 50000:
                continue

            # 获取轮廓的边界矩形，然后在原始图像上绘制该矩形
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        resize = Screen.resize_image(frame)
        cv2.imshow('Frame', resize)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def frame_diff_test(limit_frames: int):
    camera = Camera('auto_focus', 'usb', resolution=(2592, 1944), fps=30).videoCapture

    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
    background = None

    while True:
        # 读取视频流
        grabbed, frame_lwpCV = camera.read()
        # 对帧进行预处理，先转灰度图，再进行高斯滤波。
        # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
        gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
        gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

        # 将第一帧设置为整个输入的背景
        if background is None:
            background = gray_lwpCV
            continue
        # 对于每个从背景之后读取的帧都会计算其与北京之间的差异，并得到一个差分图（different map）。
        # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
        diff = cv2.absdiff(background, gray_lwpCV)
        diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
        diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀

        # 显示矩形框
        contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
        for c in contours:
            if cv2.contourArea(c) < 1500:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
                continue
            (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
            cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('contours', frame_lwpCV)
        cv2.imshow('dis', diff)

        key = cv2.waitKey(1) & 0xFF
        # 按'q'健退出循环
        if key == ord('q'):
            break
    # When everything done, release the capture
    camera.release()
    cv2.destroyAllWindows()


def main():
    frame_diff(10000)


if __name__ == '__main__':
    main()
