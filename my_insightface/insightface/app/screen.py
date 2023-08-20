from collections import deque
from timeit import default_timer as current_time
import cv2
import numpy as np
from numpy import ndarray
from ..data import LightImage


class Screen:
    def __init__(self):
        self._interval_time_sum_cnt = 0
        self._frame_cnt = 0
        self.image_size = None
        self._interval_time = deque(maxlen=2)
        self._interval = deque(maxlen=200)
        self._temp_sum = 0
        self.ave_fps = 0
        self._pre = 0
        self._cur = 0

    def show(self, image2show: LightImage) -> ndarray:
        self._frame_cnt += 1
        if self._frame_cnt > 10000:
            self._frame_cnt = 0
        image2show_nd_arr = self.resize_image(self._draw_on(image2show))
        res = self._draw_fps(image2show_nd_arr)
        # cv2.imshow('screen', image2show_nd_arr)
        return res

    @staticmethod
    def resize_image(image2resize: ndarray, target_size: tuple[int,int] =(1080, 560)) -> ndarray:
        """
        cv2.INTER_AREA：区域插值 效果最好，但速度慢
        cv2.INTER_CUBIC ：三次样条插值，效率居中
        cv2.INTER_LINEAR ：线性插值，效果最差，但速度最快
        :param target_size: 理想的目标大小
        :param image2resize: image to resize
        :return: resized image ndarray
        """
        original_size = image2resize.shape[:2]
        ratio = min(target_size[0] / original_size[1], target_size[1] / original_size[0])  # 计算缩放比例
        new_size = (int(original_size[1] * ratio), int(original_size[0] * ratio))  # 等比例缩放
        resized_image = cv2.resize(image2resize, new_size, interpolation=cv2.INTER_AREA)
        return resized_image

    def _draw_bbox(self, dimg, bbox, bbox_color):
        """
        only draw the bbox beside the corner,and the corner is round
        :param dimg: img to draw bbox on
        :param bbox: face bboxes
        :param bbox_color: bbox color
        :return: no return
        """
        # 定义矩形的四个角的坐标
        pt1 = (bbox[0], bbox[1])
        pt2 = (bbox[2], bbox[3])
        self.bbox_thickness = 4
        # 定义直角附近线段的长度
        line_len = int(0.08 * (pt2[0] - pt1[0]) + 0.06 * (pt2[1] - pt1[1]))
        inner_line_len = int(line_len * 0.718) if bbox_color != (0, 0, 255) else line_len

        def draw_line(_pt1, _pt2):
            cv2.line(dimg, _pt1, _pt2, bbox_color, self.bbox_thickness)

        draw_line((pt1[0], pt1[1]), (pt1[0] + inner_line_len, pt1[1]))
        draw_line((pt1[0], pt1[1]), (pt1[0], pt1[1] + line_len))
        draw_line((pt2[0], pt1[1]), (pt2[0] - inner_line_len, pt1[1]))
        draw_line((pt2[0], pt1[1]), (pt2[0], pt1[1] + line_len))
        draw_line((pt1[0], pt2[1]), (pt1[0] + inner_line_len, pt2[1]))
        draw_line((pt1[0], pt2[1]), (pt1[0], pt2[1] - line_len))
        draw_line((pt2[0], pt2[1]), (pt2[0] - inner_line_len, pt2[1]))
        draw_line((pt2[0], pt2[1]), (pt2[0], pt2[1] - line_len))

    def _draw_text(self, dimg, box, name, color):
        # 文字信息显示
        self.font_scale = 3
        # 设置文本的位置，将文本放在人脸框的下方
        text_position = (box[0], box[3] + 22)
        # ft2 = cv2.freetype.createFreeType2()
        # ft2.loadFontData(fontFileName='simhei.ttf', id=0)
        # ft2.putText(img=dimg,
        #             text=name,
        #             org=text_position,
        #             fontHeight=20,
        #             color=color,
        #             thickness=-1,
        #             line_type=cv2.LINE_AA,
        #             bottomLeftOrigin=True)
        # 添加文本  中文问题还没有解决
        cv2.putText(img=dimg,
                    text=name,
                    org=text_position,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=self.font_scale,
                    color=color,
                    thickness=3,
                    lineType=cv2.LINE_AA)

    def _draw_cross(self, dimg, bbox, color):
        rotate = True if color == (0, 0, 255) else False
        x1, y1, x2, y2 = bbox
        scale = 0.2
        # 计算中心点坐标
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        self._cross_line_thickness = 4 if color != (0, 0, 255) else 5
        # 计算十字架的长度
        length = int(min(x2 - x1, y2 - y1) * scale)

        if rotate:  # 如果需要旋转
            # 创建一个旋转矩阵
            M = cv2.getRotationMatrix2D((float(center_x), float(center_y)), 45, 1)
            length *= 2.5
            # 创建旋转前的十字架坐标
            original_cross = np.array([
                [center_x - length // 2, center_y],
                [center_x + length // 2, center_y],
                [center_x, center_y - length // 2],
                [center_x, center_y + length // 2]
            ], dtype=np.float32)
            # 将旋转矩阵应用到十字架坐标
            rotated_cross = cv2.transform(original_cross.reshape(-1, 1, 2), M).squeeze().astype(int)

            # 画出旋转后的十字架
            cv2.line(dimg, tuple(rotated_cross[0]), tuple(rotated_cross[1]), color, self._cross_line_thickness)
            cv2.line(dimg, tuple(rotated_cross[2]), tuple(rotated_cross[3]), color, self._cross_line_thickness)
        else:  # 不需要旋转
            cv2.line(dimg, (center_x - length // 2, center_y), (center_x + length // 2, center_y), color,
                     self._cross_line_thickness)
            cv2.line(dimg, (center_x, center_y - length // 2), (center_x, center_y + length // 2), color,
                     self._cross_line_thickness)

        return dimg

    def _draw_fps(self, image2show_nd_arr: ndarray):
        """
        取最近200次的时间间隔，计算平均fps，从而稳定FPS显示
        Args:
            image2show_nd_arr: image to draw FPS
        Returns: None
        """
        if self._pre == 0:
            self._pre = current_time()
        elif self._cur == 0:
            self._cur = current_time()
        else:
            self._pre = self._cur
            self._cur = current_time()
            interval = self._cur - self._pre
            if self._interval.__len__() < 200:
                self._temp_sum += interval
            elif self._interval.__len__() == 200:
                self._temp_sum += interval
                self._temp_sum -= self._interval.popleft()
            self._interval.append(interval)
            self.ave_fps = 1 / self._temp_sum * self._interval.__len__()
            cv2.putText(
                image2show_nd_arr,
                f"FPS = {self.ave_fps :.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        return image2show_nd_arr
    def _draw_on(self, image2draw_on: LightImage):
        dimg = image2draw_on.nd_arr

        for face in image2draw_on.faces:
            # face=[bbox, kps, det_score, color, match_info]
            bbox = face[0].astype(int)
            bbox_color = face[3][0]
            text_color = face[3][1]
            name = face[4].name if face[4] else 'unknown'

            # cross show
            if bbox_color == (50, 205, 255):  # 一直显示十字架
                self._draw_cross(dimg, bbox, bbox_color)
            elif bbox_color == (0, 0, 255) and self._frame_cnt % 8 == 0:
                # red cross blink
                self._draw_cross(dimg, bbox, bbox_color)
            # bbox show
            if bbox_color == (50, 205, 255) and self._frame_cnt % 5 == 0:
                # 黄色闪烁
                self._draw_bbox(dimg, bbox, bbox_color)
            elif bbox_color == (152, 251, 152):  # 绿色正常显示
                self._draw_bbox(dimg, bbox, bbox_color)
            elif bbox_color == (0, 0, 255):
                self._draw_bbox(dimg, bbox, bbox_color)
            # text show
            self._draw_text(dimg, bbox, name, text_color)
        return dimg
