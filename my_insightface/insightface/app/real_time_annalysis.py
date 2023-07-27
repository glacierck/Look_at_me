from __future__ import annotations

import collections
import heapq
import queue
from collections import deque
from time import sleep
from threading import Event
import numpy as np
from numpy import ndarray
from sympy.core.random import random

from .common import Face
from .face_analysis import Milvus2Search
from pathlib import Path
import cv2
from timeit import default_timer as current_time
import functools
from my_insightface.insightface.data.image import LightImage, draw_on
from my_insightface.insightface.app.face_analysis import MatchInfo
from scipy.optimize import linear_sum_assignment

COST_TIME = {}
threads_done = Event()


def cost_time_recording(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = current_time()
        res = func(*args, **kwargs)
        end = current_time()
        COST_TIME.setdefault(func.__name__, []).append(end - start)
        return res

    return wrapper


class FaceAnalysisVideoTest(Milvus2Search):
    def __init__(self, **kwargs):
        self._test_folder = kwargs.get('test_folder')
        self._video = None
        self._screen = Screen()
        super().__init__(**kwargs)

    @cost_time_recording
    def video_read(self, results: queue.Queue):
        video_dir = Path(f'..\\my_insightface\\insightface\\data\\images\\{self._test_folder}\\video')
        video_path = list(video_dir.glob('*.mp4'))[0]
        assert video_path.exists() and video_path.is_file(), f'video_path = {video_path}'
        self._video = cv2.VideoCapture(video_path.as_posix())
        self._imgs_of_video = 0
        print('video_read start')
        while not threads_done.is_set():
            ret, frame = self._video.read()
            if ret:
                results.put(
                    LightImage(nd_arr=frame, faces=[], screen_scale=(0, 0, frame.shape[1] - 1, frame.shape[0] - 1)))
                print(f'video_2_detect_queue.qsize() = {results.qsize()}')
                self._imgs_of_video += 1
            else:
                break

    @cost_time_recording
    def detect2update(self, jobs: queue.Queue):
        print('detect2update start')
        while not threads_done.is_set():
            try:
                to_update = jobs.get(timeout=2)
                self._screen.update(to_update)
                print(f'detect2update_queue.qsize() = {to_update.qsize()}')
            except queue.Empty:
                print('detect2update_queue is empty')
                break

    @cost_time_recording
    def image_continued_show(self, jobs: queue.Queue):

        self.show_times = 0
        print('image_continued_show start')
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                threads_done.set()
                break
            try:
                start = current_time()
                to_update = jobs.get(timeout=2)
                self._screen.update(to_update)
                self._screen.show(to_update)
                self.show_times += 1
                end_time = current_time()
                sleep_time = 0.025 - (end_time - start) if (end_time - start) < 0.025 else 0
                if sleep_time == 0:
                    print(f'Warning: sleep_time = {sleep_time}')
                sleep(sleep_time)
                print(f'detect_2_show_queue.qsize() = {jobs.qsize()}')
            except queue.Empty:
                print('detect_2_show_queue is empty')
                break

    def test_stop(self):
        self.stop_milvus()
        print('test_of_face_analysis ends!')


class Detector:
    def __init__(self):
        from my_insightface.insightface.model_zoo.model_zoo import get_model
        root: Path = Path.cwd().parents[1].joinpath('models\\insightface\\det_2.5g.onnx')
        self.detector_model = get_model(root, providers=('CUDAExecutionProvider', 'CPUExecutionProvider'))
        prepare_params = {'ctx_id': 0,
                          'det_thresh': 0.5,
                          'input_size': (320, 320)}
        self.detector_model.prepare(**prepare_params)

    def __call__(self, img2detect: LightImage) -> LightImage:
        detect_params = {'max_num': 0, 'metric': 'default'}
        bboxes, kpss = self.detector_model.detect(img2detect.nd_arr, **detect_params)
        if bboxes.shape[0] > 0:
            for i in range(bboxes.shape[0]):
                kps = kpss[i] if kpss is not None else None
                bbox = bboxes[i, 0:4]
                det_score = bboxes[i, 4]
                face = [bbox, kps, det_score, None, None]
                img2detect.faces.append(face)
        else:
            print('no face detected')
        return img2detect


@cost_time_recording
def detect_thread(detector, jobs: queue.Queue, results: queue.Queue):
    print('detect_thread start')
    while not threads_done.is_set():
        try:
            reciver = [jobs.get(timeout=1) for _ in range(50)]
        except queue.Empty:
            print('detect_thread,queue.Empty')
            break
        for job in reciver:
            image_2_show = detector(job)
            results.put(image_2_show)


class Extractor:
    def __init__(self):
        from my_insightface.insightface.model_zoo.model_zoo import get_model
        root: Path = Path.cwd().parents[1].joinpath('models\\insightface\\irn50_glint360k_r50.onnx')
        self.rec_model = get_model(root, providers=('CUDAExecutionProvider', 'CPUExecutionProvider'))
        self.rec_model.prepare(ctx_id=0)

    def __call__(self, img2extract: LightImage) -> LightImage:
        for i in range(img2extract.faces.__len__()):
            face = Face(bbox=img2extract.faces[i][0],
                        kps=img2extract.faces[i][1],
                        det_score=img2extract.faces[i][2])
            self.rec_model.get(img2extract.nd_arr, face)
            img2extract.faces[i][3] = face.embedding
        return img2extract


@cost_time_recording
def extract_thread(extractor, jobs: queue.Queue, results: queue.Queue):
    print('extract_thread')
    while not threads_done.is_set():
        try:
            image_2_show = extractor(jobs.get(timeout=0.01))
            results.put(image_2_show)
            print(f'rec_2_show_queue.qsize() = {results.qsize()}')
        except queue.Empty:
            print('queue.Empty')
            continue


class kalman_filter:
    def __init__(self, x: float, y: float):
        self._kalman_fil = cv2.KalmanFilter(4, 2)
        # 常速度模型，所以状态转移矩阵A为4*4
        self._kalman_fil.transitionMatrix = np.array([[1., 0., 1., 0.],
                                                      [0., 1., 0., 1.],
                                                      [0., 0., 1., 0.],
                                                      [0., 0., 0., 1.]], np.float32)
        # 测量矩阵H为2*4，因为测量值是坐标值，所以H为2*4
        self._kalman_fil.measurementMatrix = np.array([[1., 0., 0., 0.],
                                                       [0., 1., 0., 0.]], np.float32)
        # 过程噪声协方差矩阵Q为4*4
        # 表示了关于状态估计的不确定性。我们在这里初始化为较小的值，表示我们相信我们的初始状态估计。
        self._kalman_fil.processNoiseCov = np.array([[1., 0., 0., 0.],
                                                     [0., 1., 0., 0.],
                                                     [0., 0., 1., 0.],
                                                     [0., 0., 0., 1.]], np.float32) * 1e-4
        # 定义测量噪声协方差矩阵，表示了我们对观测噪声的不确定性
        self._kalman_fil.measurementNoiseCov = np.array([[1., 0.],
                                                         [0., 1.]], np.float32) * 1e-1
        # 状态值可以在第一次进行人脸检测时设置，例如：
        first_detected_position = np.array([[x], [y], [0.], [0.]], np.float32)  # 你需要根据实际情况替换这个值
        self._kalman_fil.statePre = first_detected_position
        self._kalman_fil.statePost = first_detected_position

        # 先验误差估计协方差矩阵和后验误差估计协方差矩阵可以在初始化时设置为单位矩阵，然后它们将在运行过程中进行自我调整
        self._kalman_fil.errorCovPre = np.eye(4, dtype=np.float32)
        self._kalman_fil.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self):
        return self._kalman_fil.predict()

    def correct(self, x: float, y: float):
        return self._kalman_fil.correct(np.array([[x], [y]], np.float32))


class Target:
    def __init__(self, id: int, position: np.ndarray,
                 screen_scale: tuple[int, int, int, int], kps: np.ndarray):
        self._kalman_filter = None
        self.id = id
        self.position = position  # [x1, y1, x2, y2]
        self.kps = kps
        self._match_info: MatchInfo = MatchInfo(face_id=-1, name='', score=0.0)
        self._satisfied_lasting_time: float = 0.0
        self._screen_scale = screen_scale  # [x1, y1, x2, y2]

    def set_kalman_filter(self, x: float, y: float):
        self._kalman_filter = kalman_filter(x, y)

    def update_kalman_filter(self, x: float, y: float):
        self._kalman_filter.correct(x, y)

    def update_pos(self, new_target: Target):
        self.position = new_target.position
        self.kps = new_target.kps

    @property
    def screen_height(self):
        return self._screen_scale[3] - self._screen_scale[1]

    @property
    def screen_width(self):
        return self._screen_scale[2] - self._screen_scale[0]

    @property
    def bbox_width(self):
        return self.position[2] - self.position[0]

    @property
    def bbox_height(self):
        return self.position[3] - self.position[1]

    @property
    def name(self):
        if self.if_matched and self._match_info.name:
            return self._match_info.name
        else:
            return f'target_{self.id}'

    @property
    def time_satified(self) -> bool:
        return self._satisfied_lasting_time > 0.5

    @property
    def scale_satified(self) -> bool:
        target_area = self.bbox_width * self.bbox_height
        screen_area = self.screen_height * self.screen_width
        return (target_area / screen_area) > 0.1

    @property
    def if_matched(self) -> bool:
        return self._match_info.face_id != -1

    @property
    def rec_satified(self) -> bool:
        if self.scale_satified and not self.if_matched:
            return True
        elif self.if_matched and self.scale_satified and self.time_satified:
            return True
        else:
            return False

    @property
    def max_distance(self) -> float:
        return np.sqrt(self.screen_height ** 2 + self.screen_width ** 2)

    @property
    def centroid(self) -> np.ndarray[2]:
        bbox = self.position
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    @property
    def predict_centroid(self) -> np.ndarray:
        if self._kalman_filter is None:
            raise ValueError('kalman filter is not set.')
        return self._kalman_filter.predict()

    def centroid_distance(self, target: Target) -> float:
        # 和预测位置的距离
        distance = np.sqrt((self.predict_centroid[0] - target.centroid[0]) ** 2
                           + (self.predict_centroid[1] - target.centroid[1]) ** 2)
        return distance

    def direction_similarity(self, target: Target) -> float | ndarray:
        # 和预测方向的相似度
        predict_vector = (self.predict_centroid[0] - self.centroid[0],
                          self.predict_centroid[1] - self.centroid[1])
        if predict_vector[0] == 0.0 and predict_vector[1] == 0.0:
            return 1.0
        true_vector = (target.centroid[0] - self.centroid[0], target.centroid[1] - self.centroid[1])
        print(predict_vector, true_vector)
        normed_predict_vector = predict_vector / np.linalg.norm(predict_vector)
        normed_true_vector = true_vector / np.linalg.norm(true_vector)
        cos = np.dot(normed_predict_vector, normed_true_vector)
        return cos

    def iou(self, target: Target) -> float:
        bbox1 = self.position
        bbox2 = target.position
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        iou = intersection / union
        return iou

    def combined_measure(self, target: Target) -> float:
        direction_similarity = self.direction_similarity(target)
        normalized_centroid_distance = 1 - self.centroid_distance(target) / self.max_distance
        iou = self.iou(target)
        return normalized_centroid_distance * 0.6 + iou * 0.2 + direction_similarity * 0.2


class Screen:
    def __init__(self):
        self._targets: dict = {}
        self._screen_scale = (0, 0)
        self._max_ids = -1  # 目前最大的id
        self._recycled_ids = []  # 初始化回收的id优先队列
        self._index2key: list[int] = []  # 用于匈牙利算法的索引

    def update(self, image2update: LightImage):
        new_targets = [Target(id=-1, position=face[0],
                              screen_scale=image2update.screen_scale
                              , kps=face[1]) for face in image2update.faces]
        # 第一次直接赋值
        if not self._targets:
            # 初始化新目标的卡尔曼滤波器
            for target in new_targets:
                target.set_kalman_filter(*target.centroid)
            self._set_ids(new_targets)
            self._targets = {target.id: target for target in new_targets}

        else:
            # 根据匈牙利算法进行匹配，更新target状态
            cost_matrix = self.compute_cost_matrix(new_targets)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] > 0.5:
                    # This is a new target
                    new_targets[j].set_kalman_filter(*new_targets[j].centroid)
                    new_targets[j].id = self._generate_id()
                    if new_targets[j].id in self._targets:
                        raise ValueError('Target id already exists.')
                    self._targets[new_targets[j].id] = new_targets[j]
                else:
                    # This is an old target, update its properties
                    key = self._index2key[i]
                    self._index2key[i] = None
                    self._targets[key].update_pos(new_targets[j])
                    self._targets[key].update_kalman_filter(*new_targets[j].centroid)
            # Remove unmatched old targets
            for key in self._index2key:
                if key is not None and key in self._targets:
                    heapq.heappush(self._recycled_ids, key)
                    del self._targets[key]

    def compute_cost_matrix(self, new_targets: list[Target]) -> np.ndarray:
        cost_matrix = np.zeros((len(self._targets), len(new_targets)))
        for i, old_target in enumerate(self._targets.values()):
            self._index2key.append(old_target.id)
            for j, new_target in enumerate(new_targets):
                error = 1 - old_target.combined_measure(new_target)
                if error > 0.5:
                    cost_matrix[i, j] = error
                else:
                    cost_matrix[i, j] = 1
        return cost_matrix

    def _generate_id(self) -> int:
        try:
            return heapq.heappop(self._recycled_ids)
        except IndexError:
            self._max_ids += 1
            return self._max_ids

    def _set_ids(self, targets: list[Target]):
        for target in targets:
            target.id = self._generate_id()

    def show(self, image2show: LightImage):
        image2show_nd_arr = self._draw_on(image2show)
        cv2.imshow('screen', image2show_nd_arr)
        return image2show

    def _draw_on(self, image2draw_on: LightImage):
        dimg = image2draw_on.nd_arr
        red = (0, 0, 127)
        yellow = (0, 127, 127)
        light_green = (152, 251, 152)
        lavender = (238, 130, 238)
        blue = (127, 0, 0)

        for target in self._targets.values():
            # color choose
            if target.if_matched:
                bbox_color = light_green
                name_color = light_green
            elif target.scale_satified:
                bbox_color = yellow
                name_color = yellow
            else:
                bbox_color = lavender
                name_color = lavender
            # face=[bbox, kps, det_score, match_info]
            box = target.position.astype(int)
            kps = target.kps.astype(int)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), bbox_color, 1)
            if kps is not None:
                # print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            # 文字信息显示
            font_scale = 1
            # 设置文本的位置，将文本放在人脸框的上方
            text_position = (box[0] + 6, box[3] - 6)
            # 添加文本
            cv2.putText(img=dimg,
                        text=target.name,
                        org=text_position,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=font_scale,
                        color=name_color,
                        thickness=2,
                        lineType=cv2.LINE_AA)
        return dimg
