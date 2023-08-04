import heapq
import queue
from threading import Event
from typing import NamedTuple

import numpy as np
import requests

from my_insightface.insightface.app.sort_plus import associate_detections_to_trackers, KalmanBoxTracker
from my_insightface.insightface.app.common import Face
from pathlib import Path
import cv2
from timeit import default_timer as current_time
import functools
from my_insightface.insightface.data.image import LightImage
from database.milvus_standalone.common import MatchInfo
import subprocess

COST_TIME = {}
threads_done = Event()
__all__ = ['MultiThreadFaceAnalysis', 'COST_TIME', 'threads_done', 'Target',
           'Camera', 'Detector', 'Identifier', 'Screen', 'Extractor']


def cost_time_recording(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = current_time()
        res = func(*args, **kwargs)
        end = current_time()
        COST_TIME.setdefault(func.__name__, []).append(end - start)
        return res

    return wrapper


class MultiThreadFaceAnalysis:
    def __init__(self, test_folder: str, app: str, npz_refresh:bool,
                 server_refresh: bool = False, **kwargs):
        self.show_times = 0
        self.camera = Camera(test_folder=test_folder, app=app, **kwargs)  # http://192.168.0.102:4747/video
        self._detect = Detector()
        self._identifier = Identifier(self._detect, test_folder=test_folder,
                                      server_refresh=server_refresh,
                                      npz_refresh=npz_refresh, **kwargs)
        self._screen = Screen()

    def video_read(self, jobs: queue.Queue):
        print('video_read start')
        self.camera.read_video(jobs)

    @cost_time_recording
    def image2detect(self, jobs: queue.Queue, results: queue.Queue):
        print('detect_thread start')
        while not threads_done.is_set():
            try:
                detect_job = jobs.get(timeout=10)
                # print('image2detect.qsize() = ', jobs.qsize())
            except queue.Empty:
                print('detect_thread,queue.Empty')
                break
            image_2_show = self._detect(detect_job)
            results.put(image_2_show)

    @cost_time_recording
    def detect2identify(self, jobs: queue.Queue, results: queue.Queue):
        print('detect2identify start')
        while not threads_done.is_set():
            try:
                to_update = jobs.get(timeout=10)
                ide_res = self._identifier.identified_results(to_update)
                results.put(ide_res)
                # print(f'detect2identify.qsize() = {jobs.qsize()}')
            except queue.Empty:
                print('detect2identify is empty')
                break

    @cost_time_recording
    def image_show(self, jobs: queue.Queue):

        self.show_times = 0
        print('image_continued_show start')
        fps_start = 0
        fps_end = 0
        while not threads_done.is_set():
            if cv2.waitKey(1) & 0xFF == ord('q'):
                threads_done.set()
                break
            try:
                fps_start = start = current_time()
                to_update = jobs.get(timeout=10)
                if fps_end != 0:
                    cv2.putText(to_update.nd_arr, f'fps = {1 / (fps_start - fps_end):.2f}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self._screen.show(to_update)
                fps_end = current_time()
                self.show_times += 1
                end_time = current_time()
                sleep_time = 0.022 - (end_time - start) if (end_time - start) < 0.022 else 0
                if sleep_time == 0:
                    print(f'Warning: sleep_time = {sleep_time}')
                # sleep(sleep_time)
                # print(f'finally_show_queue.qsize() = {jobs.qsize()}')

            except queue.Empty:
                print('finally_show_queue is empty')
                break

    def test_stop(self):
        self._identifier.stop_milvus()
        print('test_of_face_analysis ends!')


class Camera:
    def __init__(self, app: str = '', approach: str = '', test_folder: str = 'test_01',
                 resolution: tuple = (1920, 1080)):
        self._imgs_of_video = 0
        self._test_folder = test_folder
        self._resolution = resolution
        order = [app, approach]
        self._video = None
        match order:
            case ['mp4', _]:
                video_dir = Path(__file__).absolute().parents[
                                2] / f'database\\milvus_standalone\\data\\{self._test_folder}\\video'
                video_path = list(video_dir.glob('*.mp4'))[0]
                assert video_path.exists() and video_path.is_file(), f'video_path = {video_path}'
                self._url = video_path.as_posix()
            case ['laptop', _]:
                self._url = 0
            case ['ip_webcam', 'usb']:
                self._url = 'http://localhost:8080/video'
                cmd = 'adb forward tcp:8080 tcp:8080'
                # 使用subprocess运行命令
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # 获取命令输出
                stdout, stderr = process.communicate()
                # 如果有输出，打印出来
                if stdout:
                    print('STDOUT:{}'.format(stdout))
                if stderr:
                    raise ValueError('STDERR:{}'.format(stderr))

            case ['ip_webcam', 'wifi']:
                self._url = 'http://192.168.0.103:8080/video'
            case _:
                raise ValueError(f'Wrong app or approach: {app}, {approach}')

            # url = url + f'/video?{resolution[0]}x{resolution[1]}'
        try:
            self._video = cv2.VideoCapture(self._url)
            # if not self.videoCapture.isOpened():
            #     # 如果打开视频失败，尝试发送请求到链接
            #     response = requests.get("http://192.168.0.102:4747/override")
            #     response.raise_for_status()  # 如果响应状态码不是200，引发HTTPError异常
            #
            #     # 再次尝试打开视频
            #     self.videoCapture = cv2.VideoCapture(str(url))
            if not self._video.isOpened():
                raise ValueError(
                    f"Could not open video source {self._url} even after sending request to override link")

        except requests.exceptions.RequestException as e:
            # 处理任何可能的请求错误
            print(f"Request to override link failed with error: {e}")
        # 设置帧数
        self._video.set(cv2.CAP_PROP_FPS, 60)
        self.fps = self._video.get(cv2.CAP_PROP_FPS)


    def read_video(self, results: queue.Queue):
        print('camera_read start')
        try:
            while not threads_done.is_set():
                ret, frame = self._video.read()
                if ret:
                    results.put(
                        LightImage(nd_arr=frame, faces=[], screen_scale=(0, 0, frame.shape[1] - 1, frame.shape[0] - 1)))
                    # print(f'video_2_detect_queue.qsize() = {results.qsize()}')
                    self._imgs_of_video += 1
                else:
                    break
        finally:
            self._video.release()


class Detector:
    def __init__(self):
        from my_insightface.insightface.model_zoo.model_zoo import get_model
        root: Path = Path(__file__).absolute().parents[3].joinpath('models\\insightface\\det_2.5g.onnx')
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


class Extractor:
    def __init__(self):
        from my_insightface.insightface.model_zoo.model_zoo import get_model
        root: Path = Path(__file__).absolute().parents[3].joinpath('models\\insightface\\irn50_glint360k_r50.onnx')
        self.rec_model = get_model(root, providers=('CUDAExecutionProvider', 'CPUExecutionProvider'))
        self.rec_model.prepare(ctx_id=0)

    def __call__(self, img2extract: LightImage,
                 bbox: np.ndarray[4, 2], kps: np.ndarray[5, 2], det_score: float) -> np.ndarray[512]:
        """
        get embedding of face from given target bbox and kps, and det_score
        :param img2extract: target at which image
        :param bbox: target bbox
        :param kps: target kps
        :param det_score: target det_score
        :return: face embedding
        """
        face = Face(bbox=bbox,
                    kps=kps,
                    det_score=det_score)
        self.rec_model.get(img2extract.nd_arr, face)
        return face.normed_embedding


class RawTarget(NamedTuple):
    id: int
    bbox: np.ndarray[4]
    kps: np.ndarray[5, 2]
    score: float = 0.0


class Target:
    def __init__(self, id: int, bbox: np.ndarray,
                 screen_scale: tuple[int, int, int, int], kps: np.ndarray, score: float = 0.0):

        self._hit_streak = 0  # frames of keeping existing in screen
        self._time_since_update = 0  # frames of keeping missing in screen
        self.id = id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.kps = kps
        self.score = score
        self.match_info: MatchInfo = MatchInfo(face_id=-1, name='', score=0.0)
        self.frames_since_reced: int = 0
        self._screen_scale = screen_scale  # [x1, y1, x2, y2]
        self._tracker: KalmanBoxTracker = KalmanBoxTracker(bbox)
        self.normed_embedding: np.ndarray[512] = np.zeros(512)

    def update_pos(self, bbox: np.ndarray, kps: np.ndarray, score: float):
        self.bbox = bbox
        self.kps = kps
        self.score = score

    def update_tracker(self, detect_tar: RawTarget):
        self._time_since_update = 0
        self._tracker.update(detect_tar.bbox)

    def old_enough(self, max_age: int) -> bool:
        return self._time_since_update > max_age

    def in_screen(self, min_hits: int) -> bool:
        return self._time_since_update < 1 and self._hit_streak >= min_hits

    def set_match_info(self, match_info: MatchInfo):
        self.match_info = match_info

    @property
    def get_raw_target(self) -> RawTarget:
        # get predicted bounding box from Kalman Filter
        if self._tracker is None:
            raise ValueError('tracker is None')
        bbox = self._tracker.predict()[0]
        # if failed to update before predicted bbox, reset the hit_streak
        # coming after the update_tracker is meaning that the target is  in screen continuously
        if self._time_since_update == 0:
            self._hit_streak += 1
        else:
            self._hit_streak = 0
        self._time_since_update += 1
        return RawTarget(id=self.id, bbox=bbox, kps=self.kps, score=self.score)

    @property
    def screen_height(self):
        return self._screen_scale[3] - self._screen_scale[1]

    @property
    def screen_width(self):
        return self._screen_scale[2] - self._screen_scale[0]

    @property
    def bbox_width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def name(self):
        if self.if_matched and self.match_info.name:
            return self.match_info.name
        else:
            return f'target[{self.id}]'

    @property
    def time_satified(self) -> bool:
        if not self.if_matched:
            return False
        elif self.frames_since_reced < 100:
            self.frames_since_reced += 1
            return False
        else:
            print(self.frames_since_reced)
            self.frames_since_reced = 0
            return True

    @property
    def scale_satified(self) -> bool:
        target_area = self.bbox_width * self.bbox_height
        screen_area = self.screen_height * self.screen_width
        return (target_area / screen_area) > 0.03

    @property
    def if_matched(self) -> bool:
        return self.match_info.face_id != -1

    @property
    def rec_satified(self) -> bool:
        if self.scale_satified and not self.if_matched and self.in_screen(3):
            return True
        elif self.if_matched and self.scale_satified and self.time_satified and self.in_screen(3):
            return True
        else:
            return False

    @property
    def colors(self):
        red = (0, 0, 255)
        yellow = (50, 205, 255)
        green = (152, 251, 152)
        lavender = (238, 130, 238)
        blue = (127, 0, 0)
        if self.if_matched:
            # 有匹配对象
            if self.match_info.score > 0.4:
                bbox_color = green
                name_color = green
            else:
                # 有匹配对象，但是匹配分数不够，定义为匹配失败的红色
                bbox_color = red
                name_color = red
        else:  # 还没有匹配到对象
            bbox_color = yellow
            name_color = yellow
        return bbox_color, name_color


class Identifier:
    def __init__(self, detector, max_age=120, min_hits=3, iou_threshold=0.3,
                 server_refresh=False, npz_refresh=False, test_folder='test_01'):
        from milvus_test.milvus_for_realtime import MilvusRealTime
        """
        :param detector:
        :param max_age: 超过这个帧数没被更新就删除
        :param min_hits: 超过这个帧数 才会被 识别
        :param iou_threshold: 卡尔曼滤波器的阈值
        :param server_refresh: milvus server 是否刷新
        :param npz_refresh: 是否用新的 npz 文件刷新 milvus server
        """
        self._targets: dict[int, Target] = {}
        self.max_age = max_age  # 超过该帧数没被更新就删除
        self.min_hits = min_hits  # 至少被检测到的次数才算
        self.iou_threshold = iou_threshold
        self._recycled_ids = []
        self._extractor = Extractor()
        self._detector = detector
        self._milvus = MilvusRealTime(test_folder=test_folder, refresh=server_refresh)
        if server_refresh:
            self._milvus.load_registered_faces(extractor=self._extractor,
                                               detector=self._detector, refresh=npz_refresh)
        else:
            self._milvus.load2RAM()
        self._frame_cnt = 0

        # test insert val
        self._test_ids = np.random.choice(range(100000), 100, replace=False).tolist()
        self._test_names = (np.random.choice(range(100000), 100, replace=False)).tolist()
        test_embeddings = np.random.uniform(0.1, 1, (100, 512))
        norms = np.linalg.norm(test_embeddings, axis=1,keepdims=True)
        self._test_embeddings = (test_embeddings / norms).tolist()

    def identified_results(self, image2identify: LightImage) -> LightImage:
        self._update(image2identify)
        self._extract(image2identify)
        # test 模拟搜索时候插入新的 可以慢速的插入，不影响平均fps
        if self._frame_cnt % 10000 == 0: #
            self._milvus.add_new_face(self._test_ids.pop(), self._test_names.pop(),
                                      self._test_embeddings.pop())
        self._search()
        image2identify.faces.clear()
        for i, target in enumerate(self._targets.values()):
            if not target.in_screen(self.min_hits):
                continue

            if target.match_info.face_id == -1:
                match_info = MatchInfo(face_id=-1, name=target.name, score=0.0)
                target.match_info = match_info
            image2identify.faces.append(
                [target.bbox, target.kps, target.score, target.colors, target.match_info])
        if self._frame_cnt < 100000:
            self._frame_cnt += 1
        else:
            self._frame_cnt = 0
        return image2identify

    def stop_milvus(self):
        self._milvus.stop_milvus()

    def _update(self, image2update: LightImage):
        # 更新目标
        detected_tars = [RawTarget(id=-1, bbox=face[0],
                                   kps=face[1], score=face[2]) for face in image2update.faces]
        # 第一次的时候，直接添加
        if self._targets:
            # 提取预测的位置
            predicted_tars = []
            to_del = []
            for i, tar in enumerate(self._targets.values()):
                raw_tar = tar.get_raw_target
                predicted_tars.append(raw_tar)
                pos = raw_tar.bbox
                if np.any(np.isnan(pos)):
                    to_del.append(tar.id)
            #  根据预测的位置清空即将消失的目标
            for k in to_del:
                assert k in self._targets, f'k = {k} not in self._targets'
                heapq.heappush(self._recycled_ids, k)
                del self._targets[k]

            # 根据预测的位置和检测的  **targets**  进行匹配
            matched, unmatched_det_tars, unmatched_pred_tars = associate_detections_to_trackers(detected_tars,
                                                                                                predicted_tars,
                                                                                                self.iou_threshold)
            # update pos and tracker for matched targets
            for pred_tar, detected_tar in matched:
                self._targets[pred_tar.id].update_pos(detected_tar.bbox, detected_tar.kps, detected_tar.score)
                self._targets[pred_tar.id].update_tracker(detected_tar)
        else:
            unmatched_det_tars = detected_tars
        # add new targets
        for detected_tar in unmatched_det_tars:
            new_id = self._generate_id()
            assert new_id not in self._targets, f'new_id is already in self._targets'
            self._targets[new_id] = Target(id=new_id, bbox=detected_tar.bbox,
                                           screen_scale=image2update.screen_scale,
                                           kps=detected_tar.kps)
        self._clear_old_targets()

    def _clear_old_targets(self):
        # clear dead targets
        keys = []
        for tar in self._targets.values():
            # remove dead targets
            if tar.old_enough(self.max_age):
                keys.append(tar.id)
        for k in keys:
            try:
                del self._targets[k]
            except KeyError:
                print(f'KeyError: tar.id = {k}')
            else:
                heapq.heappush(self._recycled_ids, k)

    def _search(self):
        if not self._milvus:
            raise ValueError('milvus is not initialized')
        self._milvus.face_match([tar for tar in self._targets.values()
                                 if tar.normed_embedding.all()], 0.0)

    def _extract(self, image2extract: LightImage):
        for tar in self._targets.values():
            if tar.rec_satified:
                tar.normed_embedding = self._extractor(image2extract, tar.bbox, tar.kps, tar.score)
        return image2extract

    def _generate_id(self):
        try:
            return heapq.heappop(self._recycled_ids)
        except IndexError:
            return len(self._targets)


class Screen:
    def __init__(self):
        self._frame_cnt = 0

    def show(self, image2show: LightImage):
        self._frame_cnt += 1
        if self._frame_cnt > 10000:
            self._frame_cnt = 0
        image2show_nd_arr = self._draw_on(image2show)
        cv2.imshow('screen', image2show_nd_arr)
        return image2show

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
        self.bbox_thickness = 2
        # 定义直角附近线段的长度
        line_len = int(0.08 * (pt2[0] - pt1[0]) + 0.06 * (pt2[1] - pt1[1]))
        inner_line_len = int(line_len * 0.718) if bbox_color != (0, 0, 255) else line_len

        def draw_line(_pt1, _pt2):
            cv2.line(dimg, _pt1, _pt2, bbox_color, self.bbox_thickness)

        if bbox_color == (0, 0, 255):
            # if red color, draw rectangle directly
            cv2.rectangle(dimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 1)
            r = 0  # 画直角的时候，不需要圆角
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
        self.font_scale = 1
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
                    thickness=1,
                    lineType=cv2.LINE_AA)

    def _draw_cross(self, dimg, bbox, color):
        rotate = True if color == (0, 0, 255) else False
        x1, y1, x2, y2 = bbox
        scale = 0.2
        # 计算中心点坐标
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        self._cross_line_thickness = 2 if color != (0, 0, 255) else 3
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
