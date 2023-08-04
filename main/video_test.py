from __future__ import annotations

import functools
import concurrent.futures
import time
from threading import Thread, Event
import cv2
import numpy as np
from multiprocessing import Queue
from my_insightface.insightface.model_zoo.model_zoo import get_model
from my_insightface.insightface.app.useless.face_analysis import Milvus2Search
from my_insightface.insightface.app.common import Face
from typing import *
from pathlib import Path
from my_insightface.insightface.data.image import Image
from timeit import default_timer as current_time

COST_TIME = {}
done = Event()

video_read_queue = Queue()
detected_Image_queue = Queue()
recognition_Image_queue = Queue()
milvus_search_queue = Queue()
image_show_queue = Queue()


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
        self._test_folder = None
        self._video = None
        super().__init__(**kwargs)

    @cost_time_recording
    def load_registered_data(self, test_folder: str, refresh: bool = False) -> List[Face]:
        self._test_folder = test_folder
        self.load_registered_faces(None, None, test_folder=test_folder, img_folder='known', refresh=refresh)
        return self.registered_faces

    @cost_time_recording
    def face_multi_search(self):
        print('face_multi_search start')
        while not done.is_set():
            if recognition_Image_queue.qsize() < 10:
                print(f'recognition_Image_queue.qsize() = {recognition_Image_queue.qsize()}')
                time.sleep(0.1)
                continue
            else:
                images = [recognition_Image_queue.get() for _ in range(10)]
                for matched_image in self.face_match(images):
                    image_show_queue.put(matched_image)

    @cost_time_recording
    def video_read(self):
        video_dir = Path(f'..\\my_insightface\\insightface\\data\\images\\{self._test_folder}\\video')
        video_path = list(video_dir.glob('*.mp4'))[0]
        assert video_path.exists() and video_path.is_file(), f'video_path = {video_path}'
        self._video = cv2.VideoCapture(video_path.as_posix())
        self._imgs_of_video = 0
        print('video_read start')
        while True:
            ret, frame = self._video.read()
            if ret:
                video_read_queue.put(Image(nd_arr=frame))
                print(f'video_read_queue.qsize() = {video_read_queue.qsize()}')
                time.sleep(0.1)
                self._imgs_of_video += 1
            else:
                break

    @cost_time_recording
    def image_continued_show(self):
        self.show_times = 0
        print('image_continued_show start')
        while True:
            if image_show_queue.qsize() < 10:
                print(f'image_show_queue.qsize() = {image_show_queue.qsize()}')
                time.sleep(0.1)
                continue
            else:
                image_show = image_show_queue.get()
                self.show_times += 1
                cv2.imshow('video', image_show.draw_on())
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    done.set()
                    break

    def test_stop(self):
        self.stop_milvus()
        print('test_of_face_analysis ends!')


def show_test_results():
    ave_det_time = np.mean(COST_TIME['detect_face'])
    ave_rec_time = np.mean(COST_TIME['extract_embedding'])
    ave_match_time = np.mean(COST_TIME['face_search'])

    ave_video_read_time = COST_TIME['video_read']
    ave_detect_multiprocess_time = np.mean(COST_TIME['detect_multiprocess'])
    ave_extract_multiprocess_time = np.mean(COST_TIME['extract_multiprocess'])
    ave_face_multi_search_time = np.mean(COST_TIME['face_multi_search'])
    ave_fps = 1 / (COST_TIME['image_continued_show'])

    print('\ntest cost time and accuracy as follows',
          'load_registered_data cost time = {COST_TIME["load_registered_data"][0]:.4f} sec',
          f'ave_det_time cost time = {ave_det_time:.4f} sec',
          f'ave_match_time = {ave_match_time:.6f} sec',
          f'ave_rec_time = {ave_rec_time:.4f} sec',
          f'ave_fps = {ave_fps:.4f} fps',
          f'ave_video_read_time = {ave_video_read_time:.4f} sec',
          f'ave_detect_multiprocess_time = {ave_detect_multiprocess_time:.4f} sec',
          f'ave_extract_multiprocess_time = {ave_extract_multiprocess_time:.4f} sec',
          f'ave_face_multi_search_time = {ave_face_multi_search_time:.4f} sec',
          sep='\n')


class Detector:
    def __init__(self):
        from my_insightface.insightface.model_zoo.model_zoo import get_model
        root: Path = Path.cwd().parents[1].joinpath('models\\insightface\\det_2.5g.onnx')
        self.detector_model = get_model(root, providers=('CUDAExecutionProvider', 'CPUExecutionProvider'))
        prepare_params = {'ctx_id': 0,
                          'det_thresh': 0.5,
                          'input_size': (320, 320)}
        self.detector_model.prepare(**prepare_params)

    def __call__(self, img2detect: Image) -> Image:
        detect_params = {'max_num': 0, 'metric': 'default'}
        img2detect.bboxes, img2detect.kpss = self.detector_model.detect(img2detect.nd_arr, **detect_params)
        return img2detect


@cost_time_recording
def detect_multiprocess():
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        detector = Detector()
        while not done.is_set():
            if video_read_queue.qsize() < 1:
                print(f'video_read_queue.qsize() = {video_read_queue.qsize()}')
                time.sleep(0.1)
                continue
            else:

                print('detect_multiprocess')
                detect_futures = [executor.submit(detector, video_read_queue.get()) for _ in range(3)]

                for future in detect_futures:
                    print(f'detect_futures = {future}')
                    detected_img = future.result()
                    print(f'detected_img = {detected_img}')
                    detected_Image_queue.put(detected_img)


class Extractor:
    def __init__(self):

        root: Path = Path.cwd().parents[1].joinpath('models\\insightface\\irn50_glint360k_r50.onnx')
        self.rec_model = get_model(root, providers=('CUDAExecutionProvider', 'CPUExecutionProvider'))
        self.rec_model.prepare(ctx_id=0)

    def __call__(self, img2extract: Image) -> Image:
        bboxes = img2extract.bboxes
        kpss = img2extract.kpss
        if bboxes.shape[0] == 0:
            print(f'failed to detect name:{img2extract.name}')
            return img2extract
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            self.rec_model.get(img2extract.nd_arr, face)
            img2extract.faces.append(face)
        return img2extract


@cost_time_recording
def extract_multiprocess():
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        extractor = Extractor()
        print('extract_multiprocess')
        while not done.is_set():
            if detected_Image_queue.qsize() < 10:
                print(f'detected_Image_queue.qsize() = {detected_Image_queue.qsize()}')
                time.sleep(0.1)
                continue
            else:

                extract_futures = [executor.submit(extractor, detected_Image_queue.get()) for _ in range(3)]
                for future in extract_futures:
                    print(f'extract_futures = {future}')
                    extracted_img = future.result()
                    print(f'extracted_img = {extracted_img}')
                    recognition_Image_queue.put(extracted_img)




def main():
    test_folder = 'test_03'
    test = FaceAnalysisVideoTest()
    try:
        test.load_registered_data(test_folder=test_folder, refresh=False)
        # test.load_video()

        funcs = [test.video_read, detect_multiprocess,
                 extract_multiprocess, test.face_multi_search,
                 test.image_continued_show]
        threads = []
        for fun in funcs:
            t = Thread( target=fun)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        show_test_results()
    except Exception as e:
        print(f'Exception occurs, error = {e}')
        raise e
    finally:
        test.test_stop()


if __name__ == '__main__':
    main()

