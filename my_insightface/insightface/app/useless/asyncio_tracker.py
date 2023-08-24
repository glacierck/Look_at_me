import asyncio
import functools
from asyncio import Queue
from pathlib import Path

import cv2
from my_insightface.insightface.app.screen import Drawer

from my_insightface.insightface.app.camera import Camera
from my_insightface.insightface.app.detector import Detector
from my_insightface.insightface.app.identifier import Identifier
from my_insightface.insightface.app.multi_thread_analysis import (COST_TIME, current_time)
from my_insightface.insightface.data.image import LightImage


def cost_time_recording(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = current_time()
        try:
            await func(*args, **kwargs)
        finally:
            end = current_time()
            COST_TIME.setdefault(func.__name__, []).append(end - start)
            return end - start

    return wrapper


"""

非常卡顿
"""


class AsyncioFaceAnalysis:
    def __init__(self, test_folder: str):
        self.show_times = 0
        self._test_folder = test_folder
        self._video = None
        self._camera = Camera(test_folder)
        self._identifier = Identifier()
        self._screen = Drawer()
        self._detect = Detector()
        self.threads_done = asyncio.Event()

    async def video_read(self, results: Queue):
        video_dir = Path(f'..\\my_insightface\\insightface\\data\\images\\{self._test_folder}\\video')
        video_path = list(video_dir.glob('*.mp4'))[0]
        assert video_path.exists() and video_path.is_file(), f'video_path = {video_path}'
        self._video = cv2.VideoCapture(video_path.as_posix())
        self._imgs_of_video = 0
        print('video_read start')
        while not self.threads_done.is_set():
            ret, frame = self._video.read()
            if ret:
                await results.put(
                    LightImage(nd_arr=frame, faces=[], screen_scale=(0, 0, frame.shape[1] - 1, frame.shape[0] - 1)))
                print(f'video_2_detect_queue.qsize() = {results.qsize()}')
                self._imgs_of_video += 1
            else:
                break

    async def image2detect(self, jobs: Queue, results: Queue):
        print('detect_thread start')
        while not self.threads_done.is_set():
            try:
                detect_jobs = [await asyncio.wait_for(jobs.get(), timeout=5) for _ in range(50)]
            except asyncio.TimeoutError:
                print('detect_thread,queue.Empty')
                break
            for image in detect_jobs:
                image_2_show = self._detect(image)
                await results.put(image_2_show)

    async def detect2identify(self, jobs: Queue, results: Queue):
        print('detect2identify start')
        while not self.threads_done.is_set():
            try:
                to_update = await asyncio.wait_for(jobs.get(), timeout=5)
                ide_res = self._identifier.identified_results(to_update)
                await results.put(ide_res)
                print(f'detect2identify.qsize() = {jobs.qsize()}')
            except asyncio.TimeoutError:
                print('detect2identify is empty')
                break

    async def image_show(self, jobs: Queue):

        self.show_times = 0
        print('image_continued_show start')
        fps_start = 0
        fps_end = 0
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.threads_done.set()
                break
            try:
                fps_start = start = current_time()
                to_update = await asyncio.wait_for(jobs.get(), timeout=5)
                if fps_end != 0:
                    cv2.putText(to_update.nd_arr, f'fps = {1 / (fps_start - fps_end):.2f}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self._screen.show(to_update)
                fps_end = current_time()
                self.show_times += 1
                end_time = current_time()
                sleep_time = 0.025 - (end_time - start) if (end_time - start) < 0.025 else 0
                if sleep_time == 0:
                    print(f'Warning: sleep_time = {sleep_time}')
                await asyncio.sleep(sleep_time)
                print(f'detect_2_show_queue.qsize() = {jobs.qsize()}')

            except asyncio.TimeoutError:
                print('detect_2_show_queue is empty')
                break

    # def test_stop(self):
    #     self.stop_milvus()
    #     print('test_of_face_analysis ends!')
