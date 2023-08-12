import queue
from threading import Event
from time import sleep

import cv2
from timeit import default_timer as current_time
import functools

from .camera import Camera
from .detector import Detector
from .identifier import Identifier
from .screen import Screen

COST_TIME = {}
threads_done = Event()
__all__ = ["MultiThreadFaceAnalysis", "COST_TIME", "threads_done"]


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
    def __init__(
            self,
            test_folder: str = "test_01",
            camera_params: dict = None,
            identifier_params: dict = None,
    ):
        self.show_times = 0
        camera_params = {
                            "app": "",
                            "approach": "",
                            "test_folder": test_folder,
                            "resolution": (1920, 1080),
                            "fps": 30,
                        } | camera_params
        identifier_params = {
                                "server_refresh": False,
                                "npz_refresh": False,
                                "flush_threshold": 1000,
                            } | identifier_params
        self.camera = Camera(**camera_params)  # http://192.168.0.102:4747/video
        self._detect = Detector()
        self._identifier = Identifier(
            self._detect, test_folder=test_folder, **identifier_params
        )
        self._screen = Screen()

    def video_read(self, jobs: queue.Queue):
        print("video_read start")
        self.camera.read_video(jobs)

    @cost_time_recording
    def image2detect(self, jobs: queue.Queue, results: queue.Queue):
        print("detect_thread start")
        while not threads_done.is_set():
            try:
                # print('image2detect.qsize() = ', jobs.qsize())
                detect_job = jobs.get(timeout=10)

            except queue.Empty:
                print("detect_thread,queue.Empty")
                break
            image_2_show = self._detect(detect_job)
            results.put(image_2_show)

    @cost_time_recording
    def detect2identify(self, jobs: queue.Queue, results: queue.Queue):
        print("detect2identify start")
        while not threads_done.is_set():
            try:
                # print(f'detect2identify.qsize() = {jobs.qsize()}')
                to_update = jobs.get(timeout=10)

                ide_res = self._identifier.identified_results(to_update)
                results.put(ide_res)

            except queue.Empty:
                print("detect2identify is empty")
                break

    @cost_time_recording
    def image_show(self, jobs: queue.Queue):
        # from new_detector_fps import draw_bbox
        self.show_times = 0
        print("image_continued_show start")
        fps_start = 0
        fps_end = 0
        while not threads_done.is_set():
            start = current_time()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                threads_done.set()
                break
            try:
                # print(f'finally_show_queue.qsize() = {jobs.qsize()}')
                to_update = jobs.get(timeout=15)
                if fps_end != 0:
                    cv2.putText(
                        to_update.nd_arr,
                        f"fps = {1 / (fps_end - fps_start):.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                fps_start = current_time()
                self._screen.show(to_update)
                # draw_bbox(to_update)
                self.show_times += 1
                end_time = current_time()
                sleep_time = (
                    1 / 30 - (end_time - start) if (end_time - start) < 1 / 30 else 0
                )
                if sleep_time == 0:
                    print(f"Warning: sleep_time = {sleep_time}")
                # sleep(sleep_time)
                fps_end = current_time()

            except queue.Empty:
                print("finally_show_queue is empty")
                break

    def test_stop(self):
        self._identifier.stop_milvus()
        print("test_of_face_analysis ends!")
