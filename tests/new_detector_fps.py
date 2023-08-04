from threading import Thread
from queue import Queue

import cv2

from performance_test import ave_fps_test
from my_insightface.insightface.app.multi_thread_analysis import MultiThreadFaceAnalysis, COST_TIME
from my_insightface.insightface.app.screen import Screen
from my_insightface.insightface.data import LightImage

video_2_detect_queue = Queue(maxsize=400)
detect_2_show_queue = Queue(maxsize=200)


def draw_bbox(image2show: LightImage):
    faces = image2show.faces
    for face in faces:
        bbox = list(map(int,face[0]))
        cv2.rectangle(image2show.nd_arr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    resized_image = Screen.resize_image(image2show.nd_arr)
    cv2.imshow('test', resized_image)


def new_detector_fps(resolution: tuple[int, int], fps: int) -> tuple[tuple[str], tuple[str]]:
    # auto_focus,manual_focus
    camera_params = {'app': 'auto_focus', 'approach': 'usb',
                     'fps': fps, 'resolution': resolution}
    test = MultiThreadFaceAnalysis(test_folder='test_02',
                                   camera_params=camera_params, identifier_params={'flush_threshold': 1000})
    try:
        video_read_thread = Thread(target=test.video_read, args=(video_2_detect_queue,))
        detect_thread = Thread(target=test.image2detect, args=(video_2_detect_queue, detect_2_show_queue))
        image_show_thread = Thread(target=test.image_show, args=(detect_2_show_queue,))
        video_read_thread.start()
        detect_thread.start()
        image_show_thread.start()

        video_read_thread.join()
        detect_thread.join()
        image_show_thread.join()
        print('all thread tasks done')
    except Exception as e:
        print(f'Exception occurs, error = {e}')
        raise e
    finally:
        ave_fps = round(test.show_times / COST_TIME['image_show'][0], 1)
        if test.camera.params['fps'][0] == test.camera.params['fps'][1]:
            res_fps = (test.camera.params['fps'][0], ave_fps)
        else:
            res_fps = (*test.camera.params['fps'], ave_fps)
        # test.test_stop()
    print('camera_fps_test done')
    return res_fps, test.camera.params['resolution']


def main():
    ave_fps_test('.\\test_results\\new_detector_fps.py.json', new_detector_fps)


if __name__ == '__main__':
    main()
