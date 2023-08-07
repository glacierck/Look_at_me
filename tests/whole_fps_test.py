from threading import Thread
from queue import Queue

from performance_test import ave_fps_test
from my_insightface.insightface.app.multi_thread_analysis import MultiThreadFaceAnalysis, COST_TIME

video_2_detect_queue = Queue(maxsize=400)
detect_2_rec_queue = Queue(maxsize=200)
rec_2_show_queue = Queue(maxsize=400)


def whole_fps_test(resolution: tuple[int, int], fps: int) -> tuple[tuple[str], tuple[str]]:
    # auto_focus,manual_focus
    camera_params = {'app': 'auto_focus', 'approach': 'usb',
                     'fps': fps, 'resolution': resolution}
    identifier_params = {'server_refresh': False, 'npz_refresh': False}
    test = MultiThreadFaceAnalysis(test_folder='test_02',
                                   camera_params=camera_params, identifier_params=identifier_params)
    try:
        video_read_thread = Thread(target=test.video_read, args=(video_2_detect_queue,))
        detect_thread = Thread(target=test.image2detect,
                               args=(video_2_detect_queue, detect_2_rec_queue))
        identify_thread = Thread(target=test.detect2identify,
                                 args=(detect_2_rec_queue, rec_2_show_queue))
        image_show_thread = Thread(target=test.image_show, args=(rec_2_show_queue,))
        video_read_thread.start()
        detect_thread.start()
        identify_thread.start()
        image_show_thread.start()

        video_read_thread.join()
        detect_thread.join()
        identify_thread.join()
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
        test.test_stop()
    print('camera_fps_test done')
    return res_fps, test.camera.params['resolution']


def main():
    ave_fps_test('whole_fps_test.json', whole_fps_test)


if __name__ == '__main__':
    main()
