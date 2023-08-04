from threading import Thread
from queue import Queue

from performance_test import ave_fps_test
from my_insightface.insightface.app.multi_thread_analysis import MultiThreadFaceAnalysis, COST_TIME, threads_done

video_2_show = Queue(maxsize=400)


def camera_fps_test(resolution: tuple[int, int], fps: int) -> tuple[tuple[str], tuple[str]]:
    camera_params = {'app': 'auto_focus', 'approach': 'usb',
                     'fps': fps, 'resolution': resolution}
    test = MultiThreadFaceAnalysis(test_folder='test_02', camera_params=camera_params, identifier_params={})
    try:
        video_read_thread = Thread(target=test.video_read, args=(video_2_show,))
        image_show_thread = Thread(target=test.image_show, args=(video_2_show,))
        video_read_thread.start()
        image_show_thread.start()
        video_read_thread.join()
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
    ave_fps_test('camera_test.json', camera_fps_test)


if __name__ == '__main__':
    main()
