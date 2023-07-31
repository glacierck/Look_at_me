import json
import pprint
from pathlib import Path
from threading import Thread
from queue import Queue

from my_insightface.insightface.app.real_time_tracker import MultiThreadFaceAnalysis, COST_TIME, threads_done

video_2_show = Queue(maxsize=400)


def ave_fps_test():
    if Path('ave_fps.json').exists():
        with open('ave_fps.json', 'r') as f:
            ave_fps = json.load(f)
        with open('std_fpss.json', 'r') as f:
            std_fpss = json.load(f)
    else:
        ave_fps = {}
        std_fpss = {}
    resolution = input('input resolution: ')
    measure_fps, std_fps = camera_fps_test(resolution=resolution)
    ave_fps[resolution] = measure_fps
    std_fpss[resolution] = std_fps
    with open('ave_fps.json', 'w') as f:
        json.dump(ave_fps, f)
    with open('std_fpss.json', 'w') as f:
        json.dump(std_fpss, f)
    pprint.pprint(ave_fps)
    pprint.pprint(std_fpss)


def camera_fps_test(resolution: str) -> float:
    test = MultiThreadFaceAnalysis(test_folder='test_04', app='ip_webcam', approach='usb')
    try:
        video_read_thread = Thread(target=test.video_read, args=(video_2_show,))
        image_show_thread = Thread(target=test.image_show, args=(video_2_show,))
        video_read_thread.start()
        image_show_thread.start()
        video_read_thread.join()
        image_show_thread.join()
        print(resolution + 'all thread tasks done')
    except Exception as e:
        print(f'Exception occurs, error = {e}')
        raise e
    finally:
        ave_fps = round(test.show_times / COST_TIME['image_show'][0],1)
        test.show_times = 0
        COST_TIME['image_show'][0] = 0
        threads_done.clear()
        test.test_stop()
    print(resolution + 'camera_fps_test done')
    return ave_fps, test.camera.fps


def main():
    ave_fps_test()


if __name__ == '__main__':
    main()
