from __future__ import annotations
from threading import Thread
from queue import Queue
from my_insightface.insightface.app.real_time_tracker import MultiThreadFaceAnalysis,COST_TIME

video_2_detect_queue = Queue(maxsize=400)
detect_2_rec_queue = Queue(maxsize=200)
rec_2_show_queue = Queue(maxsize=400)


def main():
    test_folder = 'test_04'
    test = MultiThreadFaceAnalysis(test_folder=test_folder,app='ip_webcam', approach='usb')
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
        print('ave_fps = ', test.show_times / COST_TIME['image_show'][0])
        test.test_stop()

if __name__ == '__main__':
    main()
