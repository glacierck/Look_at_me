import os
from datetime import datetime
from threading import Thread
from queue import Queue
from time import sleep

import redis
from flask_socketio import SocketIO
from gunicorn.app.base import BaseApplication
from line_profiler_pycharm import profile
from my_insightface.insightface.app.identifier import matched_and_in_screen_deque
from performance_test import ave_fps_test
from my_insightface.insightface.app.multi_thread_analysis import (
    MultiThreadFaceAnalysis,
    COST_TIME, threads_done, streaming_event
)
from web.velzon.apps import create_app

video_2_detect_queue = Queue(maxsize=400)
detect_2_rec_queue = Queue(maxsize=200)
rec_2_screen_queue = Queue(maxsize=400)

flask_app = create_app()
redis_url = 'redis://localhost:6379/0'
socketio = SocketIO(flask_app, message_queue=redis_url)

data = []
class GunicornApp(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

def update_data():
    print("socketio.start_background_task: update_data start")
    global socketio,data
    streaming_event.set()
    while True:
        new_data = matched_and_in_screen_deque.get()
        print("new_data = ", new_data)
        print("data = ", data)
        new_items = []
        deleted_items = []
        for item in new_data:
            if item["ID"] not in [i["ID"] for i in data]:
                new_item = {"ID": item["ID"], "Name": f"New {item['Name']}", "Identity": "Student",
                            "Date": datetime.now().strftime("%d %b, %H:%M"), "Status": "Accessed"}
                new_items.append(new_item)
                data.append(new_item)
        for item in data:
            if item["ID"] not in [i["ID"] for i in new_data]:
                data.remove(item)
                deleted_items.append(item)

        update_info = {"deleted": deleted_items, "added": new_items}
        print("update_info = ", update_info)
        socketio.emit('update_table', update_info)




@socketio.on('connect')
def handle_connection():
    print("socketio.on('connect')")

    # socketio.start_background_task(update_data)

def run_server():
    print("start running GunicornApp")
    options = {
        'bind': '%s:%s' % ('127.0.0.1', '5000'),
        'workers': 2, # 设置工作进程数
        'threads': 4, # 每个工作进程的线程数
        'worker_class': 'eventlet'
    }
    GunicornApp(flask_app, options).run()





@profile
def whole_fps_test(
        resolution: tuple[int, int], fps: int
) -> tuple[tuple[str], tuple[str]]:
    global socketio
    # auto_focus,manual_focus
    camera_params = {
        "app": "laptop",
        "approach": "usb",
        "fps": fps,
        "resolution": resolution,
    }
    identifier_params = {"server_refresh": True, "npz_refresh": True}
    test = MultiThreadFaceAnalysis(
        test_folder="test_02",
        camera_params=camera_params,
        identifier_params=identifier_params,
    )

    try:
        video_read_thread = Thread(target=test.video_read, args=(video_2_detect_queue,))
        video_read_thread.daemon = True
        detect_thread = Thread(
            target=test.image2detect, args=(video_2_detect_queue, detect_2_rec_queue)
        )
        detect_thread.daemon = True
        identify_thread = Thread(
            target=test.detect2identify, args=(detect_2_rec_queue, rec_2_screen_queue)
        )
        screen_2_web_thread = Thread(target=test.image2web, args=(rec_2_screen_queue,))

        background_task_1 = Thread(target=update_data)

        video_read_thread.start()
        detect_thread.start()
        identify_thread.start()
        screen_2_web_thread.start()
        background_task_1.start()
        run_server()
        # socketio.run(flask_app, debug=False)

        # flask_thread.start()

        # flask_thread.start()
        # start_background_task_1.start()
        # start_background_task_2.start()
        sleep(180)
        test.test_stop()
        os._exit(0)
        print("sleep 20s")
        threads_done.set()
        print("set threads_done")
        streaming_event.clear()
        print("clear streaming_event")
        # video_read_thread.join()
        # detect_thread.join()
        # identify_thread.join()
        # screen_2_web_thread.join()
        print("all thread tasks done")
    except Exception as e:
        print(f"Exception occurs, error = {e}")
        raise e
    finally:
        threads_done.set()
        streaming_event.clear()
        image2web_redis = redis.Redis(host='localhost', port=6379)
        image2web_redis.flushall()
        # ave_fps = round(test.show_times / COST_TIME["image2web"][0], 1)
        # if test.camera.params["fps"][0] == test.camera.params["fps"][1]:
        #     res_fps = (test.camera.params["fps"][0], ave_fps)
        # else:
        #     res_fps = (*test.camera.params["fps"], ave_fps)

    print("camera_fps_test done")
    return "", test.camera.params["resolution"]


def main():
    ave_fps_test("tests/whole_fps_test.json", whole_fps_test)


if __name__ == "__main__":
    main()
