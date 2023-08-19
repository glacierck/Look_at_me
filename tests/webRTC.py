import base64
import os
from datetime import datetime
from threading import Thread
from queue import Queue
from time import sleep

import redis
import requests
from flask_socketio import SocketIO
from line_profiler_pycharm import profile
from my_insightface.insightface.app.identifier import matched_and_in_screen_deque
from performance_test import ave_fps_test
from my_insightface.insightface.app.multi_thread_analysis import (
    MultiThreadFaceAnalysis,
    COST_TIME, threads_done, streaming_event
)
from web.velzon.apps import create_app
import eventlet
video_2_detect_queue = Queue(maxsize=400)
detect_2_rec_queue = Queue(maxsize=200)
rec_2_screen_queue = Queue(maxsize=400)

flask_app = create_app()
socketio = SocketIO(flask_app,async_mode="eventlet")
data = []

def update_data():
    print("socketio.start_background_task: update_data start")
    while True:
        new_data = matched_and_in_screen_deque.get()
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
        if all(update_info):
            socketio.emit('update_table', update_info)
        socketio.sleep(0.01)

def update_image():
    print("socketio.start_background_task: update_image start")

    image2web_redis = redis.Redis(host='localhost', port=6379)
    while True:
        frame_bytes = image2web_redis.rpop("frames")
        if frame_bytes:
            frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")
            socketio.emit('video_frame', frame_base64)
        socketio.sleep(0.01)
def run_server():
    socketio.run(flask_app,port=8080, debug=False)
@socketio.on('connect')
def handle_connection():
    streaming_event.set()
    print("Client connected")
    socketio.start_background_task(update_data)
    socketio.start_background_task(update_image)


@profile
def whole_fps_test(
        resolution: tuple[int, int], fps: int
) -> tuple[tuple[str], tuple[str]]:
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


        identify_thread.daemon = True
        screen_2_web_thread.daemon=True
        # flask_thread.daemon = True


        video_read_thread.start()
        detect_thread.start()
        identify_thread.start()
        screen_2_web_thread.start()

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

        # ave_fps = round(test.show_times / COST_TIME["image2web"][0], 1)
        # if test.camera.params["fps"][0] == test.camera.params["fps"][1]:
        #     res_fps = (test.camera.params["fps"][0], ave_fps)
        # else:
        #     res_fps = (*test.camera.params["fps"], ave_fps)

    print("camera_fps_test done")
    return "", test.camera.params["resolution"]


def main():
    ave_fps_test("whole_fps_test.json", whole_fps_test)


if __name__ == "__main__":
    main()
