from datetime import datetime
from threading import Thread, Event
from queue import Queue
from time import sleep

import redis
from flask_socketio import SocketIO
from line_profiler_pycharm import profile
from werkzeug.serving import make_server

from my_insightface.insightface.app.identifier import matched_and_in_screen_queue
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
socketio = SocketIO(flask_app)
data = []


def update_data():
    while True:
        new_data = matched_and_in_screen_queue.get()
        for item in new_data:
            if item["ID"] not in [i["ID"] for i in data]:
                new_item = {"ID": item["ID"], "Name": f"New {item['Name']}", "Identity": "Student",
                            "Date": datetime.now().strftime("%d %b, %H:%M"), "Status": "Accessed"}
                data.append(new_item)
        for item in data:
            if item["ID"] not in [i["ID"] for i in new_data]:
                data.remove(item)
        socketio.emit('update_data', data)


@profile
def whole_fps_test(
        resolution: tuple[int, int], fps: int
) -> tuple[tuple[str], tuple[str]]:
    # auto_focus,manual_focus
    camera_params = {
        "app": "auto_focus",
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
        detect_thread = Thread(
            target=test.image2detect, args=(video_2_detect_queue, detect_2_rec_queue)
        )
        identify_thread = Thread(
            target=test.detect2identify, args=(detect_2_rec_queue, rec_2_screen_queue)
        )
        screen_2_web_thread = Thread(target=test.image2web, args=(rec_2_screen_queue,))

        socketio.start_background_task(update_data)
        flask_thread = Thread(target=socketio.run, args=(flask_app,),
                              kwargs={"allow_unsafe_werkzeug": True, "debug": False})

        video_read_thread.start()
        detect_thread.start()
        identify_thread.start()
        screen_2_web_thread.start()
        flask_thread.start()

        sleep(180)
        print("sleep 180s")
        socketio.stop()
        threads_done.set()
        streaming_event.clear()
        print("shutdown flask_server")
        video_read_thread.join()
        detect_thread.join()
        identify_thread.join()
        screen_2_web_thread.join()
        print("all thread tasks done")
    except Exception as e:
        print(f"Exception occurs, error = {e}")
        raise e
    finally:
        socketio.stop()
        threads_done.set()
        streaming_event.clear()
        ave_fps = round(test.show_times / COST_TIME["image2web"][0], 1)
        if test.camera.params["fps"][0] == test.camera.params["fps"][1]:
            res_fps = (test.camera.params["fps"][0], ave_fps)
        else:
            res_fps = (*test.camera.params["fps"], ave_fps)
        test.test_stop()
    print("camera_fps_test done")
    return res_fps, test.camera.params["resolution"]


def main():
    ave_fps_test("whole_fps_test.json", whole_fps_test)


if __name__ == "__main__":
    main()
