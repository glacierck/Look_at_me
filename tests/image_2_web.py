import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue
from time import sleep

from flask_socketio import SocketIO
from line_profiler_pycharm import profile

from my_insightface.insightface.app.identifier import matched_and_in_screen_deque
from my_insightface.insightface.app.multi_thread_analysis import (
    MultiThreadFaceAnalysis,
    threads_done, streaming_event, image2web_deque
)
from web.velzon.apps import create_app

video_2_detect_queue = Queue(maxsize=400)
detect_2_rec_queue = Queue(maxsize=200)
rec_2_screen_queue = Queue(maxsize=400)
flask_app = create_app()
socketio = SocketIO(flask_app, async_mode='gevent')
data = []


@profile
def update_data():
    print("socketio.start_background_task: update_data start")
    global socketio, data
    max_time = 0.01
    while streaming_event.is_set():
        socketio.sleep(max_time)
        try:
            new_data = matched_and_in_screen_deque.popleft()
        except IndexError:
            # 也要有打印空的时间,精确到sec
            print(datetime.now().strftime("%d %b, %H:%M:%S"), "matched_and_in_screen_queue is empty")
        else:
            # new_data = [{"ID": random.randint(0, 100), "Name": f"New {random.randint(0, 100)}"} for i in range(10)]
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
            # print(datetime.now().strftime("%d %b, %H:%M:%S"), "update_info = ", update_info)
            if any(update_info):
                socketio.emit('update_table', update_info)
            else:
                print("update_info is empty time: ", datetime.now().strftime("%d %b, %H:%M:%S"))


@profile
def update_image():
    print("socketio.start_background_task: update_image start")
    max_time = 0.006
    while streaming_event.is_set():
        # frame_tuple = image2web_redis.brpop("frames", timeout=1)
        socketio.sleep(max_time)
        try:
            frame_bytes = image2web_deque.popleft()
        except IndexError:
            print(datetime.now().strftime("%d %b, %H:%M:%S"), "image2web_deque is empty")
        else:
            print(datetime.now().strftime("%d %b, %H:%M:%S"), "got frame_bytes")
            frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")
            socketio.emit('video_frame', frame_base64)


@socketio.on('client_ready')
def handle_client_ready():
    print("socketio.on('client_ready')")
    streaming_event.set()
    socketio.start_background_task(update_image)
    socketio.start_background_task(update_data)


@socketio.on('connect')
def handle_connection():
    print("socketio.on('connect')")


def task_done_callback(future):
    print(f"\nTask completed. Result: {future.result()}")


@profile
def main():
    camera_params = {
        "app": "ip_webcam",
        "approach": "usb",
        "fps": 30,
        "resolution": (1280, 720),
    }
    identifier_params = {"server_refresh": True, "npz_refresh": True}
    with ThreadPoolExecutor(max_workers=12) as executor:
        test = MultiThreadFaceAnalysis(
            test_folder="test_02",
            camera_params=camera_params,
            identifier_params=identifier_params,
        )
        try:
            # 提交任务到线程池
            future1 = executor.submit(test.video_read, video_2_detect_queue)
            future2 = executor.submit(test.image2detect, video_2_detect_queue, detect_2_rec_queue)
            future3 = executor.submit(test.detect2identify, detect_2_rec_queue, rec_2_screen_queue)
            future4 = executor.submit(test.image2web, rec_2_screen_queue)
            future5 = executor.submit(socketio.run, flask_app, debug=False, port=8088, use_reloader=False)
            # 为每个任务添加回调
            future1.add_done_callback(task_done_callback)
            future2.add_done_callback(task_done_callback)
            future3.add_done_callback(task_done_callback)
            future4.add_done_callback(task_done_callback)
            future5.add_done_callback(task_done_callback)

            # 等待所有任务完成
            sleep(45)
        except SystemExit as e:
            print("exception:", e)
        finally:
            print("sleep 45s")
            threads_done.set()
            print("threads_done.set()")
            streaming_event.clear()
            print("streaming_event.clear()")
            test.test_stop()
            socketio.stop()
            print("socketio.stop()")
            print("all  done")


if __name__ == "__main__":
    main()
