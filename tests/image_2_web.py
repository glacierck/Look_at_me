import base64
import os
import timeit
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Thread
from queue import Queue
from time import sleep

import redis
from flask_socketio import SocketIO
from line_profiler_pycharm import profile
from my_insightface.insightface.app.identifier import matched_and_in_screen_deque
from performance_test import ave_fps_test
from my_insightface.insightface.app.multi_thread_analysis import (
    MultiThreadFaceAnalysis,
    COST_TIME, threads_done, streaming_event, image2web_deque
)
from web.velzon.apps import create_app

video_2_detect_queue = Queue(maxsize=400)
detect_2_rec_queue = Queue(maxsize=200)
rec_2_screen_queue = Queue(maxsize=400)
flask_app = create_app()
socketio = SocketIO(flask_app)
data = []
@profile
def update_data():
    print("socketio.start_background_task: update_data start")
    global socketio,data
    while streaming_event.is_set():
        start = timeit.default_timer()
        try:
            new_data = matched_and_in_screen_deque.popleft()
        except IndexError:
            # 也要有打印空的时间,精确到sec
            print(datetime.now().strftime("%d %b, %H:%M:%S"),"matched_and_in_screen_queue is empty")
            continue
        end = timeit.default_timer()
        # new_data = [{"ID": random.randint(0, 100), "Name": f"New {random.randint(0, 100)}"} for i in range(10)]
        # frequency control
        max_time = 0.01
        sleep_time = max_time - (end - start) if (end - start) < max_time else 0
        socketio.sleep(sleep_time)
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
        socketio.emit('update_table', update_info)
@profile
def update_image():
    print("socketio.start_background_task: update_image start")
    # global image2web_redis

    cnt = 0
    print("connected to redis")
    start = timeit.default_timer()
    while streaming_event.is_set():
        # frame_tuple = image2web_redis.brpop("frames", timeout=1)
        try:
            frame_bytes = image2web_deque.popleft()
        except IndexError:
            print(datetime.now().strftime("%d %b, %H:%M:%S"), "image2web_deque is empty")
            continue
        else:
            print(datetime.now().strftime("%d %b, %H:%M:%S"), "got frame_bytes" )
            frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")
            socketio.emit('video_frame', frame_base64)
        socketio.sleep(0.01)
        cnt += 1
        print("cnt = ", cnt)
        if cnt == 1000:
            end = timeit.default_timer()
            print("fps = ", cnt / (end - start))
            # os._exit(0)


@socketio.on('client_ready')
def handle_client_ready():
    print("socketio.on('client_ready')")
    streaming_event.set()
    socketio.start_background_task(update_image)
    socketio.start_background_task(update_data)
    # image2web_redis.flushall()

@socketio.on('connect')
def handle_connection():
    print("socketio.on('connect')")


@profile
def whole_fps_test(
        resolution: tuple[int, int], fps: int
) -> tuple[tuple[str], tuple[str]]:
    global socketio
    # auto_focus,manual_focus
    camera_params = {
        "app": "ip_webcam",
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
    print("image2web_redis.flushall()")
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

        # background_task_1 = Thread(target=update_data)

        video_read_thread.start()
        detect_thread.start()
        identify_thread.start()
        screen_2_web_thread.start()
        socketio.run(flask_app, debug=False,port=8088)
        sleep(180)
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
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        os._exit(0)
    finally:
        threads_done.set()
        streaming_event.clear()
        test.test_stop()
        # ave_fps = round(test.show_times / COST_TIME["image2web"][0], 1)
        # if test.camera.params["fps"][0] == test.camera.params["fps"][1]:
        #     res_fps = (test.camera.params["fps"][0], ave_fps)
        # else:
        #     res_fps = (*test.camera.params["fps"], ave_fps)

    print("camera_fps_test done")
    return "", test.camera.params["resolution"]


def main():

    camera_params = {
        "app": "ip_webcam",
        "approach": "usb",
        "fps": 30,
        "resolution": (1920,1080),
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
            executor.submit(test.video_read, video_2_detect_queue)
            executor.submit(test.image2detect, video_2_detect_queue, detect_2_rec_queue)
            executor.submit(test.detect2identify, detect_2_rec_queue, rec_2_screen_queue)
            executor.submit(test.image2web, rec_2_screen_queue)
            executor.submit(socketio.run, flask_app, debug=False, port=8088)
            # 运行 Flask 服务器
            # socketio.run(flask_app, debug=False, port=8088)

            # 等待所有任务完成
            sleep(30)
            print("sleep 180s")
            print("all thread tasks done")
        except KeyboardInterrupt:
            print("KeyboardInterrupt")

        finally:
            threads_done.set()
            streaming_event.clear()
            test.test_stop()
            os._exit(0)
            socketio.stop()
            executor.shutdown()
        print("camera_fps_test done")
    return "", test.camera.params["resolution"]

if __name__ == "__main__":
    main()
    # ave_fps_test("whole_fps_test.json", whole_fps_test)

