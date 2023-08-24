import base64
from collections import deque
from datetime import datetime

from flask_socketio import SocketIO
from line_profiler_pycharm import profile

from my_insightface.insightface.app.drawer import streaming_event, image2web_deque
from my_insightface.insightface.app.identifier import matched_and_in_screen_deque
from web.velzon.apps import create_app


class SocketIOApp(SocketIO):
    def __init__(self, rec_info_deque: deque, image2web_deque: deque, async_mode='gevent'):
        self.app = create_app()
        super().__init__(self.app, async_mode=async_mode)
        self._rec_info = []
        self._rec_info_deque = rec_info_deque
        self._image2web_deque = image2web_deque
        self._register_events()

    def _register_events(self):
        @self.on("client_ready")
        def handle_client_ready():
            print("socketio.on('client_ready')")
            streaming_event.set()
            self.start_background_task(self._update_image)
            self.start_background_task(self._update_rec_info)

        @self.on("connect")
        def handle_connection():
            print("socketio.on('connect')")

    def run(self, host=None, port=None, **kwargs):
        super().run(self.app, host=host, port=port, **kwargs)

    @profile
    def _update_rec_info(self):
        max_time = 0.01
        while streaming_event.is_set():
            self.sleep(max_time)
            try:
                new_data = self._rec_info_deque.popleft()
            except IndexError:
                # 也要有打印空的时间,精确到sec
                # print(datetime.now().strftime("%d %b, %H:%M:%S"), "matched_and_in_screen_queue is empty")
                continue
            else:
                # new_data = [{"ID": random.randint(0, 100), "Name": f"New {random.randint(0, 100)}"} for i in range(10)]
                new_items = []
                deleted_items = []
                for item in new_data:
                    if item["ID"] not in [i["ID"] for i in self._rec_info]:
                        new_item = {"ID": item["ID"], "Name": f"New {item['Name']}", "Identity": "Student",
                                    "Date": datetime.now().strftime("%d %b, %H:%M"), "Status": "Accessed"}
                        new_items.append(new_item)
                        self._rec_info.append(new_item)
                for item in self._rec_info:
                    if item["ID"] not in [i["ID"] for i in new_data]:
                        self._rec_info.remove(item)
                        deleted_items.append(item)

                update_info = {"deleted": deleted_items, "added": new_items}
                # print(datetime.now().strftime("%d %b, %H:%M:%S"), "update_info = ", update_info)
                if any(update_info):
                    self.emit('update_table', update_info)
                else:
                    print("update_info is empty time: ", datetime.now().strftime("%d %b, %H:%M:%S"))

    @profile
    def _update_image(self):
        print("socketio.start_background_task: update_image start")
        max_time = 0.006
        while streaming_event.is_set():
            # frame_tuple = image2web_redis.brpop("frames", timeout=1)
            self.sleep(max_time)
            try:
                frame_bytes = self._image2web_deque.popleft()
            except IndexError:
                # print(datetime.now().strftime("%d %b, %H:%M:%S"), "image2web_deque is empty")
                continue
            else:
                print(datetime.now().strftime("%d %b, %H:%M:%S"), "got frame_bytes")
                frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")
                self.emit('video_frame', frame_base64)


socketio_app = SocketIOApp(matched_and_in_screen_deque, image2web_deque)
