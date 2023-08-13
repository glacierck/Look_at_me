import pprint
import random
import threading
from datetime import datetime
from time import sleep

from flask_socketio import SocketIO

from apps import create_app

# 初始化20个数据
data = [
    {"ID": f"VZ21{i:02}", "Name": f"User {i}", "Identity": "Student", "Date": datetime.now().strftime("%d %b, %H:%M"),
     "Status": "Paid"} for i in range(20)]

app = create_app()
socketio = SocketIO(app)


def update_data():
    # 每5秒随机删除一些数据并添加新数据
    while True:
        socketio.sleep(5)
        deleted_index = random.randint(0, len(data) - 1)
        deleted_item = data.pop(deleted_index)
        new_item = {"ID": deleted_item["ID"], "Name": f"New {deleted_item['Name']}", "Identity": "Student",
                    "Date": datetime.now().strftime("%d %b, %H:%M"), "Status": "Paid"}
        data.append(new_item)
        pprint.pprint(data)
        socketio.emit('update_data', data)


def main():
    # 启动更新数据线程
    socketio.start_background_task(update_data)
    socketio.run(app, allow_unsafe_werkzeug=True, debug=True)


if __name__ == '__main__':
    main()
