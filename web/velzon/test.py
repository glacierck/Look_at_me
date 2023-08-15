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
        socketio.sleep(60)
        deleted_index = random.randint(0, len(data) - 1)
        deleted_item = data.pop(deleted_index)
        new_item = {"ID": deleted_item["ID"], "Name": f"New {deleted_item['Name']}", "Identity": "Student",
                    "Date": datetime.now().strftime("%d %b, %H:%M"), "Status": "Paid"}
        data.append(new_item)
        pprint.pprint(data)
        socketio.emit('update_data', data)


def background_task():
    subjectsData = {
        "Math": 80,
        "English": 85,
        "Physics": 90,
        "Chemistry": 88,
        "Biology": 87,
        "History": 78,
        "Geography": 82,
        "Chinese": 90,
        "Literature": 85,
        "Politics": 88,
        "Music": 87,
        "Art": 78,
        "PE": 82,
        "Computer": 90,
        "Design": 85
    }

    while True:
        # 随机选择一个学科
        subject = random.choice(list(subjectsData.keys()))

        # 随机决定是增加还是减少分数
        change = random.choice([-3, 3])
        subjectsData[subject] = max(60, min(100, subjectsData[subject] + change))

        # 只发送变化的数据
        data_to_send = {subject: subjectsData[subject]}
        socketio.emit('update_chart', data_to_send)
        socketio.sleep(1)


@socketio.on('connect')
def handle_connection():
    socketio.start_background_task(background_task)


def main():
    # 启动更新数据线程
    socketio.start_background_task(update_data)
    socketio.run(app, port=8080, debug=True)


if __name__ == '__main__':
    main()
