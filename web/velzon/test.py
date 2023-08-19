import pprint
import random
from datetime import datetime
from flask_socketio import SocketIO
from apps import create_app

# 初始化20个数据
data = [
    {"ID": f"VZ21{i:02}", "Name": f"User {i}", "Identity": "Student", "Date": datetime.now().strftime("%d %b, %H:%M"),
     "Status": "Paid"} for i in range(10)]

app = create_app()
socketio = SocketIO(app)

# 学科与其缩写的映射
SUBJECTS_ABBREVIATIONS = {
    "Math": "MA",
    "English": "EN",
    "Physics": "PH",
    "Chemistry": "CH",
    "Biology": "BI",
    "History": "HI",
    "Geography": "GE",
    "Chinese": "CN",
    "Literature": "LT",
    "Politics": "PO",
    "Music": "MU",
    "Art": "AR",
    "PE": "PE",
    "Computer": "CP",
    "Design": "DS"

}

# 学科的人数统计
subjects_count = {subject: 0 for subject in SUBJECTS_ABBREVIATIONS.keys()}

# 这是更新名片的函数，名片格式不准变
def update_data():
    global subjects_count
    # 每1秒随机删除一些数据并添加新数据
    while True:
        socketio.sleep(1)
        new_items= []
        deleted_items = []
        for _ in range(random.randint(1, 5)):
            # 随机删除一个学生
            if len(data) <2:
                break
            deleted_item = data.pop(random.randint(0, len(data) - 1))
            deleted_items.append(deleted_item)
        for _ in range(random.randint(1, 5)):
            # 随机选择一个学科作为学生的学科
            subject_name = random.choice(list(SUBJECTS_ABBREVIATIONS.keys()))
            # 更新学科的人数
            subjects_count[subject_name] += 1
            # 创建学生的ID
            student_id = f"{SUBJECTS_ABBREVIATIONS[subject_name]}21{random.randint(1, 9)}{random.randint(0, 99)}"
            new_item = {
                "ID": student_id,
                "Name": student_id,
                "Identity": "Student",
                "Date": datetime.now().strftime("%d %b, %H:%M"),
                "Status": "Paid"
            }
            new_items.append(new_item)
            data.append(new_item)
        update_info = {"deleted": deleted_items, "added": new_items}
        socketio.emit('update_table', update_info)



    # 这是图表函数
def background_task():
    global subjects_count

    while True:
        # 获取当前学科人数的副本，以避免在循环中修改原始数据
        current_subjects_count = subjects_count.copy()

        # 遍历每个学科，并更新在学校的人数
        for subject in current_subjects_count.keys():
            # 随机决定是增加还是减少人数
            change = random.choice([-1, 1])
            current_subjects_count[subject] = max(0, current_subjects_count[subject] + change)

        # 找到发生变化的学科，并发送这些数据
        data_to_send = {subject: count for subject, count in current_subjects_count.items() if count > subjects_count[subject]}
        socketio.emit('update_chart', data_to_send)
        subjects_count = current_subjects_count  # 更新全局变量

        socketio.sleep(1)


# 用于定时发送实时数据
# 这是字段更新的函数
def send_data():
    global subjects_count
    while True:
        # 这里的数据可以来自数据库或其他实时源
        in_school = sum(subjects_count.values())
        total = 50
        real_time_status = {
            'total': total,
            'in_school': in_school,  # 更新在学校的学生人数
            'out_school': total - in_school,
            'percentage': round(in_school / total * 100, 2)
        }
        socketio.emit('update_field', real_time_status)
        socketio.sleep(1)  # 每1秒发送一次


@socketio.on('connect')
def handle_connection():
    socketio.start_background_task(background_task)
    socketio.start_background_task(update_data)
    socketio.start_background_task(send_data)

def main():
    # 启动更新数据线程
    socketio.run(app, port=8088, debug=True)


if __name__ == '__main__':
    main()
