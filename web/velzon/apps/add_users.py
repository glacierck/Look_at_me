import json
import pprint
import random

from sqlalchemy import and_, not_

from . import db
from .models import Role, User

# 预定义一些可能的学科
subjects = ['ME', 'CS', 'EE']


def generate_predefined_users(num_students: int = 50, num_counselors: int = 50, num_admins: int = 1):
    _predefined_users = []

    # 生成学生
    for _ in range(num_students):
        subject = random.choice(subjects)
        roll = str(random.randint(10, 30))
        class_number = str(random.randint(1, 8)).zfill(1)
        student_number = str(random.randint(1, num_students)).zfill(2)
        username = f"{subject}{roll}{class_number}{student_number}"
        _predefined_users.append({'username': username, 'role': 'Student'})
    assert 1 <= num_counselors <= 99, "Number of counselors should be between 1 and 99!"
    # 生成领导者
    for _ in range(num_counselors):
        subject = random.choice(subjects)
        roll = str(random.randint(10, 30))
        counselor_number = str(random.randint(1, num_counselors)).zfill(2)
        username = f"{subject}{roll}X{counselor_number}"
        _predefined_users.append({'username': username, 'role': 'Counselor'})
    assert num_admins == 1, "Only one administrator is allowed!"
    # 生成管理员
    for _ in range(num_admins):
        username = "admin00X00"
        _predefined_users.append({'username': username, 'role': 'Administrator'})

    return _predefined_users




if __name__ == '__main__':
    # 调用函数生成预定义的用户
    predefined_users = generate_predefined_users()
    pprint.pprint(predefined_users)


def init_test_data(init_db: bool = False):
    # 调用函数生成预定义的用户
    if init_db:
        predefined_users = generate_predefined_users()
        create_predefined_users(predefined_users)
        create_relationship()
    # 调用该函数将用户信息导出为JSON
    export_users_to_json()
    print("Test data initialized successfully!")


def create_predefined_users(_predefined_users):

    # 创建用户
    for user_info in _predefined_users:
        existing_user = User.query.filter_by(username=user_info['username']).first()
        if existing_user is not None:
            continue
        role = Role.query.filter_by(name=user_info['role']).first()

        user = User(username=user_info['username'],
                    roles=[role])  # 设置角色
        db.session.add(user)
    # 保存更改
    db.session.commit()
    print("predefined Users created successfully!")


def create_relationship():
    # 所有用户除了admin自己都要被他领导
    users = User.query.filter(User.username != 'admin00X00').all()
    admin = User.query.filter_by(username='admin00X00').first()
    for user in users:
        user.leaders.append(admin)

    for subject in subjects:
        # find leader
        leaders = User.query.filter(User.username.like(f'{subject}__X%')).all()
        # find students
        students = User.query.filter(
            and_(
                User.username.like(f'{subject}___%'),
                not_(User.username.like(f'{subject}__X%'))
            )
        ).all()
        for student in students:
            for leader in leaders:
                student.leaders.append(leader)

    # 保存更改
    db.session.commit()
    print("create_relationship successfully!")


def export_users_to_json():
    # 查询所有用户
    users = User.query.all()

    # 创建一个包含所有用户信息的列表
    users_list = []
    for user in users:
        user_info = {
            'id': user.id,
            'email': user.email,
            'username': user.username,
            'role': [role.name for role in user.roles],
            'leader': [leader.username for leader in user.leaders],
            'followers': [follower.username for follower in user.followers]
        }
        users_list.append(user_info)

    # 将用户信息保存为JSON文件
    with open('users.json', 'w', encoding='utf-8') as f:
        json.dump(users_list, f, ensure_ascii=False, indent=4)

    print("Users exported to users.json successfully!")
