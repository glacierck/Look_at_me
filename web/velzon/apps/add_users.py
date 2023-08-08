import json
from random import choice

from . import db
from .models import Role, User, user2leader_map

# 定义10个用户的电子邮件和用户名
users_data = [
    {'email': f'user{i}@example.com', 'username': f'User{i}', 'password': 'password'}
    for i in range(5)
]


def init_test_data():
    new_test_user()
    add_leader()
    # 调用该函数将用户信息导出为JSON
    export_users_to_json()
    print("Test data initialized successfully!")


def new_test_user():
    from random import choice

    # 随机选择角色名
    role_names = ['Student', 'Counselor', 'Administrator']
    # 创建用户
    for user_data in users_data:
        existing_user = User.query.filter_by(email=user_data['email']).first()
        if existing_user is not None:
            continue
        role_name = choice(role_names)
        role = Role.query.filter_by(name=role_name).first()
        user = User(email=user_data['email'],
                    username=user_data['username'],
                    role=[role])  # 设置角色
        user.password = user_data['password']
        db.session.add(user)
    # 保存更改
    db.session.commit()
    print("Users created successfully!")


def add_leader():
    # 随机选择用户作为领导人
    leaders = User.query.all()

    for user in users_data:
        leader = choice(leaders)
        user_record = User.query.filter_by(email=user['email']).first()
        predefined_user2leader_mappings = [{'user_id': user_record.id, 'leader_id': leader.id}]
        db.session.execute(user2leader_map.insert(), predefined_user2leader_mappings)

    # 保存更改
    db.session.commit()
    print("Association tables initialized successfully!")


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
            'role': [role.name for role in user.role],
            'leader': [leader.username for leader in user.leader],
            'followers': [follower.username for follower in user.followers]
        }
        users_list.append(user_info)

    # 将用户信息保存为JSON文件
    with open('users.json', 'w', encoding='utf-8') as f:
        json.dump(users_list, f, ensure_ascii=False, indent=4)

    print("Users exported to users.json successfully!")