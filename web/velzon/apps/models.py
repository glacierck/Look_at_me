from typing import NamedTuple

from flask import current_app
from flask_login import UserMixin

from itsdangerous import URLSafeTimedSerializer as Serializer, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash

from . import db


class Permission(NamedTuple):
    VIEW_SELF_STATUS = 0x01
    VIEW_STUDENT_STATUS = 0x02
    VIEW_ALL_STATUS = 0x04


# db.ForeignKey('users.id') 外键约束
user2role_map = db.Table('user2role',
                         db.Column('user_id', db.Integer, db.ForeignKey('users.id')),
                         db.Column('role_name', db.String(64), db.ForeignKey('roles.name'))
                         )
user2leader_map = db.Table('user2leader',
                           db.Column('user_id', db.Integer, db.ForeignKey('users.id')),
                           db.Column('leader_id', db.Integer, db.ForeignKey('users.id'))
                           )


def initialize_test_data():
    predefined_user2role_mappings = [
        {'user_id': 1, 'role_name': 'Student'},
        {'user_id': 2, 'role_name': 'Administrator'},
    ]

    predefined_user2leader_mappings = [
        {'user_id': 1, 'leader_id': 2},
        # ...
    ]

    # 使用数据库会话来插入预定义的映射关系
    db.session.execute(user2role_map.insert(), predefined_user2role_mappings)
    db.session.execute(user2leader_map.insert(), predefined_user2leader_mappings)
    db.session.commit()

    print("Association tables initialized successfully!")


# 初始化函数


class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    default = db.Column(db.Boolean, default=False, index=True)
    permissions = db.Column(db.Integer)

    def __init__(self, **kwargs):
        super(Role, self).__init__(**kwargs)
        if self.permissions is None:
            self.permissions = 0

    @staticmethod
    def insert_roles():
        roles = {
            'Student': [Permission.VIEW_SELF_STATUS],
            'Counselor': [Permission.VIEW_SELF_STATUS, Permission.VIEW_STUDENT_STATUS],
            'Administrator': [Permission.VIEW_SELF_STATUS, Permission.VIEW_STUDENT_STATUS, Permission.VIEW_ALL_STATUS],
        }
        default_role = 'Student'
        for role_name, perms in roles.items():
            role = Role.query.filter_by(name=role_name).first()
            if role is None:
                role = Role(name=role_name)
            role.reset_permissions()
            for perm in perms:
                role.add_permission(perm)
            role.default = (role.name == default_role)
            db.session.add(role)
        db.session.commit()
        print("Roles initialized successfully!")

    # 二进制的与运算对权限的操作
    def add_permission(self, perm):
        if not self.has_permission(perm):
            self.permissions += perm

    def remove_permission(self, perm):
        if self.has_permission(perm):
            self.permissions -= perm

    def reset_permissions(self):
        self.permissions = 0

    def has_permission(self, perm):
        return self.permissions & perm == perm

    def __repr__(self):
        return '<Role %r>' % self.name


class User(db.Model, UserMixin):
    __tablename__ = 'users'
    # -------------------login needed --------------------------
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, index=True, nullable=False)
    # 密码
    password_hash = db.Column(db.String(128))
    username = db.Column(db.String(100), unique=True, index=True, nullable=False)
    leader_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    confirmed = db.Column(db.Boolean, default=False)

    role = db.relationship('Role', secondary=user2role_map)  # 单向引用创建的role
    leader = db.relationship('User', secondary=user2leader_map,
                             primaryjoin=(user2leader_map.c.user_id == id),
                             secondaryjoin=(user2leader_map.c.leader_id == id),
                             backref=db.backref('followers', lazy='dynamic'))

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        # 反向引用创建的role
        if self.role is None:
            if self.email == current_app.config['FLASKY_ADMIN']:
                self.role = Role.query.filter_by(name='Administrator').first()
            else:
                self.role = Role.query.filter_by(default=True).first()

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_confirmation_token(self, expiration=600):
        s = Serializer(current_app.config['SECRET_KEY'], str(expiration))
        return s.dumps({'confirm': str(self.id)})  # .decode('utf-8')

    @classmethod
    def get_user_from_token(cls, token):
        # ++gpt++
        s = Serializer(current_app.config['SECRET_KEY'], "600")
        try:
            data = s.loads(token)
        except SignatureExpired:
            return None
        temp = cls.query.filter_by(id=int(data.get('confirm'))).first()
        temp = cls.query.get(int(data.get('confirm')))
        return temp

    def confirm(self, token):
        s = Serializer(current_app.config['SECRET_KEY'], "600")
        try:
            data = s.loads(token)
        except SignatureExpired:
            return False
        if int(data.get('confirm')) != self.id:
            return False
        self.confirmed = True
        db.session.add(self)
        return True
    # -------------------login needed --------------------------
