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
# 保了数据库层面的引用完整性，并允许数据库执行某些优化和约束检查
user_role_map = db.Table('user_roles',
                         db.Column('user_id', db.Integer, db.ForeignKey('users.id')),
                         db.Column('role_id', db.Integer, db.ForeignKey('roles.id'))
                         )
leaders_follower_map = db.Table('leaders_followers',
                                db.Column('leader_id', db.Integer, db.ForeignKey('users.id')),
                                db.Column('follower_id', db.Integer, db.ForeignKey('users.id'))
                                )


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
    # 邮箱 认证用户，方便后续找回密码
    email = db.Column(db.String(100), unique=True, index=True, nullable=True)
    # 密码
    password_hash = db.Column(db.String(128))

    # 给定
    # student
    # ME        21           5     14
    # leader roll in year   class  number
    # leader
    # ME        21           X     01
    # administrator
    # admin     00           X     00
    username = db.Column(db.String(100), unique=True, index=True, nullable=False)
    confirmed = db.Column(db.Boolean, default=False)

    # backref 可以让role直接访问user
    roles = db.relationship('Role', secondary=user_role_map, backref=db.backref('users'))
    # 因为关联表中都是user，他需要定义primaryjoin和secondaryjoin来指定关联的条件, lazy='dynamic'禁止自动执行查询
    leaders = db.relationship('User', secondary=leaders_follower_map,
                              primaryjoin=(leaders_follower_map.c.follower_id == id),
                              secondaryjoin=(leaders_follower_map.c.leader_id == id),
                              backref=db.backref('followers', lazy='dynamic'))

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        # 反向引用创建的role
        if self.roles is None:
            if self.email == current_app.config['FLASKY_ADMIN']:
                self.roles = [Role.query.filter_by(name='Administrator').first()]
            else:
                self.roles = [Role.query.filter_by(default=True).first()]

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
