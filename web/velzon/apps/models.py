import random
from datetime import datetime
from typing import NamedTuple
from pathlib import Path
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
user_role_map = db.Table(
    "user_roles",
    db.Column("user_id", db.Integer, db.ForeignKey("users.id")),
    db.Column("role_id", db.Integer, db.ForeignKey("roles.id")),
)
leaders_follower_map = db.Table(
    "leaders_followers",
    db.Column("leader_id", db.Integer, db.ForeignKey("users.id")),
    db.Column("follower_id", db.Integer, db.ForeignKey("users.id")),
)


class Activity(db.Model):
    __tablename__ = "passports"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    # -----------------web_show-----------------
    action = db.Column(db.Boolean, nullable=False, default=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    time = db.Column(db.Time, nullable=False, default=datetime.utcnow)
    # -----------------web_show-----------------


class Role(db.Model):
    __tablename__ = "roles"
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
            "Student": [Permission.VIEW_SELF_STATUS],
            "Counselor": [Permission.VIEW_SELF_STATUS, Permission.VIEW_STUDENT_STATUS],
            "Administrator": [
                Permission.VIEW_SELF_STATUS,
                Permission.VIEW_STUDENT_STATUS,
                Permission.VIEW_ALL_STATUS,
            ],
        }
        default_role = "Student"
        for role_name, perms in roles.items():
            role = Role.query.filter_by(name=role_name).first()
            if role is None:
                role = Role(name=role_name)
            role.reset_permissions()
            for perm in perms:
                role.add_permission(perm)
            role.default = role.name == default_role
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
        return "<Role %r>" % self.name


class User(db.Model, UserMixin):
    __tablename__ = "users"
    # ----------------------passport needed-----------------------------------
    face_rec_passport_state = db.Column(db.Integer, default=0)

    # 改变状态的时候记录一条数据
    activity = db.relationship("Activity", backref="user", lazy="dynamic")
    # 0: out school 1: in school
    position = db.Column(db.Boolean, default=False)

    # ----------------------passport needed-----------------------------------
    # ----------------------profile needed-----------------------------------
    avatar_path = db.Column(db.String(64), nullable=True, default="")
    first_name = db.Column(db.String(22), nullable=True, default="")
    last_name = db.Column(db.String(22), nullable=True, default="")
    phone = db.Column(db.String(11), nullable=True, default="")
    enrolled_data = db.Column(db.String(10), nullable=True, default="2021-09-01")

    # Portfolio
    qq = db.Column(db.String(20), nullable=True, default="")
    wechat = db.Column(db.String(20), nullable=True, default="")
    twitter = db.Column(db.String(20), nullable=True, default="")
    instagram = db.Column(db.String(20), nullable=True, default="")
    city = db.Column(db.String(10), nullable=True, default="Nanjing")
    country = db.Column(db.String(10), nullable=True, default="China")
    about_me = db.Column(db.Text(), nullable=True, default="I am a student in NJU")
    school = db.Column(db.String(20), nullable=True, default="Nanjing University")

    # ----------------------profile needed-----------------------------------

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
    roles = db.relationship(
        "Role", secondary=user_role_map, backref=db.backref("users")
    )
    # 因为关联表中都是user，他需要定义primaryjoin和secondaryjoin来指定关联的条件, lazy='dynamic'禁止自动执行查询
    leaders = db.relationship(
        "User",
        secondary=leaders_follower_map,
        primaryjoin=(leaders_follower_map.c.follower_id == id),
        secondaryjoin=(leaders_follower_map.c.leader_id == id),
        backref=db.backref("followers", lazy="dynamic"),
    )

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        # 反向引用创建的role
        if self.roles is None:
            if self.email == current_app.config["FLASKY_ADMIN"]:
                self.roles = [Role.query.filter_by(name="Administrator").first()]
            else:
                self.roles = [Role.query.filter_by(default=True).first()]

    # -------------------passport needed --------------------------
    @property
    def f_rec_passport_state(self):
        match self.face_rec_passport_state:
            case 0:
                return "Empty"
            case 1:
                return "Checking"
            case 2:
                return "Approved"

    # property装饰器将一个方法变成属性调用,但是有时候报错不会提示，结果异常需注意
    @property
    def recent_activity(self):
        # 返回最近10条记录
        index_with_recent_activity = (
            (
                str(i).zfill(2),
                {
                    "action": activity.action,
                    "date": activity.date.strftime("%Y-%m-%d"),
                    "time": activity.time.strftime("%H:%M:%S"),
                },
            )
            for i, activity in enumerate(
            self.activity.order_by(
                Activity.date.desc(), Activity.time.desc()
            ).limit(8),
            1,
        )
        )
        # recent_activity = []
        # for activity in  self.activity.order_by(Activity.timestamp.desc()).limit(5):
        #     recent_activity.append(activity)

        return index_with_recent_activity

    # -----profile needed-----------------------------------
    @property
    def avatar_file(self):
        """
        如果自己没有头像，那么就随机的从头像文件夹中选择一个头像
        Returns
        -------

        """
        path = Path(self.avatar_path)
        if path.exists() and path.is_file():
            return path.name
        else:
            default_avatars = list(
                Path(current_app.config["USER_AVATAR_DIR"])
                .joinpath("default")
                .glob("*.jpg")
            )
            return random.choice(default_avatars).name

    @property
    def profile_completion(self):
        count = 0
        # 定义你想要计数的属性列表
        attributes_to_count = [
            "face_rec_passport_state",
            "position",
            "avatar_path",
            "first_name",
            "last_name",
            "phone",
            "enrolled_data",
            "qq",
            "wechat",
            "twitter",
            "instagram",
            "city",
            "country",
            "about_me",
            "school",
        ]
        # 迭代属性列表并检查是否为None
        for key in attributes_to_count:
            if getattr(self, key):
                count += 1
        res = round((count / len(attributes_to_count)) * 100, 2)
        return res

    @property
    def hometown(self):
        return self.city + "," + self.country

    @property
    def full_name(self):
        if not self.first_name or not self.last_name:
            return self.username
        else:
            return self.first_name + self.last_name

    @property
    def profession(self):
        return self.username[:2]

    @property
    def identity(self):
        return self.roles[0].name

    # -----profile needed-----------------------------------
    @property
    def password(self):
        raise AttributeError("password is not a readable attribute")

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_confirmation_token(self, expiration=600):
        s = Serializer(current_app.config["SECRET_KEY"], str(expiration))
        return s.dumps({"confirm": str(self.id)})  # .decode('utf-8')

    @classmethod
    def get_user_from_token(cls, token):
        # ++gpt++
        s = Serializer(current_app.config["SECRET_KEY"], "600")
        try:
            data = s.loads(token)
        except SignatureExpired:
            return None
        temp = cls.query.filter_by(id=int(data.get("confirm"))).first()
        temp = cls.query.get(int(data.get("confirm")))
        return temp

    def confirm(self, token):
        s = Serializer(current_app.config["SECRET_KEY"], "600")
        try:
            data = s.loads(token)
        except SignatureExpired:
            return False
        if int(data.get("confirm")) != self.id:
            return False
        self.confirmed = True
        db.session.add(self)
        return True

    # -------------------login needed --------------------------
