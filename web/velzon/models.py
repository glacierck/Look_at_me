from flask_login import UserMixin
from . import db


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True,  index=True)
    # 密码
    password_hash = db.Column(db.String(128))
    username = db.Column(db.String(100), unique=True,index=True)
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
