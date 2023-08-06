from flask import current_app
from flask_login import UserMixin
from . import db
from itsdangerous import URLSafeTimedSerializer as Serializer, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, index=True)
    # 密码
    password_hash = db.Column(db.String(128))
    username = db.Column(db.String(100), unique=True, index=True)
    # role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
    confirmed = db.Column(db.Boolean, default=False)

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
        return s.dumps({'confirm': str(self.id)})#.decode('utf-8')

    @classmethod
    def get_user_from_token(cls, token):
        # ++gpt++
        s = Serializer(current_app.config['SECRET_KEY'],"600")
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
