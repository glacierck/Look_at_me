from pathlib import Path

import cv2
import magic
import numpy as np
from flask import current_app
from flask_login import current_user
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from werkzeug.datastructures import FileStorage
from wtforms import StringField, TextAreaField, SubmitField, PasswordField
from wtforms.validators import (
    DataRequired,
    Length,
    EqualTo,
    ValidationError,
    Regexp,
    Optional,
)

from ... import db
from ...models import User


# 自定义 validators
class ImageFileAllowed:
    def __init__(self, allowed_mimes, message=None):
        self.allowed_mimes = allowed_mimes
        self.message = message

    def __call__(self, form, field):
        file = field.data
        mime = magic.from_buffer(file.read(1024), mime=True)
        file.seek(0)  # 重置文件指针，以便后续操作

        if mime not in self.allowed_mimes:
            raise ValidationError(self.message or "File type is not allowed.")


class ProfileUpdateForm(FlaskForm):
    # ------------------profile details------------------
    avatar = FileField(
        "Avatar",
        validators=[
            Optional(),
            ImageFileAllowed(
                allowed_mimes=[
                    "image/jpeg",
                    "image/png",
                    "image/bmp",
                    "image/tiff",
                    "image/webp",
                    "image/svg+xml",
                ],
                message="Image files only",
            ),
        ],
    )
    first_name = StringField(
        "First Name", default="", validators=[Optional(), Length(0, 22)]
    )
    last_name = StringField(
        "Last Name", default="", validators=[Optional(), Length(0, 22)]
    )
    phone = StringField("Phone", default="", validators=[DataRequired(), Length(0, 22)])
    qq = StringField("QQ", default="", validators=[Optional(), Length(0, 20)])
    wechat = StringField("Wechat", default="", validators=[Optional(), Length(0, 20)])
    twitter = StringField("Twitter", default="", validators=[Optional(), Length(0, 20)])
    instagram = StringField(
        "Instagram", default="", validators=[Optional(), Length(0, 22)]
    )
    city = StringField("City", default="", validators=[Optional(), Length(0, 22)])
    zip_code = StringField(
        "Zip Code", default="", validators=[Optional(), Length(0, 22)]
    )
    country = StringField("Country", default="", validators=[Optional(), Length(0, 22)])
    about_me = TextAreaField(
        "About Me", default="", validators=[Optional(), Length(0, 500)]
    )
    school = StringField("School", default="", validators=[Optional(), Length(0, 22)])
    submit = SubmitField("Update")
    # ------------------profile details------------------

    # ------------------reset password------------------
    old_password = PasswordField("Old Password", default="", validators=[Optional()])
    new_password = PasswordField(
        "New Password",
        default="",
        validators=[
            Optional(),
            EqualTo("confirm_password", message="Passwords must match"),
            Regexp(
                r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,}$",
                message="password must have at least 8 characters, 1 uppercase letter, 1 lowercase letter and 1 number",
            ),
        ],
    )
    confirm_password = PasswordField(
        "Confirm Password", default="", validators=[Optional()]
    )

    def update(self, user):
        attr_list = [
            "first_name",
            "last_name",
            "phone",
            "qq",
            "wechat",
            "twitter",
            "instagram",
            "city",
            "country",
            "about_me",
            "school",
            "zip_code",
        ]
        for attr in attr_list:
            data = getattr(self, attr).data
            if data is not None and data != "":
                setattr(user, attr, data)
        # 处理头像上传
        avatar_file = self.avatar.data
        if avatar_file and isinstance(avatar_file, FileStorage):
            # 读取图像
            image_np = cv2.imdecode(
                np.fromstring(avatar_file.read(), np.uint8), cv2.IMREAD_COLOR
            )

            # 转换为JPG格式
            _, buffer = cv2.imencode(".jpg", image_np)
            jpg_image = buffer.tobytes()

            # 创建保存路径
            image_dir = Path(current_app.config["USER_AVATAR_DIR"])
            image_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            image_path = image_dir / f"{user.username}.jpg"

            # 保存图像文件
            with open(image_path, "wb") as f:
                f.write(jpg_image)
            # 更新用户头像路径
            user.avatar_path = image_path.as_posix()
        db.session.commit()
        return user

    def validate_old_password(self, password: PasswordField):
        if not current_user.verify_password(password.data):
            raise ValidationError("Old password is wrong!")

    def change_password(self, user: User):
        user.password = self.new_password.data
        db.session.commit()


class PassportForm(FlaskForm):
    # ------------------passport needed------------------
    face_recognition = FileField(
        "Face Image",
        validators=[
            Optional(),
            ImageFileAllowed(
                allowed_mimes=[
                    "image/jpeg",
                    "image/png",
                    "image/bmp",
                    "image/tiff",
                    "image/webp",
                    "image/svg+xml",
                ],
                message="Image files only",
            ),
        ],
    )
    submit = SubmitField("Update")

    def upload_passport(self, user: User):
        passport_file = self.face_recognition.data
        assert isinstance(passport_file, FileStorage) and passport_file, "No file"
        if passport_file and isinstance(passport_file, FileStorage):
            # 读取图像
            image_np = cv2.imdecode(
                np.fromstring(passport_file.read(), np.uint8), cv2.IMREAD_COLOR
            )

            # 转换为JPG格式
            _, buffer = cv2.imencode(".jpg", image_np)
            jpg_image = buffer.tobytes()

            # 创建保存路径
            image_dir = (
                    Path(current_app.config["USER_PASSPORT_DIR"]) / "face_recognition"
            )
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{user.id}-{user.full_name}.jpg"

            # 保存图像文件
            with open(image_path, "wb") as f:
                f.write(jpg_image)
            user.face_rec_passport_state = 1
            db.session.commit()
