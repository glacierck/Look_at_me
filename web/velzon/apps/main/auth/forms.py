from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Length, Email, Regexp
from wtforms import ValidationError
from ...models import User

__all__ = ['LoginForm', 'RegistrationForm']


def validate_email(field):
    user = User.query.filter_by(email=field.data.lower()).first()
    if user and user.confirmed:
        raise ValidationError('Email already registered.')


def validate_username(field):
    user = User.query.filter_by(username=field.data).first()
    if user and user.confirmed:
        raise ValidationError('Username already in use.')


class LoginForm(FlaskForm):
    # form的字段名与<input>的name属性值相同
    # validators 是一些验证条件，DataRequired-》必填
    email = StringField('Email', default="User@example.com", validators=[DataRequired(), Length(1, 64),
                                                                         Email()])

    password = PasswordField('Password', default="", validators=[DataRequired()])
    remember_me = BooleanField('Remember')
    submit = SubmitField('Login In')


class RegistrationForm(FlaskForm):
    email = StringField('Email', default="User@example.com", validators=[DataRequired(), Length(1, 64),
                                                                         Email()])
    # Regexp 正则表达式验证
    username = StringField('Username', default="", validators=[
        DataRequired(), Length(1, 64),
        Regexp('^[A-Za-z][A-Za-z0-9_.]*$', 0,
               'Usernames must have only letters, numbers, dots or '
               'underscores')])
    password = PasswordField('Password', default="", validators=[DataRequired()])
    submit = SubmitField('Register')

    def validate_email(self, field):
        validate_email(field)

    def validate_username(self, field):
        validate_username(field)


class ResendEmailForm(FlaskForm):
    email = StringField('Email', default="", validators=[DataRequired(), Length(1, 64),
                                                         Email()])
    submit = SubmitField('Register')

    def validate_email(self, field):
        validate_email(field)
