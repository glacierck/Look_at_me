from threading import Thread
from flask import render_template, current_app
from flask_mail import Message, Mail
# from ...main import app

app = current_app
app.config.update(
    Mail_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_MAX_EMAILS=20,
    MAIL_USERNAME='zhouge1831@gmail.com',
    MAIL_PASSWORD='zfemanabknfwttsi'
)
# 密码是应用专用密码 不是邮箱密码
mail = Mail(app)


def send_async_email(app, msg):
    with app.app_context():
        mail.send(msg)


def send_email(to, subject, template, **kwargs):
    #     app = current_app._get_current_object()
    msg = Message(app.config['FLASKY_MAIL_SUBJECT_PREFIX'] + ' ' + subject,
                  sender=app.config['FLASKY_MAIL_SENDER'], recipients=[to])
    msg.body = render_template(template + '.txt', **kwargs)
    msg.html = render_template(template + '.html', **kwargs)
    thr = Thread(target=send_async_email, args=[app, msg])
    thr.start()
    return thr
