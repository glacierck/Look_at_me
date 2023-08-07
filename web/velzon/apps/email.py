from threading import Thread
from flask import render_template, current_app
from flask_mail import Message, Mail

app = current_app._get_current_object()
with app.app_context():
    # 邮箱配置
    app.config.update(
        MAIL_SERVER='smtp.googlemail.com',
        MAIL_PORT=587,
        MAIL_USE_TLS=True,
        MAIL_MAX_EMAILS=500,
        MAIL_USERNAME='zhouge1831@gmail.com',
        MAIL_PASSWORD='zfemanabknfwttsi',
        FLASKY_MAIL_SUBJECT_PREFIX="Velzon",
        FLASKY_MAIL_SENDER="Velzon <zhouge1831@gmail.com>",
    )  # 密码是应用专用密码 不是邮箱密码
    mail = Mail(app)


def send_async_email(app, msg):
    with app.app_context():
        mail.send(msg)


def send_email(to, subject, template, **kwargs):
    app = current_app._get_current_object()
    msg = Message(app.config['FLASKY_MAIL_SUBJECT_PREFIX'] + ' ' + subject,
                  sender=app.config['FLASKY_MAIL_SENDER'], recipients=[to])
    msg.body = render_template(template + '.text', **kwargs)
    msg.html = render_template(template + '.html', **kwargs)
    thr = Thread(target=send_async_email, args=[app, msg])
    thr.start()
    return thr

# from threading import Thread
# from flask import render_template
# from flask_mail import Message, Mail


# class Mailer:
#     def __init__(self, app):
#         self.app = app
#         self.app.config.update(
#             Mail_SERVER='smtp.gmail.com',
#             MAIL_PORT=587,
#             MAIL_USE_TLS=True,
#             MAIL_MAX_EMAILS=20,
#             MAIL_USERNAME='zhouge1831@gmail.com',
#             MAIL_PASSWORD='zfemanabknfwttsi'
#         )  # 密码是应用专用密码 不是邮箱密码
#         self.mail = Mail(self.app)
#
#     def send_async_email(self, msg):
#         with self.app.app_context():
#             self.mail.send(msg)
#
#     def send_email(self, to, subject, template, **kwargs):
#         msg = Message(self.app.config['FLASKY_MAIL_SUBJECT_PREFIX'] + ' ' + subject,
#                       sender=self.app.config['FLASKY_MAIL_SENDER'], recipients=[to])
#         msg.body = render_template(template + '.txt', **kwargs)
#         msg.html = render_template(template + '.html', **kwargs)
#         thr = Thread(target=self.send_async_email, args=[msg])
#         thr.start()
#         return thr
