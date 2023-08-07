import sys
from pathlib import Path

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from wtforms import ValidationError

from ..auth.forms import LoginForm, RegistrationForm
from ..models import User
from .. import db

# template_dir = Path(__file__).parent.parent / 'templates'

authentication = Blueprint('authentication', __name__, template_folder='templates',
                           static_folder='static')


@authentication.route('/account/login', methods=['POST', 'GET'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower()).first()
        # only allow login if the user is confirmed and the password is correct
        if user is not None and user.verify_password(form.password.data):
            login_user(user, form.remember_me.data)
            _next = request.args.get('next')
            if _next is None or not _next.startswith('/'):
                _next = url_for('dashboards.index')
            return redirect(_next)
        flash('Invalid email or password.')
    return render_template('authentication/auth-signin-cover.html', form=form)


# ------------------------------register------------------------------------------


@authentication.route('/_404_alt')
def _404_alt():
    return render_template('authentication/auth-404-alt.html')


# success page after email confirmation
# 按钮导向主页
@authentication.route('/authentication/auth-success-msg-cover')
def auth_success_msg_cover():
    return render_template('authentication/auth-success-msg-cover.html')


@authentication.route('/confirm/<token>')
def confirm(token):
    """
    click  the link in the email to confirm the account
    :param token:
    :return:
    """
    # according to token, get user
    user = User.get_user_from_token(token)
    if user is None:
        # Invalid token
        flash('Invalid or expired token.')
        return redirect(url_for('authentication._404_alt'))
    elif user.confirmed:
        # Already confirmed
        return redirect(url_for('dashboards.index'))
    elif user.confirm(token):
        # confirm successfully
        db.session.commit()
        # flash('You have confirmed your account. Thanks!')
        login_user(user, remember=True)
        return redirect(url_for('authentication.auth_success_msg_cover'))
    else:
        # something wrong
        return redirect(url_for('authentication._404_alt'))


# page of registration
# needed to modify
# page of email has been sent
@authentication.route('/authentication/wait-for-confirmation/<token>', methods=['POST', 'GET'])
def wait_for_confirmation(token):
    from ..email import send_email
    """
    we can resend the email here
    :return:
    """
    # html 的form中的action意味着提交表单提交到哪个页面
    user = User.get_user_from_token(token)
    assert user is not None, "user from get_user_from_token(token) is None"
    resend_form = RegistrationForm()
    if resend_form.is_submitted():
        if user.email != resend_form.email.data.lower():
            user.email = resend_form.email.data.lower()
            db.session.commit()
        token = user.generate_confirmation_token()
        send_email(user.email, 'Confirm Your Account',
                   'auth/email-to-verify', user=user, token=token)
        return redirect(url_for('authentication.wait_for_confirmation', token=token))
    return render_template('authentication/auth-wait-email-confirmation.html',
                           form=resend_form, user_email=user.email, token=token)  # 关键字传入到html作为变量{{value}}


@authentication.route('/authentication/auth-signup-cover', methods=['POST', 'GET'])
def register():
    from ..email import send_email
    register_form = RegistrationForm()
    # print(1,file=sys.stderr)
    # resister_form.validate_on_submit() 会调用 is_submitted 和 validate_*** 函数
    if register_form.validate_on_submit():
        # unique email and username
        user = User.query.filter_by(email=register_form.email.data.lower()).first()
        user_2 = User.query.filter_by(username=register_form.username.data).first()
        assert user == user_2, "user and user_2 are not the same"
        if not user and not user_2:
            user = User(email=register_form.email.data.lower(),
                        username=register_form.username.data,
                        password=register_form.password.data)
            db.session.add(user)
            db.session.commit()
        token = user.generate_confirmation_token()
        # pycharm debug 模式下，send_email 会报错
        send_email(user.email, 'Confirm Your Account',
                   'auth/email-to-verify', user=user, token=token)
        # flash('A confirmation email has been sent to you by email.')
        return redirect(url_for('authentication.wait_for_confirmation', token=token))
    elif register_form.email.data or register_form.username.data:
        flash('Email or username has been registered.')
    return render_template('authentication/auth-signup-cover.html', form=register_form)


# -----------------------------logout-------------------------------------
@authentication.route('/authentication/auth-logout-cover')
@login_required
def auth_logout_cover():
    return render_template('authentication/auth-logout-cover.html')


@authentication.route('/account/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('authentication.auth_logout_cover'))
