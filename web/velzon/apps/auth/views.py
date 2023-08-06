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
        if user is not None and user.verify_password(form.password.data) and user.confirmed:
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
        flash('Account already confirmed.')
        return redirect(url_for('dashboards.index'))
    elif user.confirm(token):
        # confirm successfully
        db.session.commit()
        flash('You have confirmed your account. Thanks!')
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
    user = User.get_user_from_token(token)
    resend_form = RegistrationForm()
    if resend_form.validate_on_submit():
        # unique email
        try:
            resend_form.validate_email(resend_form.email)
        except ValidationError:
            flash('Email or Username already registered.')
            redirect(url_for('authentication.wait_for_confirmation'))

        assert user is not None, "user from get_user_from_token(token) is None"
        user.email = resend_form.email.data.lower()
        db.session.commit()
        token = user.generate_confirmation_token()
        send_email(user.email, 'Confirm Your Account',
                   'auth/email-to-verify', user=user, token=token)
        return redirect(url_for('authentication.wait_for_confirmation', token=token))

    return render_template('authentication/auth-wait-email-confirmation.html',
                           form=resend_form, user_email=user.email)  # 关键字传入到html作为变量{{value}}


@authentication.route('/authentication/auth-signup-cover', methods=['POST', 'GET'])
def register():
    from ..email import send_email
    register_form = RegistrationForm()
    # print(1,file=sys.stderr)
    if register_form.validate_on_submit():
        # unique email and username
        try:
            # print(2,file=sys.stderr)
            register_form.validate_email(register_form.email)
            register_form.validate_username(register_form.username)
        except ValidationError:
            # print(3,file=sys.stderr)
            flash('Email or Username already registered.')
            redirect(url_for('authentication.register'))
        # print(4,file=sys.stderr)
        user = User(email=register_form.email.data.lower(),
                    username=register_form.username.data,
                    password=register_form.password.data)
        # print(5,file=sys.stderr)
        db.session.add(user)
        db.session.commit()
        # print(6,file=sys.stderr)
        token = user.generate_confirmation_token()
        # print(7,file=sys.stderr)
        send_email(user.email, 'Confirm Your Account',
                   'auth/email-to-verify', user=user, token=token)
        flash('A confirmation email has been sent to you by email.')
        return redirect(url_for('authentication.wait_for_confirmation', token=token))
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
    return redirect(url_for('authentication.auth-logout-cover'))
