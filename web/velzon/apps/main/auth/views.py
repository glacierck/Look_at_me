from flask import Blueprint, render_template, request, redirect, url_for, flash, abort, current_app
from flask_login import login_user, logout_user, login_required
from ..auth.forms import LoginForm, RegistrationForm, ResendEmailForm
from ...models import User
from ... import db

# template_dir = Path(__file__).parent.parent / 'templates'

authentication = Blueprint('authentication', __name__, template_folder='templates',
                           static_folder='static')


@authentication.route('/account/login', methods=['POST', 'GET'])
def login():
    form = LoginForm()
    if form.is_submitted():
        user = User.query.filter_by(email=form.email.data.lower()).first()
        # only allow login if the user is confirmed and the password is correct
        if user is None:
            flash(f"Invalid User: '{form.email.data}'!")
        elif not user.confirmed:
            flash(f'Please sign up to confirm your account first!')
        elif not user.verify_password(form.password.data):
            flash(f'Invalid password: {form.password.data}!')
        else:
            login_user(user, form.remember_me.data)
            _next = request.args.get('next')
            if _next is None or not _next.startswith('/'):
                _next = url_for('dashboards.index')
            return redirect(_next)
    return render_template('authentication/auth-signin-cover.html', form=form)


# ------------------------------register------------------------------------------


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
        abort(404)
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
        abort(404)


# page of registration
# needed to modify
# page of email has been sent
@authentication.route('/authentication/wait-for-confirmation/<token>', methods=['POST', 'GET'])
def wait_for_confirmation(token):
    from ...email import send_email
    """
    we can resend the email here
    :return:
    """
    # html 的form中的action意味着提交表单提交到哪个页面
    user = User.get_user_from_token(token)
    assert user is not None, "user from get_user_from_token(token) is None"
    resend_form = ResendEmailForm()
    if resend_form.validate_on_submit():
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
    """
    use the given username to register
    Returns:
    """
    from ...email import send_email
    register_form = RegistrationForm()
    # resister_form.validate_on_submit() 会调用 is_submitted 和 validate_*** 函数
    if register_form.validate_on_submit():
        # unique email and username
        user = User.query.filter_by(username=register_form.username.data).first()
        if not user:
            user = User(email=register_form.email.data.lower(),
                        username=register_form.username.data,
                        password=register_form.password.data, )
            db.session.add(user)
        else:
            user.email = register_form.email.data.lower()
            user.password = register_form.password.data
        db.session.commit()
        token = user.generate_confirmation_token()
        # pycharm debug 模式下，send_email 会报错
        send_email(user.email, 'Confirm Your Account',
                   'auth/email-to-verify', user=user, token=token)
        # flash('A confirmation email has been sent to you by email.')
        return redirect(url_for('authentication.wait_for_confirmation', token=token))
    return render_template('authentication/auth-signup-cover.html', form=register_form)


# -----------------------------logout-------------------------------------


@authentication.route('/authentication/logout')
def logout():
    logout_user()
    return render_template('authentication/auth-logout-cover.html')
