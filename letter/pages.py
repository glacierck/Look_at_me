from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from wtforms import ValidationError

from ..auth.forms import LoginForm, RegistrationForm
from ..models import User
from .. import db
from web.velzon.apps.email import send_email

pages = Blueprint('pages', __name__, template_folder='templates',
                  static_folder='static', )


# Pages page
@pages.route('/pages/starter')
@login_required
def starter():
    return render_template('pages/pages/pages-starter.html')


@pages.route('/pages/profile')
@login_required
def profile():
    return render_template('pages/pages/pages-profile.html')


@pages.route('/pages/settings')
@login_required
def profile_settings():
    return render_template('pages/pages/pages-profile-settings.html')


@pages.route('/pages/team')
@login_required
def team():
    return render_template('pages/pages/pages-team.html')


@pages.route('/pages/timeline')
@login_required
def timeline():
    return render_template('pages/pages/pages-timeline.html')


@pages.route('/pages/faqs')
@login_required
def faqs():
    return render_template('pages/pages/pages-faqs.html')


@pages.route('/pages/pricing')
@login_required
def pricing():
    return render_template('pages/pages/pages-pricing.html')


@pages.route('/pages/gallery')
@login_required
def gallery():
    return render_template('pages/pages/pages-gallery.html')


@pages.route('/pages/maintenance')
@login_required
def maintenance():
    return render_template('pages/pages/pages-maintenance.html')


@pages.route('/pages/comingsoon')
@login_required
def comingsoon():
    return render_template('pages/pages/pages-comingsoon.html')


@pages.route('/pages/sitemap')
@login_required
def sitemap():
    return render_template('pages/pages/pages-sitemap.html')


@pages.route('/pages/search_results')
@login_required
def search_results():
    return render_template('pages/pages/pages-search-results.html')


@pages.route('/pages/privacy_policy')
@login_required
def privacy_policy():
    return render_template('pages/pages/pages-privacy-policy.html')


@pages.route('/pages/term_conditions')
@login_required
def term_conditions():
    return render_template('pages/pages/pages-term-conditions.html')


# Landing page
@pages.route('/landing')
@login_required
def landing():
    return render_template('pages/pages/pages-landing.html')


@pages.route('/nft-landing')
@login_required
def nft_landing():
    return render_template('pages/pages/pages-nft-landing.html')


@pages.route('/job-landing')
@login_required
def job_landing():
    return render_template('pages/pages/pages-job-landing.html')


# Authentication Pages
@pages.route('/authentication/auth-signin-basic')
@login_required
def auth_signin_basic():
    return render_template('pages/authentication/auth-signin-basic.html')


@pages.route('/authentication/auth-signup-basic')
@login_required
def auth_signup_basic():
    return render_template('pages/authentication/auth-signup-basic.html')


@pages.route('/authentication/auth-pass-reset-basic')
@login_required
def auth_pass_reset_basic():
    return render_template('pages/authentication/auth-pass-reset-basic.html')


@pages.route('/authentication/auth-pass-reset-cover')
def auth_pass_reset_cover():
    return render_template('pages/authentication/auth-pass-reset-cover.html')


@pages.route('/authentication/auth-pass-change-basic')
@login_required
def auth_pass_change_basic():
    return render_template('pages/authentication/auth-pass-change-basic.html')


@pages.route('/authentication/auth-pass-change-cover')
@login_required
def auth_pass_change_cover():
    return render_template('pages/authentication/auth-pass-change-cover.html')


@pages.route('/authentication/auth-lockscreen-basic')
@login_required
def auth_lockscreen_basic():
    return render_template('pages/authentication/auth-lockscreen-basic.html')


@pages.route('/authentication/auth-lockscreen-cover')
@login_required
def auth_lockscreen_cover():
    return render_template('pages/authentication/auth-lockscreen-cover.html')


@pages.route('/authentication/auth-logout-basic')
@login_required
def auth_logout_basic():
    return render_template('pages/authentication/auth-logout-basic.html')


@pages.route('/authentication/auth-success-msg-basic')
@login_required
def auth_success_msg_basic():
    return render_template('pages/authentication/auth-success-msg-basic.html')


@pages.route('/authentication/auth-twostep-basic')
@login_required
def auth_twostep_basic():
    return render_template('pages/authentication/auth-twostep-basic.html')


@pages.route('/authentication/auth-twostep-cover')
@login_required
def auth_twostep_cover():
    return render_template('pages/authentication/auth-twostep-cover.html')


@pages.route('/authentication/auth-404-basic')
@login_required
def auth_404_basic():
    return render_template('pages/authentication/auth-404-basic.html')


@pages.route('/authentication/auth-404-cover')
@login_required
def auth_404_cover():
    return render_template('pages/authentication/auth-404-cover.html')


@pages.route('/authentication/auth-500')
@login_required
def auth_500():
    return render_template('pages/authentication/auth-500.html')


@pages.route('/authentication/auth-offline')
@login_required
def auth_offline():
    return render_template('pages/authentication/auth-offline.html')


# Actual Auth pages(working)
# @pages.route('/account/login')
# def login():
#     return render_template('pages/account/login.html')
# @pages.route('/account/login')
# # @login_required
# def auth_signin_cover():
#     return render_template('pages/authentication/auth-signin-cover.html')


# @pages.route('/account/login', methods=['POST'])
# def login_post():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')
#         remember = True if request.form.get('remember') else False
#
#         user = User.query.filter_by(email=email).first()
#         print("sagar", user)
#
#         if not user or not check_password_hash(user.password, password):
#             flash("Invalid Credentials")
#             return redirect(url_for('pages.auth_signin_cover'))
#
#         login_user(user, remember=remember)
#         return redirect(url_for('dashboards.index'))
@pages.route('/account/login', methods=['POST', 'GET'])
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
    return render_template('pages/authentication/auth-signin-cover.html', form=form)


# @pages.route('/account/signup')
# def signup():
#     return render_template('pages/account/signup.html')


# @pages.route('/account/signup', methods=['POST'])
# def signup_post():
#     email = request.form.get('email')
#     username = request.form.get('username')
#     password = request.form.get('password')
#
#     user_email = User.query.filter_by(email=email).first()
#     user_username = User.query.filter_by(username=username).first()
#
#     if user_email:
#         flash("User email already Exists")
#         return redirect(url_for('pages.signup'))
#     if user_username:
#         flash("Username already Exists")
#         return redirect(url_for('pages.signup'))
#
#     new_user = User(email=email, username=username, password=generate_password_hash(password, method="sha256"))
#     db.session.add(new_user)
#     db.session.commit()
#
#     return redirect(url_for('pages.login'))
# marked
# ------------------------------register------------------------------------------


@pages.route('/_404_alt')
def _404_alt():
    return render_template('pages/authentication/auth-404-alt.html')


# success page after email confirmation
# 按钮导向主页
@pages.route('/authentication/auth-success-msg-cover')
def auth_success_msg_cover():
    return render_template('pages/authentication/auth-success-msg-cover.html')


@pages.route('/confirm/<token>')
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
        return redirect(url_for('page._404_alt'))
    elif user.confirmed:
        # Already confirmed
        flash('Account already confirmed.')
        return redirect(url_for('dashboards.index'))
    elif user.confirm(token):
        # confirm successfully
        db.session.commit()
        flash('You have confirmed your account. Thanks!')
        return redirect(url_for('pages.auth_success_msg_cover'))
    else:
        # something wrong
        return redirect(url_for('page._404_alt'))


# page of registration
# needed to modify
# page of email has been sent
@pages.route('/authentication/wait-for-confirmation/<token>', method=['POST', 'GET'])
def wait_for_confirmation(token):
    """
    we can resend the email here
    :return:
    """
    user = User.get_user_from_token(token)
    resend_form = RegistrationForm()
    if resend_form.validate_on_submit():
        # unique email
        try:
            resend_form.validate_email(resend_form.email.data)
        except ValidationError:
            flash('Email or Username already registered.')
            redirect(url_for('pages.wait_for_confirmation'))

        assert user is not None, "user from get_user_from_token(token) is None"
        user.email = resend_form.email.data.lower()
        db.session.commit()
        token = user.generate_confirmation_token()
        send_email(user.email, 'Confirm Your Account',
                   'auth/email-to-verify', user=user, token=token)
        return redirect(url_for(f'/authentication/wait-for-confirmation/{token}'))
    return render_template('pages/authentication/auth-wait-email-confirmation.html',
                           form=resend_form, user_email=user.email)  # 关键字传入到html作为变量{{value}}


@pages.route('/authentication/auth-signup-cover', methods=['POST', 'GET'])
def register():
    register_form = RegistrationForm()
    if register_form.validate_on_submit():
        # unique email and username
        try:
            register_form.validate_email(register_form.email.data)
            register_form.validate_username(register_form.username.data)
        except ValidationError:
            flash('Email or Username already registered.')
            redirect(url_for('pages.register'))

        user = User(email=register_form.email.data.lower(),
                    username=register_form.username.data,
                    password=register_form.password.data)
        db.session.add(user)
        db.session.commit()
        token = user.generate_confirmation_token()
        send_email(user.email, 'Confirm Your Account',
                   'auth/email-to-verify', user=user, token=token)
        flash('A confirmation email has been sent to you by email.')
        return redirect(url_for(f'/authentication/wait-for-confirmation/{token}'))
    return render_template('pages/authentication/auth-signup-cover.html', form=register_form)


# -----------------------------logout-------------------------------------
@pages.route('/authentication/auth-logout-cover')
@login_required
def auth_logout_cover():
    return render_template('pages/authentication/auth-logout-cover.html')


@pages.route('/account/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('pages.auth-logout-cover'))
