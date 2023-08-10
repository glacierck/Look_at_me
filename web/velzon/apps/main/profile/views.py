from flask import Blueprint, render_template
from flask_login import login_required

from ...models import User

profile = Blueprint('profile', __name__, template_folder='templates',
                    static_folder='static')


@profile.route('/profile/<username>/details')
@login_required
def details(username):
    user = User.query.filter_by(username=username).first_or_404()
    return render_template('pages/pages/pages-profile.html', user=user)


@profile.route('/profile/<username>/settings')
@login_required
def settings(username):
    user = User.query.filter_by(username=username).first_or_404()
    return render_template('pages/pages/pages-profile-settings.html', user=user)
