from flask import Blueprint, render_template, url_for, redirect, flash, request, jsonify
from flask_login import login_required, current_user

from .forms import ProfileUpdateForm, PassportForm
from ...models import User, Activity

profile = Blueprint(
    "profile", __name__, template_folder="templates", static_folder="static"
)


@profile.route("/profile/<username>/details", methods=["GET", "POST"])
@login_required
def details(username):
    passport_form = PassportForm()
    if passport_form.validate_on_submit():
        passport_form.upload_passport(current_user)
        return redirect(url_for("profile.details", username=current_user.username))
    return render_template("pages/pages/pages-profile.html", form=passport_form)


@profile.route("/profile/<username>/settings", methods=["GET", "POST"])
@login_required
def settings(username):
    update_form = ProfileUpdateForm()
    action = request.form.get("action")
    if action and action == "update_profile" and update_form.validate_on_submit():
        update_form.update(current_user)
        return redirect(url_for("profile.details", username=current_user.username))
    elif action == "change_password" and update_form.validate_on_submit():
        update_form.change_password(current_user)
        return redirect(url_for("authentication.login"))
    elif action == "change_password" and not update_form.validate_on_submit():
        return redirect(
            url_for("profile.settings", username=current_user.username)
            + "#changePassword"
        )
    return render_template("pages/pages/pages-profile-settings.html", form=update_form)
