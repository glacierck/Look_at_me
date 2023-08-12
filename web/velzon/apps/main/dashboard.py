from flask import Blueprint, render_template, current_app
from flask_login import login_required, current_user

dashboards = Blueprint(
    "dashboards",
    __name__,
    template_folder="templates",
    static_folder="static",
)


@dashboards.route("/")
@login_required
def index():
    user = current_user
    return render_template("dashboards/index.html", username=user.username)


@dashboards.route("/dashboard-analytics/")
@login_required
def dashboard_analytics():
    return render_template("dashboards/dashboard-analytics.html")


@dashboards.route("/dashboard-crm/")
@login_required
def dashboard_crm():
    return render_template("dashboards/dashboard-crm.html")


# @app.route("/video_feed")
# def video_feed():
#     streaming_event.set()  # 客户端开始请求，设置事件
#     response = Response(
#         generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
#     )
#     response.call_on_close(streaming_event.clear)  # 客户端停止请求，重置事件
#     return response


@dashboards.route("/dashboard-crypto/")
@login_required
def dashboard_crypto():
    return render_template("dashboards/dashboard-crypto.html")


@dashboards.route("/dashboard-projects/")
@login_required
def dashboard_projects():
    return render_template("dashboards/dashboard-projects.html")


@dashboards.route("/dashboard-nft/")
@login_required
def dashboard_nft():
    return render_template("dashboards/dashboard-nft.html")


@dashboards.route("/dashboard-job/")
@login_required
def dashboard_job():
    return render_template("dashboards/dashboard-job.html")
