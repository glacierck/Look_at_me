import time

import cv2
from flask import Blueprint, render_template, current_app, Response
from flask_login import login_required, current_user

from my_insightface.insightface.app.multi_thread_analysis import streaming_event, image2web_queue, threads_done

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


def generate():
    while True:
        # print("image2web_queue.qsize() = ", image2web_queue.qsize())
        # print("streaming_event.is_set() = ", streaming_event.is_set())
        if streaming_event.is_set() and not image2web_queue.empty():  # 只有在有客户端请求时才获取图像
            frame = image2web_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # time.sleep(0.1)  # 控制获取速度
        # elif image2web_queue.empty():
        #     print("image2web_queue is empty")


@dashboards.route("/video_feed")
@login_required
def video_feed():
    streaming_event.set()  # 客户端开始请求，设置事件
    response = Response(
        generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    response.call_on_close(streaming_event.clear)  # 客户端停止请求，重置事件
    return response


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
