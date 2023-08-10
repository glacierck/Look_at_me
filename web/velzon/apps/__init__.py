from pathlib import Path

from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template
from flask_login import LoginManager

db = SQLAlchemy()

db_dir = Path(__file__).absolute().parents[3] / 'database/sqlite3/db.sqlite'


def create_app():
    app = Flask(__name__)
    app.debug = True
    app.config['SECRET_KEY'] = b'_5#y2L"F4Q8z\n\xec]/'
    app.config['WTF_CSRF_ENABLED'] = True
    app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

    app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///" + db_dir.as_posix()
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['FLASKY_ADMIN'] = '1831768457@qq.com'
    db.init_app(app)
    from .models import Role
    from .add_users import init_test_data
    with app.app_context():
        db.create_all()
        Role.insert_roles()
        init_test_data(True)

    login_manager = LoginManager(app)
    login_manager.session_protection = 'strong'
    login_manager.login_view = 'authentication.login'  # set the login view
    from .models import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # 请求未知页面或者路由
    @app.errorhandler(404)
    def _404_alt(error):
        return render_template('authentication/auth-404-alt.html'), 404

    # 有触发的异常没有处理
    @app.errorhandler(500)
    def auth_500(error):
        return render_template('pages/authentication/auth-500.html'), 500

    from .main import dashboards, apps, layouts, pages, components, authentication, profile

    app.register_blueprint(dashboards, url_prefix="/")
    app.register_blueprint(apps, url_prefix="/")
    app.register_blueprint(layouts, url_prefix="/")
    app.register_blueprint(pages, url_prefix="/")
    app.register_blueprint(components, url_prefix="/")
    app.register_blueprint(authentication, url_prefix="/")
    app.register_blueprint(profile, url_prefix="/")
    return app
