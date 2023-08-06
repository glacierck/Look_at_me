from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    app.debug = True
    app.config['SECRET_KEY'] = b'_5#y2L"F4Q8z\n\xec]/'
    app.config['WTF_CSRF_ENABLED'] = True
    app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

    app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db.sqllite"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    from .models import User
    with app.app_context():
        db.create_all()

    login_manager = LoginManager(app)
    login_manager.session_protection = 'strong'
    login_manager.login_view = 'authentication.login'  # set the login view

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from .main import dashboards, apps, layouts, pages, components,authentication

    app.register_blueprint(dashboards, url_prefix="/")
    app.register_blueprint(apps, url_prefix="/")
    app.register_blueprint(layouts, url_prefix="/")
    app.register_blueprint(pages, url_prefix="/")
    app.register_blueprint(components, url_prefix="/")
    app.register_blueprint(authentication, url_prefix="/")
    return app
