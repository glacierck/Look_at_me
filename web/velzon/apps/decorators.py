from functools import wraps
from flask import abort
from flask_login import current_user
from .models import Permission


def permission_required(permission):
    def decorator(func):
        @wraps(func)
        def decorated_function(*args, **kwargs):
            if not current_user.can(permission):
                abort(403)
            return func(*args, **kwargs)

        return decorated_function

    return decorator


def admin_required(func):
    @permission_required(Permission.VIEW_ALL_STATUS)
    def decorated_function(*args, **kwargs):
        return func(*args, **kwargs)
    return decorated_function


def counselor_required(func):
    @permission_required(Permission.VIEW_STUDENT_STATUS)
    def decorated_function(*args, **kwargs):
        return func(*args, **kwargs)
    return decorated_function
