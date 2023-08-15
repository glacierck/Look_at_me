from flask_login import current_user

from .side_bar_apps import apps
from .component import components
from .layout import layouts
from .page import pages
from .dashboard import dashboards
from .auth.views import authentication
from .profile.views import profile

__all__ = ['apps', 'components', 'layouts', 'pages', 'dashboards', 'authentication', 'profile']

from ..models import Permission


@dashboards.app_context_processor
def inject_permissions():
    subjectsData = {
        "Math": 95,
        "English": 85,
        "Physics": 98,
        "Chemistry": 90,
        "Biology": 85,
        "History": 65,
        "Geography": 75

    }
    return {'Permission': Permission, 'current_user': current_user, 'subjectsData': subjectsData}
