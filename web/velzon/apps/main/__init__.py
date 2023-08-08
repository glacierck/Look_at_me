from .side_bar_apps import apps
from .component import components
from .layout import layouts
from .page import pages
from .dashboard import dashboards
from ..auth.views import authentication
__all__ = ['apps', 'components', 'layouts', 'pages', 'dashboards', 'authentication']

from ..models import Permission


@dashboards.app_context_processor
def inject_permissions():
    return dict(Permission=Permission)