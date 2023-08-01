from ..app.common import Face

__all__ = ['flatten_list', 'get_digits', 'get_nodigits']


# 展开嵌套列表,最底层元素是Face或者Image对象
def flatten_list(nested_list):
    from ..data.image import Image
    try:
        assert not isinstance(nested_list, Face) or isinstance(nested_list, Image)
        for sublist in nested_list:
            for element in flatten_list(sublist):
                yield element
    except (TypeError, AssertionError):
        yield nested_list


def get_digits(s: str) -> str:
    return ''.join(c for c in s if c.isdigit())


def get_nodigits(s: str) -> str:
    return ''.join(c for c in s if not c.isdigit())
