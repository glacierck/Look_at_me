from __future__ import annotations

from ..app.common import Face



def removesuffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


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


def strip_digits(s: str) -> str:
    return ''.join(c for c in s if not c.isdigit())
