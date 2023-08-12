import functools
import json
import pprint
import timeit
from pathlib import Path

import numpy as np

root = '.\\test_results'


def ave_fps_test(result_filename: str = 'ave_fps_test.json', test_func=None):
    """
    available resolutions:
    800x600
    1024x768
    1600x1200
    1920x1080
    2048x1536
    2560x1440
    2592x1944
    """
    if Path(result_filename).exists():
        with open(result_filename, 'r') as f:
            camera_test = json.load(f)
    else:
        camera_test = {}

    # resolution = input('\ninput resolution: ').strip().split('x')
    resolution = "1920x1080".split('x')
    # fps = int(input('input fps: '))
    fps = 30
    fps, resolution = test_func(resolution=resolution, fps=fps)
    # set,actual,measure
    key = f'{resolution[0]}x{resolution[1]}'
    camera_test[key] = fps

    with open(result_filename, 'w') as f:
        json.dump(camera_test, f)
    print('set,actually get,measure camera_test:')
    pprint.pprint(camera_test)


def ave_operation_test(test_func):
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        file_path = Path(root) / f'{test_func.__name__}.json'
        if file_path.exists() and file_path.is_file():
            with open(file_path, 'r') as f:
                operation_test = json.load(f)
        else:
            operation_test = {}
        test_time = int(input('\ninput test times: ').strip())
        ret = None
        cost_times = []
        for i in range(test_time):
            print(f'testing {test_func.__name__} in {i}th time...')
            start_time = timeit.default_timer()
            ret = test_func(*args, **kwargs)
            end_time = timeit.default_timer()
            cost_times.append(end_time - start_time)
        if isinstance(ret, str):
            operation = test_func.__name__ +' ' + ret
        else:
            operation = test_func.__name__

        operation_test[operation] = (test_time, round(np.mean(cost_times), 3))
        with open(file_path, 'w') as f:
            json.dump(operation_test, f)
        print(f'test {test_func.__name__} {test_time} times done:')
        pprint.pprint(operation_test)

    return wrapper
