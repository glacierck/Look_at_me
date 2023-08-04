import numpy as np

from performance_test import ave_operation_test

"""
光是创建一个tracker就要花费大约0.1s，这个速度太慢了，不适合实时应用
"""
@ave_operation_test
def creat_trackers():
    from my_insightface.insightface.app.detector import ObjectTracker
    from my_insightface.insightface.data import LightImage
    # Generate a random color image of size 2000x2500 pixels
    # Each pixel has 3 values (Red, Green, Blue), each value is an integer from 0 to 255
    image = np.random.randint(0, 256, (2000, 2500, 3), dtype=np.uint8)
    img = LightImage(nd_arr=image)
    bbox = [0, 0, 100, 100]
    tracker = ObjectTracker(img, bbox)
    return tracker.tracker_name


from my_insightface.insightface.app.detector import ObjectTracker


@ave_operation_test
def update_tracker(tracker: ObjectTracker):
    from my_insightface.insightface.data import LightImage
    # Generate a random color image of size 2000x2500 pixels
    # Each pixel has 3 values (Red, Green, Blue), each value is an integer from 0 to 255
    image = np.random.randint(0, 256, (2000, 2500, 3), dtype=np.uint8)
    img = LightImage(nd_arr=image)
    tracker.update(img)
    return tracker.tracker_name


def update():
    from my_insightface.insightface.app.multi_thread_analysis import ObjectTracker
    from my_insightface.insightface.data import LightImage
    # Generate a random color image of size 2000x2500 pixels
    # Each pixel has 3 values (Red, Green, Blue), each value is an integer from 0 to 255
    image = np.random.randint(0, 256, (2000, 2500, 3), dtype=np.uint8)
    img = LightImage(nd_arr=image)
    bbox = [0, 0, 100, 100]
    tracker = ObjectTracker(img, bbox)
    update_tracker(tracker)

def main():
    update()
if __name__ == '__main__':
    main()