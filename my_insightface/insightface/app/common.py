from typing import NamedTuple

import numpy as np
from numpy.linalg import norm as l2norm

# from easydict import EasyDict
__all__ = ['Face', 'RawTarget','Target']

from database.milvus_standalone.common import MatchInfo
from my_insightface.insightface.app.sort_plus import KalmanBoxTracker


class Face(dict):
    def __init__(self, d=None, **kwargs):
        """

        :param d:
        :param kwargs:
        """
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            # 把k作为self的属性，并且v作为该属性的值，等效于self.k=v
            setattr(self, k, v)
        # Class attributes
        # for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property
    def face_location(self):
        return self.bbox if self.bbox else None

    @property
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'


class RawTarget(NamedTuple):
    id: int
    bbox: np.ndarray[4]
    kps: np.ndarray[5, 2]
    score: float = 0.0


class Target:
    def __init__(self, id: int, bbox: np.ndarray,
                 screen_scale: tuple[int, int, int, int], kps: np.ndarray, score: float = 0.0):

        self._hit_streak = 0  # frames of keeping existing in screen
        self._time_since_update = 0  # frames of keeping missing in screen
        self.id = id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.kps = kps
        self.score = score
        self.match_info: MatchInfo = MatchInfo(face_id=-1, name='', score=0.0)
        self.frames_since_reced: int = 0
        self._screen_scale = screen_scale  # [x1, y1, x2, y2]
        self._tracker: KalmanBoxTracker = KalmanBoxTracker(bbox)
        self.normed_embedding: np.ndarray[512] = np.zeros(512)

    def update_pos(self, bbox: np.ndarray, kps: np.ndarray, score: float):
        self.bbox = bbox
        self.kps = kps
        self.score = score

    def update_tracker(self, detect_tar: RawTarget):
        self._time_since_update = 0
        self._tracker.update(detect_tar.bbox)

    def old_enough(self, max_age: int) -> bool:
        return self._time_since_update > max_age

    def in_screen(self, min_hits: int) -> bool:
        return self._time_since_update < 1 and self._hit_streak >= min_hits

    def set_match_info(self, match_info: MatchInfo):
        self.match_info = match_info

    @property
    def get_raw_target(self) -> RawTarget:
        # get predicted bounding box from Kalman Filter
        if self._tracker is None:
            raise ValueError('tracker is None')
        bbox = self._tracker.predict()[0]
        # if failed to update before predicted bbox, reset the hit_streak
        # coming after the update_tracker is meaning that the target is  in screen continuously
        if self._time_since_update == 0:
            self._hit_streak += 1
        else:
            self._hit_streak = 0
        self._time_since_update += 1
        return RawTarget(id=self.id, bbox=bbox, kps=self.kps, score=self.score)

    @property
    def screen_height(self):
        return self._screen_scale[3] - self._screen_scale[1]

    @property
    def screen_width(self):
        return self._screen_scale[2] - self._screen_scale[0]

    @property
    def bbox_width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def name(self):
        if self.if_matched and self.match_info.name:
            return self.match_info.name
        else:
            return f'target[{self.id}]'

    @property
    def time_satified(self) -> bool:
        if not self.if_matched:
            return False
        elif self.frames_since_reced < 100:
            self.frames_since_reced += 1
            return False
        else:
            print(self.frames_since_reced)
            self.frames_since_reced = 0
            return True

    @property
    def scale_satified(self) -> bool:
        target_area = self.bbox_width * self.bbox_height
        screen_area = self.screen_height * self.screen_width
        return (target_area / screen_area) > 0.03

    @property
    def if_matched(self) -> bool:
        return self.match_info.face_id != -1

    @property
    def rec_satified(self) -> bool:
        if self.scale_satified and not self.if_matched and self.in_screen(3):
            return True
        elif self.if_matched and self.scale_satified and self.time_satified and self.in_screen(3):
            return True
        else:
            return False

    @property
    def colors(self):
        red = (0, 0, 255)
        yellow = (50, 205, 255)
        green = (152, 251, 152)
        lavender = (238, 130, 238)
        blue = (127, 0, 0)
        if self.if_matched:
            # 有匹配对象
            if self.match_info.score > 0.4:
                bbox_color = green
                name_color = green
            else:
                # 有匹配对象，但是匹配分数不够，定义为匹配失败的红色
                bbox_color = red
                name_color = red
        else:  # 还没有匹配到对象
            bbox_color = yellow
            name_color = yellow
        return bbox_color, name_color
