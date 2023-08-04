from pathlib import Path

import cv2
from numpy import ndarray

from ..data import LightImage

__all__ = ['Detector']


# 性能和精确度太差
class ObjectTracker:
    def __init__(self, img2track: LightImage, bbox: ndarray[4]):
        """
        KCF,CSRT,,MOSSE
        :param img2track:LightImage
        :param bbox:ndarray[4]
        """
        self.tracker_name = 'CSRT'
        self._tracker = cv2.legacy.TrackerCSRT_create()
        self._tracker.init(img2track.nd_arr, tuple(bbox))

    def update(self, img2track: LightImage):
        success, bbox = self._tracker.update(img2track.nd_arr)
        if success:
            return bbox
        else:
            return None


class Detector:
    def __init__(self):
        from my_insightface.insightface.model_zoo.model_zoo import get_model
        root: Path = Path.cwd().parents[1].joinpath('models\\insightface\\det_2.5g.onnx')
        self.detector_model = get_model(root, providers=('CUDAExecutionProvider', 'CPUExecutionProvider'))
        prepare_params = {'ctx_id': 0,
                          'det_thresh': 0.5,
                          'input_size': (320, 320)}
        self.detector_model.prepare(**prepare_params)
        self._frame_count = 0
        self._trackers = []

    def __call__(self, img2detect: LightImage) -> LightImage:
        detect_params = {'max_num': 0, 'metric': 'default'}
        bboxes, kpss = self.detector_model.detect(img2detect.nd_arr, **detect_params)
        for i in range(bboxes.shape[0]):
            kps = kpss[i] if kpss is not None else None
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            face = [bbox, kps, det_score, None, None]
            img2detect.faces.append(face)
        return img2detect
