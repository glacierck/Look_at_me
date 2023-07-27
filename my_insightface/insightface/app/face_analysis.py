# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division, annotations

import itertools
from typing import AnyStr, List, Tuple, NamedTuple

from pathlib import Path

import numpy as np
import onnxruntime
import cv2
from ..app.common import Face
from ..model_zoo import model_zoo
from ..utils import ensure_available
from milvus_standalone.milvus_lite import Milvus
from ..data.image import Image

__all__ = ['FaceAnalysis']


class MatchInfo(NamedTuple):
    score: float
    face_id: int
    name: str


class FaceAnalysis:
    def __init__(self, names, root: Path, allowed_modules=None, **kwargs):
        """
        :param names: 具体模型文件的名称
        :param root: 模型文件的所在目录
        :param allowed_modules:
        :param kwargs:
        :var onnx_files: onnx_files路径列表
        """
        self.det_size = None
        self.det_thresh = None
        self.det_model = None
        self.registered_faces = []
        self._face_id = 0
        self.models = {}
        self.milvus = Milvus(base_dir='test_milvus') if kwargs.get('if_milvus', True) else None
        self.names = names
        self.root = root
        self.allowed_modules = allowed_modules
        self.kwargs = kwargs
        self._onnx_files = [ensure_available(name=name, root=root) for name in names]

    def _load_model(self):
        onnxruntime.set_default_logger_severity(3)  # 日志级别设置为3，只显示ERROR
        for onnx_file in self._onnx_files:
            model = model_zoo.get_model(onnx_file, **self.kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif self.allowed_modules is not None and model.taskname not in self.allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (
                    self.allowed_modules is None or model.taskname in self.allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        if 'detection' not in self.models:
            raise ValueError('detection model not found')
        self.det_model = self.models['detection']

    def prepare(self, ctx_id: int, det_thresh: float = 0.5, det_size: tuple[int, int] = (640, 640)):
        self._load_model()
        self.det_thresh = det_thresh
        if not isinstance(det_thresh, float) or det_thresh < 0 or det_thresh > 1:
            raise ValueError('det_thresh must be a float between 0 and 1')
        if not isinstance(det_size, tuple) and all([isinstance(x, int) for x in det_size]):
            raise ValueError('det_size must be tuple[int, int]')

        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def detect(self, target: Image, max_num=0, metric='default') -> Image:
        """
        初始化Image对象的bboxes和kpss属性
        :param image: Image对象
        :param max_num: 最大检测人脸数
        :return: Image对象
        """
        assert isinstance(target, Image), 'target must be Image object'
        assert target.nd_arr is not None, 'target.nd_arr must be not None'
        target.bboxes, target.kpss = self.det_model.detect(target.nd_arr, max_num=max_num, metric=metric)

        return target

    def get(self, rec_image: Image, **kwargs) -> Image:
        bboxes = rec_image.bboxes
        kpss = rec_image.kpss
        face_name = kwargs.get('face_name', 'Unknown')
        if bboxes.shape[0] == 0:
            print(f'failed to detect name:{rec_image.name}')
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score, name=face_name, id=self._face_id)
            self._face_id += 1
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue
                model.get(rec_image.nd_arr, face)
            rec_image.faces.append(face)
        return rec_image


class Milvus2Search:
    def __init__(self, **kwargs):
        self.milvus = Milvus()
        self.__image_root = Path(__file__).parents[1].absolute() / 'data' / 'images'
        self.kwargs = kwargs
        self.registered_faces: list[Face] = []

    def get_faces_from_npys(self, img_folder: str, test_folder: str) -> List[Face]:

        img_dir = self.__image_root / test_folder / img_folder
        id_npy = img_dir / 'id.npy'  # shape: (n,)
        name_npy = img_dir / 'name.npy'  # shape: (n,)
        kps_npy = img_dir / 'kps.npy'  # shape: (n, 5, 2)
        bbox_npy = img_dir / 'bbox.npy'  # shape: (n, 4)
        embedding_npy = img_dir / 'embedding.npy'  # shape: (n, 512)
        normed_embedding_npy = img_dir / 'normed_embedding.npy'  # shape: (n, 512)
        if not (
                id_npy.exists() and name_npy.exists() and embedding_npy.exists() and kps_npy.exists() and bbox_npy.exists()):
            print('npy files not exists')
            return []

        print('\nloading registered faces from npy files')
        ids = np.load(str(id_npy)).astype(np.int64)
        names = np.load(str(name_npy))
        kps = np.load(str(kps_npy))
        bboxes = np.load(str(bbox_npy))
        embeddings = np.load(str(embedding_npy)).astype(np.float32)
        assert len(ids) == len(names) == len(kps) == len(bboxes) == len(embeddings), \
            'npy files not match'
        faces = []
        for i in range(len(ids)):
            face = Face(id=ids[i], name=names[i], kps=kps[i],
                        bbox=bboxes[i], embedding=embeddings[i])
            faces.append(face)
        # milvus 相似度设置为IP用的是normed_embedding
        normed_embeddings = np.array([face.normed_embedding for face in faces], dtype=np.float32)
        np.save(str(normed_embedding_npy), normed_embeddings)
        if self.milvus.has_collection:
            assert len(ids) == len(names) == len(normed_embeddings), 'npy files not match'
            # self.milvus.insert_from_files(
            # file_paths=[str(id_npy), str(name_npy), str(normed_embedding_npy)])
            self.milvus.insert([[id for id in ids],
                                [name for name in names],
                                [embedding for embedding in normed_embeddings]
                                ])

        return faces

    def faces_to_npys(self, faces: List[Face], img_folder: str, test_folder: str) -> None:
        img_dir = self.__image_root / test_folder / img_folder
        id_npy = img_dir / 'id.npy'  # shape: (n,) , int 32
        name_npy = img_dir / 'name.npy'  # shape: (n,), varchar
        kps_npy = img_dir / 'kps.npy'  # shape: (n, 5, 2), float32
        bbox_npy = img_dir / 'bbox.npy'  # shape: (n, 4), float32
        embedding_npy = img_dir / 'embedding.npy'  # shape: (n, 512) , float32
        ids, names, kps, bboxes, embeddings = [], [], [], [], []
        for face in faces:
            ids.append(face.id)
            names.append(face.name)
            kps.append(face.kps)
            bboxes.append(face.bbox)
            embeddings.append(face.embedding)
        np.save(str(id_npy), ids)
        np.save(str(name_npy), names)
        np.save(str(kps_npy), kps)
        np.save(str(bbox_npy), bboxes)
        np.save(str(embedding_npy), embeddings)
        print('\nsave target faces to npy files done')

    def load_registered_faces(self, detector, extractor, img_folder: str, **kwargs) -> List[Face]:
        from ..utils.my_tools import flatten_list
        from ..data.image import Image, get_images
        test_folder = kwargs.get('test_folder', None)
        refresh = kwargs.get('refresh', False)
        if not refresh:
            self.registered_faces = self.get_faces_from_npys(img_folder, test_folder)
        if self.registered_faces:
            print(f'\nload  {len(self.registered_faces)} registered_faces done')
            if self.milvus and self.milvus.get_entity_num:
                print('\n load registered faces to milvus done')
            return self.registered_faces

        print('\nloading registered faces from images')
        # 根据给出的图片folder，加载所有的人脸图片，返回一个Face对象的列表
        images: list[Image] = get_images(img_folder=img_folder, **kwargs)
        assert len(images) > 0, 'No images found in ' + img_folder
        assert detector, 'detector is None'
        assert extractor, 'extractor is None'
        res = []
        for image in images:
            detector(image)
            extractor(image)
            res.append(image.faces)
        self.registered_faces = list(flatten_list(res))
        assert self.registered_faces, f'No faces found in folder: {img_folder} !'
        print(f'\nload  {len(self.registered_faces)} registered_faces done')

        self.faces_to_npys(self.registered_faces, img_folder, test_folder)
        return self.registered_faces

    @staticmethod
    def _set_match_info(cur_image: Image, match_infos: list[dict]) -> Image:
        # cur_image设置match_info
        for i, cur_face in enumerate(cur_image.faces):
            cur_face.match_info = MatchInfo(
                score=match_infos[i].get('score', 0.0),
                face_id=match_infos[i].get('face_id', -1),
                name=match_infos[i].get('name', '')
            )
        return cur_image

    def search_by_milvus(self, cur_images: list[Image]) -> list[Image]:

        per_image_faces_num = []
        normed_embeddings: list[np.ndarray] = []
        for cur_image in cur_images:
            per_image_faces_num.append(len(cur_image.faces))
            for face in cur_image.faces:
                normed_embeddings.append(face.normed_embedding)
        results: list[list[dict]] = self.milvus.search(normed_embeddings)
        j = 0
        per_image_faces_num = list(itertools.accumulate(per_image_faces_num))
        match_infos = []
        for i, result in enumerate(results):
            result = result[0]  # top_k=1
            if result['score'] > 0.4:
                match_infos.append({'score': result['score'],
                                    'face_id': result['id'],
                                    'name': result['name']})
            else:
                match_infos.append([])
            if i == per_image_faces_num[j] - 1:
                self._set_match_info(cur_images[j], match_infos)
                j += 1
                match_infos = []
        return cur_images

    '''def search_by_cosine(self, cur_image: Image) -> Image:
        # 用余弦相似度进行匹配
        normed_ip = lambda Face_a, Face_b: np.dot(Face_a.normed_embedding, Face_b.normed_embedding)
        for cur_face in cur_image.faces:
            match_info_params = {'score': 0.0,
                                 'face_id': -1,
                                 'name': ''}
            for face in self.registered_faces:
                if not isinstance(face.normed_embedding, np.ndarray):
                    raise TypeError('embedding must be a numpy array')
                score = normed_ip(cur_face, face)
                if score > match_info_params['score'] > 0.5:
                    match_info_params['score'] = score
                    match_info_params['face_id'] = face.id
                    match_info_params['name'] = face.name
            self._set_match_info(cur_face, **match_info_params)
        return cur_image'''

    def face_match(self, cur_images: list[Image], match_thresh: float = 0.6) -> list[Image] | None:
        """
        人脸匹配，返回一cur_face对象，cur_face设置了match_info，作为匹配结果
        if_matched为True表示匹配成功，False表示匹配的score低于阈值
        result为True表示匹配正确，False表示匹配失败
        score为匹配的分数，越高越好
        :param cur_images: 多张图片，包含多个人脸，等待匹配
        :param match_thresh: 匹配的阈值，大于该值才算匹配成功，计算方式这里都是normed_embedding的内积
        :return: cur_face
        """
        from ..data.image import Image
        # 类型检查
        if not isinstance(cur_images[0], Image):
            raise TypeError('cur_face is not a Image object !')
        if not self.registered_faces:
            raise ValueError('registered_faces is empty !')
        if not 0.0 <= match_thresh <= 1.0:
            raise ValueError('match_thresh is not in [0.0,1.0]')
        if self.milvus:
            return self.search_by_milvus(cur_images)
        else:
            return  # self.search_by_cosine(cur_images)

    def stop_milvus(self):
        if self.milvus and self.milvus.has_collection:
            self.milvus.shut_down()
