import itertools
from pathlib import Path

import cv2
import numpy as np
from numpy import ndarray
from numpy.lib.npyio import NpzFile
from .milvus_lite import Milvus
from my_insightface.insightface.app.face_analysis import MatchInfo
from my_insightface.insightface.app.real_time_tracker import Target, Detector, Extractor

from my_insightface.insightface.data.image import LightImage
from my_insightface.insightface.utils.my_tools import get_digits, get_nodigits


class MilvusRealTime:
    def __init__(self, test_folder: str = 'test_01', img_folder: str = 'known'):
        self._match_threshold: float = 0.5
        self._milvus = Milvus()
        self._image_root = Path(__file__).parent.absolute() / 'data'
        self._image_folder: Path = self._image_root / test_folder / img_folder
        self._npz_path: Path = self._image_folder / 'faces.npz'

    def _get_faces_from_images(self, **kwargs) -> NpzFile:
        ext_names = {'.jpg', '.png', '.jpeg'}
        detector = kwargs.get('detector', Detector())
        extractor = kwargs.get('extractor', Extractor())
        faces2npz = []
        img_paths = [str(img_path) for img_path in self._image_folder.glob('*')
                     if img_path.suffix in ext_names]
        if not img_paths:
            raise FileNotFoundError(f'no image found in {img_paths}')
        print(f'\nloading faces from {self._image_folder}')
        for i, img_path in enumerate(img_paths):
            print(f'processing {i}/{len(img_paths)} image from {img_path}')
            try:
                img_ndarray = cv2.imread(img_path)
                if img_ndarray is None:
                    raise FileNotFoundError
            except FileNotFoundError:
                print(f"The file at {img_path} does not exist or is not a valid image.")
                return
            img = LightImage(img_ndarray, [],
                             (0, 0, img_ndarray.shape[1] - 1, img_ndarray.shape[0] - 1))
            detector(img)
            if len(img.faces) > 1:
                print(f'Warning: more than one face detected in {img_path}')
                continue
                # raise ValueError('face should be one')
            normed_embedding = extractor(img, bbox=img.faces[0][0],
                                         kps=img.faces[0][1], det_score=img.faces[0][2])
            # name consists of digits and letters: 123abc
            file_name = Path(img_path).stem
            _id = get_digits(file_name)
            if not _id:
                print(f'Warning: no id found in {img_path}')
                continue
            _name = get_nodigits(file_name)
            faces2npz.append([_id, _name, normed_embedding])
        self._faces_to_npz(faces2npz)
        try:
            faces_npz: NpzFile = np.load(str(self._npz_path))
        except OSError:
            print('npz file not found,after np.savez_compressed')
            raise
        else:
            return faces_npz

    def _get_faces_from_npz(self, **kwargs) -> None:
        """
        load faces from npz files
        :return:
        """
        faces_npz = None
        print('\nloading registered faces from npz files')
        try:
            if kwargs.get('refresh', False):
                raise OSError
            faces_npz: NpzFile = np.load(str(self._npz_path))
        except (OSError, FileNotFoundError):
            print('npz file not found, loading from images')
            faces_npz: NpzFile = self._get_faces_from_images(**kwargs)
        finally:
            # shape: (n,)
            ids: ndarray = faces_npz['ids']
            mask = ids != ''
            ids = ids[mask].astype(np.int64)
            # shape: (n,)
            names: ndarray = faces_npz['names'][mask]
            # shape: (n, 512)
            normed_embeddings: ndarray = faces_npz['normed_embeddings'][mask].astype(np.float32)
            faces_npz.close()
        if not (len(ids) == len(names) == len(normed_embeddings)):
            raise ValueError('npz files not match')
        if not (ids.all() or names.all() or normed_embeddings.all()):
            raise ValueError('npz files not complete')
        if self._milvus.has_collection:
            # self._milvus.insert_from_files(
            # file_paths=[str(id_npy), str(name_npy), str(normed_embedding_npy)])
            self._milvus.insert([[_id for _id in ids],
                                 [name for name in names],
                                 [embedding for embedding in normed_embeddings]
                                 ])

    def _faces_to_npz(self, faces: list[list]) -> None:

        ids, names, embeddings = [], [], []
        for face in faces:
            ids.append(face[0])
            names.append(face[1])
            embeddings.append(face[2])
        np.savez_compressed(str(self._npz_path),
                            ids=ids, names=names, normed_embeddings=embeddings)
        print(f'\nsave {len(faces)} target faces to Npzfile done')

    def _search_by_milvus(self, targets: list[Target]) -> list[Target]:

        normed_embeddings: list[np.ndarray] = [target.normed_embedding for target in targets]
        results: list[list[dict]] = self._milvus.search(normed_embeddings)

        for i, result in enumerate(results):
            result = result[0]  # top_k=1
            if result['score'] > self._match_threshold:
                targets[i].set_match_info(MatchInfo(score=result['score'],
                                                    face_id=result['id'], name=result['name']))
        return targets

    @property
    def server_params(self) -> bool:
        return self._milvus.milvus_params

    @property
    def total_faces(self) -> int:
        return self._milvus.get_entity_num()

    def load_registered_faces(self, detector: Detector, extractor: Extractor, refresh: bool = False) -> None:
        kwargs = {'detector': detector, 'extractor': extractor, 'refresh': refresh}
        self._get_faces_from_npz(**kwargs)
        return

    def face_match(self, targets: list[Target], match_thresh: float = 0.6) -> list[Target] | None:
        """
        人脸匹配，返回一cur_face对象，cur_face设置了match_info，作为匹配结果
        if_matched为True表示匹配成功，False表示匹配的score低于阈值
        result为True表示匹配正确，False表示匹配失败
        score为匹配的分数，越高越好
        :param targets: target列表，每个target包含了人脸的信息
        :param match_thresh: 匹配的阈值，大于该值才算匹配成功，计算方式这里都是normed_embedding的内积
        :return: cur_face
        """
        # 类型检查
        if not isinstance(targets, list):
            raise TypeError('targets is not a list')
        if not targets:
            return []
        if not 0.0 <= match_thresh <= 1.0:
            raise ValueError('match_thresh is not in [0.0,1.0]')
        self._match_threshold = match_thresh
        assert self._milvus, '_milvus is off'
        return self._search_by_milvus(targets)

    def stop_milvus(self):
        if self._milvus and self._milvus.has_collection:
            self._milvus.shut_down()
