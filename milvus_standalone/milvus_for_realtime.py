import itertools
from pathlib import Path

import numpy as np

from milvus_lite import Milvus
from my_insightface.insightface.app.face_analysis import MatchInfo
from my_insightface.insightface.app.real_time_tracker import Face, Target
from my_insightface.insightface.data.image import Image


class MilvusRealTime:
    def __init__(self, **kwargs):
        self._match_threshold = 0.5
        self.milvus = Milvus()
        self.__image_root = Path(__file__).parents[1].absolute() / 'data' / 'images'
        self.kwargs = kwargs

    def get_faces_from_npys(self, img_folder: str, test_folder: str) -> list[Face]:

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

    def faces_to_npys(self, faces: list[Face], img_folder: str, test_folder: str) -> None:
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

    def load_registered_faces(self, detector, extractor, img_folder: str, **kwargs) -> list[Face]:
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

    def _search_by_milvus(self, targets: list[Target]) -> list[Target]:

        normed_embeddings: list[np.ndarray] = [target.normed_embedding for target in targets]
        results: list[list[dict]] = self.milvus.search(normed_embeddings)

        for i, result in enumerate(results):
            result = result[0]  # top_k=1
            if result['score'] > self._match_threshold:
                targets[i].set_match_info(MatchInfo(score=result['score'],
                                                    face_id=result['id'], name=result['name']))
        return targets

    def face_match(self, targets: list[Target], match_thresh: float = 0.6) -> list[Target]:
        """
        人脸匹配，返回一cur_face对象，cur_face设置了match_info，作为匹配结果
        if_matched为True表示匹配成功，False表示匹配的score低于阈值
        result为True表示匹配正确，False表示匹配失败
        score为匹配的分数，越高越好
        :param cur_images: 多张图片，包含多个人脸，等待匹配
        :param match_thresh: 匹配的阈值，大于该值才算匹配成功，计算方式这里都是normed_embedding的内积
        :return: cur_face
        """
        # 类型检查
        if not isinstance(targets, list):
            raise TypeError('targets is not a list')
        if not self.registered_faces:
            raise ValueError('registered_faces is empty !')
        if not 0.0 <= match_thresh <= 1.0:
            raise ValueError('match_thresh is not in [0.0,1.0]')
        self._match_threshold = match_thresh
        assert self.milvus, 'milvus is off'
        return self._search_by_milvus(targets)

    def stop_milvus(self):
        if self.milvus and self.milvus.has_collection:
            self.milvus.shut_down()
