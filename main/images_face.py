from __future__ import annotations

import functools
import logging
import pprint

from my_insightface.insightface.app.face_analysis import FaceAnalysis
from my_insightface.insightface.app.common import Face
from typing import *
from pathlib import Path
from my_insightface.insightface.data.image import Image, get_images
from my_insightface.insightface.utils.my_tools import flatten_list
from timeit import default_timer as current_time

COST_TIME = {}


def cost_time_recording(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = current_time()
        res = func(*args, **kwargs)
        end = current_time()
        COST_TIME[func.__name__] = end - start
        return res

    return wrapper


class FaceAnalysisTest(FaceAnalysis):
    def __init__(self, root: Path = Path.cwd().parents[1].joinpath('models\\insightface'),
                 providers: list[str] = ('CUDAExecutionProvider', 'CPUExecutionProvider'),
                 if_milvus: bool = True, **kwargs):
        self._wrong_faces: list[Face] = []
        self._unkn_imgs: list[Image] = []
        self._test_folder = None
        self._rec_models = ['w600k_r50.onnx', 'irn50_glint360k_r50.onnx', 'irn_r100_glintk360k.onnx']
        self._det_models = ['det_10g.onnx', 'det_2.5g.onnx']
        self._landmark_models = ['2d106det.onnx', '1k3d68.onnx']
        self._test_models = (self._det_models[1], self._rec_models[1])
        allowed_modules = ['detection', 'recognition']
        super().__init__(names=self._test_models, root=root, providers=providers, if_milvus=if_milvus,
                         allowed_modules=allowed_modules)
        prepare_params = {'ctx_id': kwargs.get('ctx_id', 0),
                          'det_thresh': kwargs.get('det_thresh', 0.5),
                          'det_size': kwargs.get('det_size', (320, 320))}
        self.prepare(**prepare_params)  # 0:GPU, -1:CPU

        # Use logging instead of print
        logging.info(f'Face_analysis.det_thresh = {self.det_thresh}')
        logging.info(f'Face_analysis.det_size = {self.det_size}')
        logging.info(f'Face_analysis.det_model = {self.det_model}')
        logging.info(f'Face_analysis.rec_model = {self.models["recognition"]}')
        logging.info(f'FaceAnalysisTest init done, models: {self._test_models}')

    @cost_time_recording
    def load_registered_data(self, test_folder: str, refresh: bool = False) -> List[Face]:
        self.load_registered_faces(test_folder=test_folder, img_folder='known', refresh=refresh)
        return self.registered_faces

    @cost_time_recording
    def load_unknown_images(self, test_folder: str) -> List[Image]:
        self._test_folder = test_folder
        unkn_img_dir = Path(f'..\\my_insightface\\insightface\\data\\images\\{self._test_folder}')
        assert unkn_img_dir.exists() and unkn_img_dir.is_dir(), f'unkn_img_dir = {unkn_img_dir} is not a dir'
        self._unkn_imgs: list[Image] = list(flatten_list([
            get_images(test_folder=test_folder, img_folder=unkn_img_dir.name, cache_name=unkn_img_dir)
            for unkn_img_dir in unkn_img_dir.iterdir() if
            unkn_img_dir.is_dir() and unkn_img_dir.name not in ['known']]))
        return self._unkn_imgs

    @cost_time_recording
    def get_unknown_faces(self) -> List[Image]:
        for image in self._unkn_imgs:
            self.detect(image)
            self.get(image, face_name=image.name)
        return self._unkn_imgs

    @cost_time_recording
    def face_search(self) -> List[Image]:
        self._res_images = [self.face_match(img) for img in self._unkn_imgs]
        return self._res_images

    def calculate_match_results(self):

        print('calculate_test_results begins')
        self._successful = 0
        self.total_faces = 0
        for res_img in self._res_images:
            for face in res_img.faces:
                self.total_faces += 1
                if face.match_info.face_id == -1:
                    print(f'face.name = {face.name},match_score={face.match_info.score}, matched None')
                    self._wrong_faces.append(face)
                elif face.match_info.name != face.name:
                    print(f'face.name = {face.name}, face.match_info.face.name = {face.match_info.name},matched wrong')
                    self._wrong_faces.append(face)
                else:
                    self._successful += 1

    def show_test_results(self):
        res_face_num = self.total_faces
        succeed_rate = self._successful / res_face_num if res_face_num else 0
        ave_rec_time = COST_TIME['get_unknown_faces'] / res_face_num if res_face_num else 0
        ave_match_time = COST_TIME['face_search'] / res_face_num if res_face_num else 0
        print('\ntest scale as follows:',
              f'test target folder = {self._test_folder}',
              f'res_faces_num = {res_face_num}', sep='\n')
        print('\nmodels parameters as follows',
              f'Face_analysis.det_thresh = {self.det_thresh}',
              f' Face_analysis.det_size = {self.det_size}',
              f'test_model = {self._test_models}', sep='\n')
        if self.milvus and self.milvus.has_collection:
            print('\nmilvus parameters as follows')
            pprint.pprint(self.milvus.milvus_params)
        print('\ntest cost time and accuracy as follows',
              f'load_registered_data cost time = {COST_TIME["load_registered_data"]:.4f} sec',
              f'get_unknown_images cost time = {COST_TIME["load_unknown_images"]:.4f} sec',
              f'ave_match_time = {ave_match_time:.6f} sec',
              f'ave_rec_time = {ave_rec_time:.4f} sec',
              f'succeed_rate = {succeed_rate * 100:.4f} %', sep='\n')

    def display_wrong_resluts(self):

        print(f'\nwrong_results_num = {len(self._wrong_faces)}')
        result_imgs: list[Image] = []
        for face in self._wrong_faces:
            wrong_face_img = None
            matched_img = None
            for img in self._unkn_imgs:
                if face.id == img.faces[0].id:
                    wrong_face_img = img
                    break
            for img in self.registered_faces:
                if face.match_info.face is None:
                    matched_img = wrong_face_img
                elif face.match_info.face.name == img.name:
                    matched_img = img
                    break
            result_imgs.append(wrong_face_img + matched_img)
        if result_imgs:
            for img in result_imgs:
                img.show(face_on=True)

    def test_stop(self):
        self.stop_milvus()
        print('test_of_face_analysis ends!')


def main():
    test_folder = 'test_01'
    test = FaceAnalysisTest()
    try:
        test.load_registered_data(test_folder=test_folder, refresh=False)
        test.load_unknown_images(test_folder=test_folder)
        test.get_unknown_faces()
        test.face_search()
        test.calculate_match_results()
        test.show_test_results()
        # test.display_wrong_resluts()
    except Exception as e:
        print(f'Exception occurs, error = {e}')
        raise e
    finally:
        test.test_stop()


if __name__ == '__main__':
    main()
