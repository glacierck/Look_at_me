from __future__ import annotations
import cv2
from pathlib import Path
import numpy as np
from ..app.common import Face
from ..utils.my_tools import get_nodigits
from typing import NamedTuple


def draw_on(image2draw_on) -> np.ndarray:
    dimg = image2draw_on.nd_arr
    for i in range(len(image2draw_on.faces)):
        face = image2draw_on.faces[i]
        # face=[bbox, kps, det_score, match_info]
        box = face[0].astype(int)
        # 淡紫色
        lavender = (238, 130, 238)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), lavender, 1)
        if face[1] is not None:
            kps = face[1].astype(int)
            # print(landmark.shape)
            for l in range(face[1].shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                           2)

        light_green = (152, 251, 152)
        if face[-1] and face[-1].name:
            font_scale = 1
            # 设置文本的位置，将文本放在人脸框的上方
            text_position = (box[0] + 6, box[3] - 6)
            # 添加文本
            cv2.putText(img=dimg,
                        text=face.match_info.name,
                        org=text_position,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=font_scale,
                        color=light_green,
                        thickness=2,
                        lineType=cv2.LINE_AA)
    return dimg


class LightImage(NamedTuple):
    nd_arr: np.ndarray
    # faces = [face, face, ...]
    faces: list[list] = []
    # face=[bbox, kps, det_score,colors,match_info]
    screen_scale: tuple[int, int, int, int] = (0, 0, 0, 0)


# Image类
class Image:
    ImageCache = {}

    def __init__(self, root: Path = Path(__file__).parent.absolute(), **kwargs):
        self.image_dir = root
        self._name = root.stem
        self.faces: list[Face] = []
        self.nd_arr = kwargs.get('nd_arr', None)

        self.images_npz = kwargs.get('image_npz', None)
        self.to_rgb = kwargs.get('to_rgb', False)
        self.use_cache = kwargs.get('use_cache', True)
        self.cache_name = kwargs.get('cache_name', None)
        self.kwargs = kwargs
        self.ext_names = ['.jpg', '.png', '.jpeg']

    def load_image(self):
        refresh = self.kwargs.get('refresh', False)
        if self.images_npz and not refresh:
            if not self.images_npz.exists() or not self.images_npz.is_file():
                print(f"{self.images_npz} doesn't exist !")
                return
            files = np.load(str(self.images_npz))
            img = files[self._name] if self._name in files else None
        else:
            assert self.image_dir.exists(), f"{self.image_dir} doesn't exist !"
            assert self.image_dir.suffix in self.ext_names, f"{self.image_dir} is not a image file !"
            print(f'get image from: {self._name}')
            img = cv2.imread(str(self.image_dir))
            assert isinstance(img, np.ndarray), f"image: {self._name} read result is not np.ndarray !"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if self.to_rgb else img
        self.nd_arr = img
        if self.use_cache and self.cache_name:
            Image.ImageCache.setdefault(self.cache_name, []).append(self)
        return self

    @property
    def face_locations(self):
        return [face.face_location for face in self.faces]

    @property
    def name(self):
        chars = '_-. '
        name = self._name.strip(chars)
        return get_nodigits(name)

    def __add__(self, other: Image):
        assert isinstance(other, Image), f"other is not Image type !"
        assert self.nd_arr is not None, f"self.img_np is None !"
        # 检查两个图像的高度是否相同，如果不同则调整为相同的高度
        if self.nd_arr.shape[0] != other.nd_arr.shape[0]:
            height = min(self.nd_arr.shape[0], other.nd_arr.shape[0])
            self.nd_arr = cv2.resize(self.nd_arr, (self.nd_arr.shape[1], height))
            other.nd_arr = cv2.resize(other.nd_arr, (other.nd_arr.shape[1], height))

        # 使用numpy的hstack函数将两个图像横向拼接在一起
        result = np.hstack((self.nd_arr, other.nd_arr))

        # 创建一个新的Image对象，并将拼接后的图像赋值给它的np属性
        new_image = Image.__new__(Image)
        new_image.nd_arr = result
        new_image.faces = [*self.faces, *other.faces]
        new_image._name = f"{self.name} + {other.name}"

        return new_image

    def draw_on(self) -> np.ndarray:

        dimg = self.nd_arr
        for i in range(self.face_count):
            face = self.faces[i]
            box = face.bbox.astype(int)
            # 淡紫色
            lavender = (238, 130, 238)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), lavender, 1)
            if face.kps is not None:
                kps = face.kps.astype(int)
                # print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)

            light_green = (152, 251, 152)
            if face.match_info and face.match_info.name:
                font_scale = 1
                # 设置文本的位置，将文本放在人脸框的上方
                text_position = (box[0] + 6, box[3] - 6)
                # 添加文本
                cv2.putText(img=dimg,
                            text=face.match_info.name,
                            org=text_position,
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=font_scale,
                            color=light_green,
                            thickness=2,
                            lineType=cv2.LINE_AA)

        return dimg

    def show(self, face_on: bool = False, write_on: bool = False):
        dim = None
        if face_on:
            dim = self.draw_on()
        if write_on:
            self.nd_arr = dim
        cv2.imshow(self.name, dim)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @property
    def face_count(self):
        return len(self.faces)

    def __repr__(self):
        return f"Image(name={self.name})"


def get_images(img_folder: str, root: Path = Path(__file__).parent.absolute(), **kwargs) -> list[Image]:
    """
        读取.\\data\\images根下读取指定目录所有的图片，并且返回对应的图片名称和图片,如果存在
        """

    print(f"\nReading images from: {img_folder}")
    print(f"Reading images from: {root}")
    print(f"cur_params: {kwargs}")
    test_folder = kwargs.get('test_folder', None)

    if test_folder:
        img_dirs = Path(root, 'images', test_folder, img_folder)
    else:
        img_dirs = Path(root, 'images', img_folder)
    assert img_dirs.exists(), f"{img_dirs} doesn't exist !"
    assert img_dirs.is_dir(), f"{img_dirs} is not a directory !"
    # 构造图片的npyz文件文件
    npyz_path = img_dirs.joinpath("images.npz")
    images_npyz = npyz_path if npyz_path.exists() and npyz_path.is_file() else None
    images = [Image(root=img_dir, images_npyz=images_npyz, **kwargs).load_image()
              for img_dir in img_dirs.iterdir() if img_dir.suffix in ['.jpg', '.png', '.jpeg']]
    if not images_npyz or kwargs.get('refresh', False):
        np.savez_compressed(str(npyz_path), **{img.name: img.nd_arr for img in images})
        print(f"\nSaving images from folder {img_folder} to: {npyz_path.name}")

    return images
