
import os
import os.path as osp
import zipfile
from .download import download_file
from pathlib import Path

BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'

def download(sub_dir, name, force=False, root='~/.insightface'):
    """
    下载出现sll证书问题
    :param sub_dir:
    :param name:
    :param force:
    :param root:
    :return:
    """
    _root = os.path.expanduser(root)
    dir_path = os.path.join(_root, sub_dir, name)
    if osp.exists(dir_path) and not force:
        return dir_path
    print('download_path:', dir_path)
    zip_file_path = os.path.join(_root, sub_dir, name + '.zip')
    model_url = "%s/%s.zip"%(BASE_REPO_URL, name)
    download_file(model_url,
             path=zip_file_path,
             overwrite=True)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(dir_path)
    #os.remove(zip_file_path)
    return dir_path

def ensure_available(name: str, root:Path):
    """
    确保文件model模型文件存在，不存在就下载
    """
    assert root is not None, "root is None"
    onnx_file_path = Path(root, name)
    if onnx_file_path.exists() and onnx_file_path.is_file():
        return onnx_file_path
    else:
        return # download('models', name, force=False, root=str(root))

def download_onnx(sub_dir, model_file, force=False, root='~/.insightface', download_zip=False):
    _root = os.path.expanduser(root)
    model_root = osp.join(_root, sub_dir)
    new_model_file = osp.join(model_root, model_file)
    if osp.exists(new_model_file) and not force:
        return new_model_file
    if not osp.exists(model_root):
        os.makedirs(model_root)
    print('download_path:', new_model_file)
    if not download_zip:
        model_url = "%s/%s"%(BASE_REPO_URL, model_file)
        download_file(model_url,
                 path=new_model_file,
                 overwrite=True)
    else:
        model_url = "%s/%s.zip"%(BASE_REPO_URL, model_file)
        zip_file_path = new_model_file+".zip"
        download_file(model_url,
                 path=zip_file_path,
                 overwrite=True)
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall(model_root)
        return new_model_file
