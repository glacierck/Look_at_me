import queue
import subprocess
from pathlib import Path

import cv2
from ..data import LightImage


class Camera:
    def __init__(
            self,
            app: str = "",
            approach: str = "",
            test_folder: str = "test_01",
            resolution: tuple = (1920, 1080),
            fps: int = 30,
    ):
        """
        cmd 运行setx OPENCV_VIDEOIO_PRIORITY_MSMF 0后重启，可以加快摄像头打开的速度
        :param app:
        :param approach:
        :param test_folder:
        :param resolution:
        :param fps:
        """
        self._imgs_of_video = 0
        self._test_folder = test_folder
        self.videoCapture = None
        order = [app, approach]
        match order:
            case ["auto_focus", _]:
                self._url = 1
            case ["manual_focus", _]:
                self._url = 0
            case ["mp4", _]:
                video_dir = (
                        Path(__file__).absolute().parents[3]
                        / f"database\\milvus_standalone\\data\\{self._test_folder}\\video"
                )
                if not video_dir.exists() and not video_dir.is_dir():
                    raise FileNotFoundError(f"video_dir = {video_dir}")
                video_path = list(video_dir.glob("*.mp4"))[0]
                assert (
                        video_path.exists() and video_path.is_file()
                ), f"video_path = {video_path}"
                self._url = video_path.as_posix()
            case ["laptop", _]:
                self._url = 0
            case ["ip_webcam", "usb"]:
                self._url = "http://localhost:8080/video"
                cmd = "adb forward tcp:8080 tcp:8080"
                # 使用subprocess运行命令
                process = subprocess.Popen(
                    cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                # 获取命令输出
                stdout, stderr = process.communicate()
                # 如果有输出，打印出来
                if stdout:
                    print("STDOUT:{}".format(stdout))
                if stderr:
                    raise ValueError("STDERR:{}".format(stderr))
            case ["ip_webcam", "wifi"]:
                self._url = "http://192.168.0.103:8080/video"
            case _:
                raise ValueError(f"Wrong app or approach: {app}, {approach}")
        try:
            print("trying to open video source...")
            self.videoCapture = cv2.VideoCapture(self._url)
            if not self.videoCapture.isOpened():
                raise ValueError(
                    f"Could not open video source {self._url} even after sending request to override link"
                )
        except ValueError as e:
            # 处理任何可能的请求错误
            print(f"Request to override link failed with error: {e}")
        print("setting camera fps and resolution...")
        #  设置帧数
        self.videoCapture.set(cv2.CAP_PROP_FPS, int(fps))
        # 设置分辨率
        self.videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
        self.videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))

        # 设置视频编解码格式 note: 务必在set分辨率之后设置，否则不知道为什么又会回到默认的YUY2
        self.videoCapture.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G")
        )
        print(f"The video  codec  is {self.cap_codec_format}")
        # 获取分辨率
        self._resolution = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        # 获取帧数
        self._fps = int(self.videoCapture.get(cv2.CAP_PROP_FPS))
        # set and true
        self.params = {
            "fps": (fps, self._fps),
            "resolution": self._resolution,
            "url": self._url,
        }

    def read_video(self, results: queue.Queue):
        from .multi_thread_analysis import threads_done

        print("camera_read start")
        try:
            while not threads_done.is_set():
                ret, frame = self.videoCapture.read()
                if ret:
                    results.put(
                        LightImage(
                            nd_arr=frame,
                            faces=[],
                            screen_scale=(0, 0, frame.shape[1] - 1, frame.shape[0] - 1),
                        )
                    )
                    self._imgs_of_video += 1
                else:
                    break
        finally:
            self.videoCapture.release()

    @property
    def cap_codec_format(self):
        # 获取当前的视频编解码器
        fourcc = self.videoCapture.get(cv2.CAP_PROP_FOURCC)
        # 因为FOURCC编码是一个32位的值，我们需要将它转换为字符来理解它
        # 将整数编码值转换为FOURCC编码的字符串表示形式
        codec_format = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        return codec_format
