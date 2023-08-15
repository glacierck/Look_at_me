"""
启动docker
并且要保证数据库是我的sqlite3
"""
import docker
from pathlib import Path


# 目录映射 Look_at_me\\database\\sqlite3 ->/app/data
# metabase访问的数据目录


def docker_start():
    print("正在启动 Docker 容器...")
    client = docker.from_env()

    # 检查 Docker 权限
    try:
        print("正在检查 Docker 权限...")
        client.ping()
    except docker.errors.APIError as e:
        raise Exception("无法访问 Docker API。请确保您具有正确的 Docker 权限。") from e

    # 检查 Docker 引擎
    if not client.ping():
        raise Exception("无法访问 Docker 引擎。请确保 Docker 引擎正在运行。")
    print("Docker 引擎正常运行。")
    print("正在检查 Docker 镜像...")
    # 检查文件路径
    host_dir = Path("D:\\Users\\Atticus\\OneDrive\\CXXY\\Competition\\DC\\FR_pj\\Look_at_me\\database\\sqlite3")
    if not host_dir.is_dir() and not host_dir.exists():
        raise FileNotFoundError(f"无法找到文件夹：{host_dir}")

    # 检查镜像
    image_name = "metabase/metabase"
    try:
        print(f"正在检查 Docker 镜像：{image_name}...")
        client.images.get(image_name)
    except docker.errors.ImageNotFound:
        raise f"无法找到 Docker 镜像：{image_name}"

    try:
        container = client.containers.get("metabase")
        print('已找到名为 "metabase" 的容器')

    except docker.errors.NotFound:
        # 如果容器不存在，那么创建它
        container = client.containers.create(
            "metabase/metabase",
            name="metabase",
            volumes={"D:\\Users\\Atticus\\OneDrive\\CXXY\\Competition\\DC\\FR_pj\\Look_at_me\\database\\sqlite3": {
                'bind': '/app/data', 'mode': 'rw'}},
            ports={'3000/tcp': 2003}  # 映射容器的 3000 端口到宿主机的 2003 端口
        )
        print('已创建新的 "metabase" 容器')

        # 启动容器
    container.start()
    print('已启动 "metabase" 容器')


def docker_stop():
    client = docker.from_env()

    try:
        # 查找并停止名为 "metabase" 的容器
        container = client.containers.get("metabase")
        container.stop()
        print('已停止 "metabase" 容器')
    except docker.errors.NotFound:
        print('未找到名为 "metabase" 的容器。')


def main():
    try:
        docker_start()
        input("请手动打开浏览器访问 http://localhost:2003 并登录，然后按回车键stop docker...")
    except Exception as e:
        print(e)
    finally:
        docker_stop()


if __name__ == '__main__':
    main()
