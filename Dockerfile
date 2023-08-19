# 使用合适的基础镜像
FROM python:3.11.4

# 设置工作目录
WORKDIR /app

# 安装Redis
RUN apt-get update && apt-get install -y redis-server

# 复制requirements.txt并安装依赖
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有源代码到工作目录
COPY . /app

# 暴露端口
EXPOSE 5000

# 运行你的多线程Flask应用程序
CMD ["python", "gunicorn_test.py"]
