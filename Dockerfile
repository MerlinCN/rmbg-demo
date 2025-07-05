# 基于官方 PyTorch 镜像
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 设置 uv 的全局配置
RUN mkdir -p /root/.config/uv && \
    echo '[[index]]\nurl = "https://pypi.tuna.tsinghua.edu.cn/simple"\ndefault = true' > /root/.config/uv/uv.toml

EXPOSE 7788

RUN uv sync

# 设置默认启动命令（可根据需要修改）
CMD ["uv", "run", "main.py"] 