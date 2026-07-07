FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p data outputs configs

# 设置执行权限
RUN chmod +x scripts/*.sh

# 默认命令
CMD ["python", "scripts/train.py", "--config", "configs/lora_config.yaml"]