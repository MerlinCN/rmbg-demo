#!/bin/bash

# 检查 HF_TOKEN 是否已设置
if [ -z "$HF_TOKEN" ]; then
  echo "请先设置 HF_TOKEN 环境变量。"
  exit 1
fi

# 启动主程序
docker run --gpus all -d -p 7788:7788 --env HF_TOKEN=$HF_TOKEN --name rmbg rmbg:latest