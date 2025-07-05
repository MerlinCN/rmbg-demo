# RMBG Demo

## 1. 模型准备

在构建和运行容器前，请先使用 [huggingface_hub-cli](https://huggingface.co/docs/huggingface_hub/package_reference/cli) 命令行工具下载所需模型文件到 `model` 目录。例如：

```bash
huggingface-cli download 你的模型仓库名 --local-dir ./model --local-dir-use-symlinks False
```

确保 `model` 目录下有完整的模型文件。

## 2. 构建 Docker 镜像

在项目根目录下执行：

```bash
docker build -t rmbg:latest .
```

## 3. 启动容器

构建完成后，可以通过以下脚本启动服务：

- Windows:

```powershell
./start.ps1
```

- Linux/macOS:

```bash
./start.sh
```

如需自定义参数，请根据脚本内容自行调整。
