import subprocess

# 下载 diffusers 的 github 源码版
subprocess.run(["pip", "install", "git+https://github.com/huggingface/diffusers.git"])

# 安装其他依赖
subprocess.run(["pip", "install", "transformers", "modelscope", "huggingface_hub", "numpy"])
    
