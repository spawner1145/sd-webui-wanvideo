# sd-webui-wanvideo

太菜了，看不懂kj nodes，所以直接用diffusers了(

本项目已支持api

---

## 注意事项

1. **models文件夹位置**

   * 如果你使用的是 `sd-webui` 插件，请将 `scripts` 文件夹内的 `models` 文件夹剪切到 `webui` 根目录下，并与原有的 `models` 文件夹合并。
   * 如果你是单独运行项目，则模型文件夹的路径结构如下：

     ```
     sd-webui-wanvideo/
     ├── install.py
     ├── requirements.txt
     ├── scripts/
     │   └── app.py  # 主脚本文件
     ├── backend/
     │   ├── api.py
     │   ├── inferrence.py
     │   └── ui.py
     ├── models/wan2.1/
     │   ├── dit/  # 放底模的
     │   │      ├── xxx001.safetensors
     │   │      ├── xxx002.safetensors
     │   │      └── ......
     │   ├── t5/  # T5 模型
     │   ├── vae/  # VAE 模型
     │   ├── lora/  # LoRA 模型
     │   └── image_encoder/  # CLIP 模型
     ├── api_examples/  # api调用示例文件
     │   ├── t2v.py
     │   ├── i2v.py
     │   └── v2v.py
     ├── license
     └── README.md
     ```
2. **启动方式**

   * **作为 `sd-webui` 插件** ：将项目放入 `extensions` 文件夹中即可。
   * **单独运行** ：
     在项目根目录下运行以下命令：

     ```
     python -m scripts.app
     ```

   **注意** ：

   * 单独运行时，需先安装依赖。请确保已安装 `torch` 和 `torchvision`，然后运行以下命令安装其他依赖：

     ```
     pip install -r requirements.txt
     ```
   * **不要运行 `install.py`** ，它仅用于插件模式。

---

## 模型下载与配置

### 下载地址

* [ModelScope](https://www.modelscope.cn/) 或 [HuggingFace](https://huggingface.co/)
  （以通义万相 2.1 文生视频 1.3B 模型为例）

#### 文生视频模型 (Wan2.1-T2V-1.3B)

* **dit 模型**
  下载 [diffusion_pytorch_model.safetensors](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/file/view/master?fileName=diffusion_pytorch_model.safetensors&status=2)，放入 `models/wan2.1/dit/` 文件夹
* **t5 模型**
  下载 [models_t5_umt5-xxl-enc-bf16.pth](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/file/view/master?fileName=models_t5_umt5-xxl-enc-bf16.pth&status=2)，放入 `models/wan2.1/t5/` 文件夹，注意在插件根目录下有 `models/wan2.1/t5/google` 这么一个文件夹，如果你作为webui插件运行，把它放到webui根目录的相应位置下
* **vae 模型**
  下载 [Wan2.1_VAE.pth](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/file/view/master?fileName=Wan2.1_VAE.pth&status=2)，放入 `models/wan2.1/vae/` 文件夹

#### 图生视频模型 (Wan2.1-I2V-14B-480P)

* **image_encoder 模型**
  下载 [models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P/file/view/master?fileName=models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth&status=2)，放入 `models/wan2.1/image_encoder/` 文件夹

#### 分片模型

* 如果遇到分片模型（如 `diffusion_pytorch_model.safetensors` 带有 `00001 of 00001` 的后缀），需要下载所有分片文件，并将它们全部放入 `dit` 文件夹
* 同时下载对应的索引文件（如 `diffusion_pytorch_model.safetensors.index.json`），也放入 `dit` 文件夹
* 使用时，界面会自动加载所有相关分片

---

## 控制模型与 Inpaint 模型

* **控制模型**

  HuggingFace 地址：[alibaba-pai/Wan2.1-Fun-1.3B-Control](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-Control)

  ModelScope 地址：[pai/Wan2.1-Fun-1.3B-Control](https://www.modelscope.cn/models/pai/Wan2.1-Fun-1.3B-Control)
* **Inpaint 模型**

  HuggingFace 地址：[alibaba-pai/Wan2.1-Fun-1.3B-InP](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP)

  ModelScope 地址：[pai/Wan2.1-Fun-1.3B-InP](https://www.modelscope.cn/models/pai/Wan2.1-Fun-1.3B-InP)
