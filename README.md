
# sd-webui-wanvideo

太菜了，看不懂kj nodes，所以直接用diffusers了(

注意一下，scripts文件夹内有个models文件夹，如果你用webui插件的把那个models文件夹剪切到webui根目录下和原来的models文件夹合并

此项目不是必须作为webui插件，scripts文件夹下的generation.py可以单独运行，如果你是单独跑generation.py，把scripts作为工作目录打开，注意先把requirements.txt里的依赖装好（注意不是项目根目录！）

模型装好，路径示例：

```
sd-webui-wanvideo/
├── install.py
├── requirements.txt
├── models.py
├── scripts/
|	├── generation.py  # 主脚本文件
|	├── gguf.py     # 施工中(
|	└── models/wan2.1/
|		├── dit/ # 放底模的
|		|     	├── xxx001.safetensors
|		|	├── xxx002.safetensors
|		|	└── ......
|		├── t5/  # 不解释(注意里面那个google的文件夹里的东西别删了)
|		├── vae/  # 不解释
|		├── lora/  # 不解释
|		└── image_encoder/  # 不解释,就是clip
├── license
└── README.md
```

如果你是webui，那么models文件夹就是在根目录下的，而然后wan2.1时models文件夹里一个子文件夹，然后相关模型的子文件夹又在wan2.1这个文件夹里

你如果要下模型：

去[modelscope(huggingface同理，这里我以wan-2.1的1.3b文生视频为例)](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/files)这里找到[diffusion_pytorch_model.safetensors](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/file/view/master?fileName=diffusion_pytorch_model.safetensors&status=2)这玩意扔dit文件夹里，[models_t5_umt5-xxl-enc-bf16.pth](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/file/view/master?fileName=models_t5_umt5-xxl-enc-bf16.pth&status=2)这玩意扔t5文件夹里，[Wan2.1_VAE.pth](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/file/view/master?fileName=Wan2.1_VAE.pth&status=2)这玩意扔vae文件夹里，然后如果要图生视频之类的，去[通义万相2.1-图生视频-14B-480P · 模型库](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P/files)比如这里，拿到[models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P/file/view/master?fileName=models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth&status=2)这玩意扔image_encoder里，然后注意，你如果遇到一个模型库里safetensors一大堆的，结尾大概是00001 of 00001这种，全都要下载，然后扔到dit文件夹里，界面里要用那个模型的时候把所有这些分片选上(dit模型),对于分片模型，一般还会有一个[diffusion_pytorch_model.safetensors.index.json](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B/file/view/master?fileName=diffusion_pytorch_model.safetensors.index.json&status=1)这种的，也一起扔到dit文件夹
