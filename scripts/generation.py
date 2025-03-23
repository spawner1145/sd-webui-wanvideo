import os
import torch
import gradio as gr
from diffusers.utils import export_to_video, load_image
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel
import numpy as np
from huggingface_hub import snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as modelscope_snapshot_download
from pathlib import Path
import gc

# 定义模型路径和输出路径
models_dir = Path("webui/models/wan")
outputs_dir = Path("webui/outputs/videos")
models_dir.mkdir(parents=True, exist_ok=True)
outputs_dir.mkdir(parents=True, exist_ok=True)

# 可用的文生视频模型
text_to_video_models = [
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
]

# 可用的图生视频模型
image_to_video_models = [
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
]

# 全局变量存储模型实例
text_pipeline = None
image_pipeline = None
text_vae = None
image_vae = None
image_encoder = None

def clear_memory():
    """清理GPU和RAM内存"""
    global text_pipeline, image_pipeline, text_vae, image_vae, image_encoder
    
    if text_pipeline is not None:
        del text_pipeline
        text_pipeline = None
    if image_pipeline is not None:
        del image_pipeline
        image_pipeline = None
    if text_vae is not None:
        del text_vae
        text_vae = None
    if image_vae is not None:
        del image_vae
        image_vae = None
    if image_encoder is not None:
        del image_encoder
        image_encoder = None
        
    torch.cuda.empty_cache()
    gc.collect()
    return "内存已清理"

def download_model(model_id, source):
    try:
        model_path = models_dir / model_id.split("/")[-1]
        if model_path.exists():
            print(f"模型 {model_id} 已存在于 {model_path}，无需再次下载。")
            return str(model_path)
        
        print(f"开始从 {source} 下载模型 {model_id}...")
        if source == "Hugging Face":
            snapshot_download(repo_id=model_id, local_dir=model_path)
        elif source == "ModelScope":
            modelscope_snapshot_download(model_id, cache_dir=models_dir)
        print(f"模型 {model_id} 下载完成，保存路径为 {model_path}。")
        return str(model_path)
    except Exception as e:
        raise gr.Error(f"模型下载失败: {str(e)}")

def text_to_video(prompt, negative_prompt, model_id, source, height, width, num_frames, guidance_scale, flow_shift, fps):
    global text_pipeline, text_vae
    
    try:
        # 生成前清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        model_path = download_model(model_id, source)
        
        if text_vae is None:
            text_vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
        if text_pipeline is None:
            scheduler = UniPCMultistepScheduler(
                prediction_type='flow_prediction',
                use_flow_sigmas=True,
                num_train_timesteps=1000,
                flow_shift=flow_shift
            )
            text_pipeline = WanPipeline.from_pretrained(model_path, vae=text_vae, torch_dtype=torch.bfloat16)
            text_pipeline.scheduler = scheduler
            text_pipeline = text_pipeline.to("cuda")

        output = text_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]
        
        output_path = outputs_dir / f"text_to_video_{int(torch.cuda.current_device())}.mp4"
        export_to_video(output, str(output_path), fps=fps)
        
        # 生成后清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        return str(output_path)
    except Exception as e:
        raise gr.Error(f"视频生成失败: {str(e)}")

def image_to_video(image, prompt, negative_prompt, model_id, source, num_frames, guidance_scale, fps):
    global image_pipeline, image_vae, image_encoder
    
    try:
        # 生成前清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        if image is None:
            raise ValueError("请提供输入图片")
            
        model_path = download_model(model_id, source)
        
        if image_encoder is None:
            image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=torch.float32)
        if image_vae is None:
            image_vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
        if image_pipeline is None:
            image_pipeline = WanImageToVideoPipeline.from_pretrained(
                model_path, 
                vae=image_vae, 
                image_encoder=image_encoder, 
                torch_dtype=torch.bfloat16
            )
            image_pipeline = image_pipeline.to("cuda")

        max_area = 720 * 1280
        aspect_ratio = image.height / image.width
        mod_value = image_pipeline.vae_scale_factor_spatial * image_pipeline.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))

        output = image_pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale
        ).frames[0]
        
        output_path = outputs_dir / f"image_to_video_{int(torch.cuda.current_device())}.mp4"
        export_to_video(output, str(output_path), fps=fps)
        
        # 生成后清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        return str(output_path)
    except Exception as e:
        raise gr.Error(f"视频生成失败: {str(e)}")

def on_ui_tabs():
    with gr.Blocks() as demo:
        gr.Markdown("## 文生视频和图生视频工具")
        
        # 添加清理内存按钮
        with gr.Row():
            clear_button = gr.Button("清理内存")
            clear_output = gr.Textbox(label="清理状态")
        clear_button.click(fn=clear_memory, inputs=[], outputs=clear_output)
        
        with gr.Tab("文生视频"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_text = gr.Textbox(label="提示词", lines=5, max_lines=10, value="A cat and a dog baking a cake together in a kitchen...")
                    negative_prompt_text = gr.Textbox(label="负提示词", lines=5, max_lines=10, value="Bright tones, overexposed...")
                    model_choice_text = gr.Dropdown(choices=text_to_video_models, label="选择模型", value=text_to_video_models[0])
                    source_choice_text = gr.Dropdown(choices=["Hugging Face", "ModelScope"], label="下载源", value="Hugging Face")
                    height_slider_text = gr.Slider(minimum=256, maximum=1080, step=16, label="高度", value=720)
                    width_slider_text = gr.Slider(minimum=256, maximum=1920, step=16, label="宽度", value=1280)
                    num_frames_slider_text = gr.Slider(minimum=16, maximum=128, step=1, label="帧数", value=81)
                    guidance_scale_slider_text = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, label="引导系数", value=5.0)
                    flow_shift_slider_text = gr.Slider(minimum=3.0, maximum=5.0, step=0.1, label="Flow Shift", value=5.0)
                    fps_slider_text = gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=16)
                    generate_button_text = gr.Button("生成视频")
                with gr.Column(scale=1):
                    output_video_text = gr.Video(label="输出视频")

            generate_button_text.click(
                fn=text_to_video,
                inputs=[prompt_text, negative_prompt_text, model_choice_text, source_choice_text, height_slider_text,
                        width_slider_text, num_frames_slider_text, guidance_scale_slider_text, flow_shift_slider_text, fps_slider_text],
                outputs=output_video_text
            )

        with gr.Tab("图生视频"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_image = gr.Image(type="pil", label="输入图片")
                    prompt_image = gr.Textbox(label="提示词", lines=5, max_lines=10, value="An astronaut hatching from an egg...")
                    negative_prompt_image = gr.Textbox(label="负提示词", lines=5, max_lines=10, value="Bright tones, overexposed...")
                    model_choice_image = gr.Dropdown(choices=image_to_video_models, label="选择模型", value=image_to_video_models[0])
                    source_choice_image = gr.Dropdown(choices=["Hugging Face", "ModelScope"], label="下载源", value="Hugging Face")
                    num_frames_slider_image = gr.Slider(minimum=16, maximum=128, step=1, label="帧数", value=81)
                    guidance_scale_slider_image = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, label="引导系数", value=5.0)
                    fps_slider_image = gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=16)
                    generate_button_image = gr.Button("生成视频")
                with gr.Column(scale=1):
                    output_video_image = gr.Video(label="输出视频")

            generate_button_image.click(
                fn=image_to_video,
                inputs=[input_image, prompt_image, negative_prompt_image, model_choice_image, source_choice_image,
                        num_frames_slider_image, guidance_scale_slider_image, fps_slider_image],
                outputs=output_video_image
            )

    return [(demo, "wan video", "video_generation_tab")]

try:
    from modules import script_callbacks
    script_callbacks.on_ui_tabs(on_ui_tabs)
except ImportError:
    print("未找到 script_callbacks 模块，可能不是在 Stable Diffusion WebUI 环境中运行。")