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
import time
import psutil

models_dir = Path("models/wan")
outputs_dir = Path("outputs/videos")
models_dir.mkdir(parents=True, exist_ok=True)
outputs_dir.mkdir(parents=True, exist_ok=True)

text_to_video_models = [
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",  # 默认 1.3B 模型
    "Wan-AI/Wan2.1-T2V-14B-Diffusers"
]

image_to_video_models = [
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",  # 默认 480P 模型
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
]

def clear_memory(aggressive=False):
    """
    增强型内存清理函数
    - aggressive=True 时执行更彻底的清理
    """
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if aggressive:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.ipc_collect()

    gc.collect()
    if aggressive:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    del obj
            except:
                pass
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / 1024**2  # MB
        if ram_usage > 8000:
            gc.collect()
            torch.cuda.empty_cache()

    return f"内存已清理{'（激进模式）' if aggressive else ''}"

def download_model(model_id, source):
    try:
        model_path = models_dir / model_id.split("/")[-1]
        if model_path.exists():
            return str(model_path)
        for old_model in models_dir.iterdir():
            if old_model.is_dir() and old_model != model_path:
                import shutil
                shutil.rmtree(old_model)
        if source == "Hugging Face":
            snapshot_download(repo_id=model_id, local_dir=model_path)
        elif source == "ModelScope":
            modelscope_snapshot_download(model_id, cache_dir=models_dir)
        return str(model_path)
    except Exception as e:
        raise gr.Error(f"模型下载失败: {str(e)}")

def text_to_video(
    prompt: str,
    negative_prompt: str = "",
    model_id: str = text_to_video_models[0],
    source: str = "Hugging Face",
    height: int = 256,
    width: int = 256,
    num_frames: int = 16,
    guidance_scale: float = 5.0,
    flow_shift: float = 5.0,
    fps: int = 8,
    num_inference_steps: int = 20,
    use_sequential_offload: bool = False,
    use_fp16: bool = True  # 新增选项，默认 fp16
):
    try:
        clear_memory(aggressive=True)
        
        model_path = download_model(model_id, source)
        dtype = torch.float16 if use_fp16 else torch.float32
        
        text_vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=500,
            flow_shift=flow_shift
        )
        text_pipeline = WanPipeline.from_pretrained(model_path, vae=text_vae, torch_dtype=dtype)
        text_pipeline.scheduler = scheduler
        text_pipeline.enable_gradient_checkpointing()
        if use_sequential_offload:
            text_pipeline.enable_sequential_cpu_offload()
            print("启用顺序 CPU 卸载，显存需求极低但速度较慢")
        else:
            text_pipeline.enable_model_cpu_offload()
            print("启用标准 CPU 卸载，平衡速度与显存")
        try:
            text_pipeline.enable_xformers_memory_efficient_attention()
            print("xformers 已启用于 text_pipeline")
        except AttributeError:
            print("text_pipeline 不支持 xformers，可能需要手动集成")

        output = text_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).frames[0]
        
        timestamp = int(time.time())
        output_path = outputs_dir / f"text_to_video_{timestamp}.mp4"
        export_to_video(output, str(output_path), fps=fps)
        
        del text_pipeline
        del text_vae
        clear_memory(aggressive=True)
        return str(output_path)
    except Exception as e:
        clear_memory(aggressive=True)
        raise gr.Error(f"视频生成失败: {str(e)}")

def image_to_video(
    image,
    prompt: str,
    negative_prompt: str = "",
    model_id: str = image_to_video_models[0],
    source: str = "Hugging Face",
    height: int = 256,
    width: int = 256,
    num_frames: int = 16,
    guidance_scale: float = 5.0,
    fps: int = 8,
    num_inference_steps: int = 20,
    use_sequential_offload: bool = False,
    use_fp16: bool = True  # 新增选项，默认 fp16
):
    try:
        clear_memory(aggressive=True)
        
        if image is None:
            raise ValueError("请提供输入图片")
            
        model_path = download_model(model_id, source)
        dtype = torch.float16 if use_fp16 else torch.float32
        
        image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=dtype)
        image_vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
        image_pipeline = WanImageToVideoPipeline.from_pretrained(
            model_path, 
            vae=image_vae, 
            image_encoder=image_encoder, 
            torch_dtype=dtype
        )
        image_pipeline.enable_gradient_checkpointing()
        if use_sequential_offload:
            image_pipeline.enable_sequential_cpu_offload()
            print("启用顺序 CPU 卸载，显存需求极低但速度较慢")
        else:
            image_pipeline.enable_model_cpu_offload()
            print("启用标准 CPU 卸载，平衡速度与显存")
        try:
            image_pipeline.enable_xformers_memory_efficient_attention()
            print("xformers 已启用于 image_pipeline")
        except AttributeError:
            print("image_pipeline 不支持 xformers，可能需要手动集成")

        max_area = 720 * 1280 if "720P" in model_id else 480 * 854
        if height * width > max_area:
            aspect_ratio = image.height / image.width
            height = round(np.sqrt(max_area * aspect_ratio))
            width = round(np.sqrt(max_area / aspect_ratio))
        
        mod_value = image_pipeline.vae_scale_factor_spatial * image_pipeline.transformer.config.patch_size[1]
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        image = image.resize((width, height))

        output = image_pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).frames[0]
        
        timestamp = int(time.time())
        output_path = outputs_dir / f"image_to_video_{timestamp}.mp4"
        export_to_video(output, str(output_path), fps=fps)
        
        del image_pipeline
        del image_vae
        del image_encoder
        clear_memory(aggressive=True)
        return str(output_path)
    except Exception as e:
        clear_memory(aggressive=True)
        raise gr.Error(f"视频生成失败: {str(e)}")

def on_ui_tabs():
    with gr.Blocks() as demo:
        gr.Markdown("## 文生视频和图生视频工具（优化版：适配 8GB 显存和 16GB RAM）")
        
        with gr.Row():
            clear_button = gr.Button("清理内存")
            clear_output = gr.Textbox(label="清理状态")
        clear_button.click(fn=lambda: clear_memory(aggressive=True), inputs=[], outputs=clear_output)
        
        with gr.Tab("文生视频"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_text = gr.Textbox(label="提示词", lines=5, max_lines=10, value="A cat and a dog baking a cake together in a kitchen...")
                    negative_prompt_text = gr.Textbox(label="负提示词", lines=5, max_lines=10, value="", placeholder="可选，留空即可")
                    model_choice_text = gr.Dropdown(choices=text_to_video_models, label="选择模型", value=text_to_video_models[0])
                    source_choice_text = gr.Dropdown(choices=["Hugging Face", "ModelScope"], label="下载源", value="Hugging Face")
                    height_slider_text = gr.Slider(minimum=256, maximum=512, step=16, label="高度", value=256)
                    width_slider_text = gr.Slider(minimum=256, maximum=768, step=16, label="宽度", value=256)
                    num_frames_slider_text = gr.Slider(minimum=16, maximum=64, step=1, label="帧数", value=16)
                    guidance_scale_slider_text = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, label="引导系数", value=5.0)
                    flow_shift_slider_text = gr.Slider(minimum=3.0, maximum=5.0, step=0.1, label="Flow Shift", value=5.0)
                    fps_slider_text = gr.Slider(minimum=1, maximum=30, step=1, label="FPS", value=8)
                    num_inference_steps_slider_text = gr.Slider(minimum=10, maximum=50, step=1, label="推理步数", value=20)
                    sequential_offload_checkbox_text = gr.Checkbox(label="使用顺序 CPU 卸载（显存极低但速度慢）", value=False)
                    fp16_checkbox_text = gr.Checkbox(label="使用 FP16（显存低，速度快）", value=True)
                    generate_button_text = gr.Button("生成视频")
                with gr.Column(scale=1):
                    output_video_text = gr.Video(label="输出视频")

            generate_button_text.click(
                fn=text_to_video,
                inputs=[prompt_text, negative_prompt_text, model_choice_text, source_choice_text, 
                        height_slider_text, width_slider_text, num_frames_slider_text, 
                        guidance_scale_slider_text, flow_shift_slider_text, fps_slider_text, 
                        num_inference_steps_slider_text, sequential_offload_checkbox_text, fp16_checkbox_text],
                outputs=output_video_text
            )

        with gr.Tab("图生视频"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_image = gr.Image(type="pil", label="输入图片")
                    prompt_image = gr.Textbox(label="提示词", lines=5, max_lines=10, value="An astronaut hatching from an egg...")
                    negative_prompt_image = gr.Textbox(label="负提示词", lines=5, max_lines=10, value="", placeholder="可选，留空即可")
                    model_choice_image = gr.Dropdown(choices=image_to_video_models, label="选择模型", value=image_to_video_models[0])
                    source_choice_image = gr.Dropdown(choices=["Hugging Face", "ModelScope"], label="下载源", value="Hugging Face")
                    height_slider_image = gr.Slider(minimum=256, maximum=512, step=16, label="高度", value=256)
                    width_slider_image = gr.Slider(minimum=256, maximum=768, step=16, label="宽度", value=256)
                    num_frames_slider_image = gr.Slider(minimum=16, maximum=64, step=1, label="帧数", value=16)
                    guidance_scale_slider_image = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, label="引导系数", value=5.0)
                    fps_slider_image = gr.Slider(minimum=1, maximum=30, step=1, label="FPS", value=8)
                    num_inference_steps_slider_image = gr.Slider(minimum=10, maximum=50, step=1, label="推理步数", value=20)
                    sequential_offload_checkbox_image = gr.Checkbox(label="使用顺序 CPU 卸载（显存极低但速度慢）", value=False)
                    fp16_checkbox_image = gr.Checkbox(label="使用 FP16（显存低，速度快）", value=True)
                    generate_button_image = gr.Button("生成视频")
                with gr.Column(scale=1):
                    output_video_image = gr.Video(label="输出视频")

            generate_button_image.click(
                fn=image_to_video,
                inputs=[input_image, prompt_image, negative_prompt_image, model_choice_image, source_choice_image,
                        height_slider_image, width_slider_image, num_frames_slider_image, 
                        guidance_scale_slider_image, fps_slider_image, num_inference_steps_slider_image, 
                        sequential_offload_checkbox_image, fp16_checkbox_image],
                outputs=output_video_image
            )

    return [(demo, "wan video", "video_generation_tab")]

try:
    from modules import script_callbacks
    script_callbacks.on_ui_tabs(on_ui_tabs)
except ImportError:
    print("未找到 script_callbacks 模块，可能不是在 Stable Diffusion WebUI 环境中运行。")