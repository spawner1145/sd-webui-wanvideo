# scripts/wan_video.py
import gradio as gr
import torch
import time
import psutil
import os
import numpy as np
from diffsynth import ModelManager, WanVideoPipeline, save_video
from modelscope import snapshot_download as ms_snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download
from PIL import Image
from tqdm import tqdm
import random

# 检查是否在 WebUI 环境中运行
try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'opts': type('Opts', (), {'outdir_samples': '', 'outdir_txt2img_samples': ''})})()

pipe = None  # 全局变量，用于存储模型管道

# 获取硬件信息
def get_hardware_info():
    info = ""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            info += f"GPU: {gpu_name}\nVRAM: {total_vram}GB\n"
        else:
            info += "GPU: 不可用\n"
        info += f"CPU: {psutil.cpu_count(logical=False)} 物理核心 / {psutil.cpu_count(logical=True)} 逻辑核心\n"
        info += f"内存: {psutil.virtual_memory().total // (1024 ** 3)}GB\n"
    except Exception as e:
        info += f"硬件信息获取失败: {str(e)}"
    return info

# 加载模型
def load_models(model_name, download_source):
    global pipe
    model_dir = os.path.join("extensions", "wan-video-generator", "models", model_name) if IN_WEBUI else os.path.join(os.path.dirname(__file__), "..", "models", model_name)
    os.makedirs(model_dir, exist_ok=True)

    model_files = [
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth"
    ]
    
    if not all(os.path.exists(os.path.join(model_dir, file)) for file in model_files):
        try:
            if download_source == "ModelScope":
                ms_snapshot_download(f"Wan-AI/{model_name}", local_dir=model_dir)
            elif download_source == "Hugging Face":
                hf_snapshot_download(repo_id=f"Wan-AI/{model_name}", local_dir=model_dir)
        except Exception as e:
            raise Exception(f"模型下载失败: {str(e)}")

    if not all(os.path.exists(os.path.join(model_dir, file)) for file in model_files):
        raise Exception("模型文件下载不完整")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_manager = ModelManager(device=device)
    
    model_list = [
        os.path.join(model_dir, "diffusion_pytorch_model.safetensors"),
        os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(model_dir, "Wan2.1_VAE.pth"),
    ]

    model_manager.load_models(model_list, torch_dtype=torch.bfloat16)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    
    if device == "cuda":
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
    
    pipe.hardware_info = get_hardware_info()
    pipe.model_name = model_name
    return pipe

# 自适应图片分辨率
def adaptive_resolution(image):
    if image is None:
        return 512, 512
    try:
        img = Image.open(image)
        width, height = img.size
        return width, height
    except Exception as e:
        return 512, 512

# 更新TeaCache建议值
def update_teacache_suggestion(model_name):
    suggestions = {
        "Wan2.1-T2V-1.3B": (0.07, "建议值: Low=0.05, Medium=0.07, High=0.08"),
        "Wan2.1-T2V-14B": (0.15, "建议值: Low=0.14, Medium=0.15, High=0.20"),
        "Wan2.1-I2V-14B-480P": (0.19, "建议值: Low=0.13, Medium=0.19, High=0.26"),
        "Wan2.1-I2V-14B-720P": (0.20, "建议值: Low=0.18, Medium=0.20, High=0.30")
    }
    default_value, suggestion = suggestions.get(model_name, (0.05, "未知模型，默认值 0.05"))
    return default_value, suggestion

# 生成文生视频
def generate_t2v(prompt, negative_prompt, num_inference_steps, seed, height, width, 
                 num_frames, cfg_scale, sigma_shift, tea_cache_l1_thresh, download_source, 
                 model_name, fps, denoising_strength=1.0, rand_device="cpu", tiled=True, 
                 tile_size=(30, 52), tile_stride=(15, 26), progress_bar_cmd=tqdm, progress_bar_st=None):
    global pipe
    model_dir = os.path.join("extensions", "wan-video-generator", "models", model_name) if IN_WEBUI else os.path.join(os.path.dirname(__file__), "..", "models", model_name)
    
    all_files_exist = all(os.path.exists(os.path.join(model_dir, file)) for file in [
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth"
    ])
    
    if pipe is None or not all_files_exist or getattr(pipe, "model_name", None) != model_name:
        pipe = load_models(model_name, download_source)
    
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        # 处理随机种子
        actual_seed = int(seed)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)

        params = {
            "prompt": prompt or "默认提示词",
            "negative_prompt": negative_prompt or "",
            "input_image": None,
            "input_video": None,
            "denoising_strength": float(denoising_strength),
            "seed": actual_seed,  # 使用实际种子
            "rand_device": rand_device,
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "cfg_scale": float(cfg_scale),
            "num_inference_steps": int(num_inference_steps),
            "sigma_shift": float(sigma_shift),
            "tiled": bool(tiled),
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "tea_cache_l1_thresh": float(tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None,
            "tea_cache_model_id": model_name,
            "progress_bar_cmd": progress_bar_cmd,
            "progress_bar_st": progress_bar_st
        }
        
        video = pipe(**params)

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"wan_video_t2v_{int(time.time())}.mp4")
        
        disk_space = psutil.disk_usage(output_dir).free // (1024 ** 3)
        if disk_space < 1:
            raise Exception("磁盘空间不足，请清理后再试")
        
        save_video(video, output_path, fps=int(fps), quality=5)

        mem_info = ""
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() // (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 3)
            mem_info = f"显存使用：{mem_used}GB / 峰值保留：{mem_reserved}GB\n"

        time_cost = time.time() - start_time
        info = f"""{pipe.hardware_info}
生成信息：
- 分辨率：{params['width']}x{params['height']}
- 总帧数：{params['num_frames']}
- 推理步数：{params['num_inference_steps']}
- 随机种子：{actual_seed} {'(随机生成)' if seed == -1 else ''}
- 总耗时：{time_cost:.2f}秒
- 帧率：{fps} FPS
- 视频时长：{params['num_frames'] / int(fps):.1f}秒
{mem_info}
- 模型版本：{model_name}
- 下载源：{download_source}
- 使用Tiled：{'是' if params['tiled'] else '否'}
- TeaCache L1阈值：{params['tea_cache_l1_thresh'] if params['tea_cache_l1_thresh'] is not None else '未使用'}
"""
        return output_path, info
    except Exception as e:
        return None, f"生成失败: {str(e)}"

# 生成图生视频
def generate_i2v(image, prompt, negative_prompt, num_inference_steps, seed, height, width, 
                num_frames, cfg_scale, sigma_shift, tea_cache_l1_thresh, download_source, 
                model_name, fps, denoising_strength=1.0, rand_device="cpu", tiled=True, 
                tile_size=(30, 52), tile_stride=(15, 26), progress_bar_cmd=tqdm, progress_bar_st=None):
    global pipe
    model_dir = os.path.join("extensions", "wan-video-generator", "models", model_name) if IN_WEBUI else os.path.join(os.path.dirname(__file__), "..", "models", model_name)
    
    all_files_exist = all(os.path.exists(os.path.join(model_dir, file)) for file in [
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth"
    ])
    
    if pipe is None or not all_files_exist or getattr(pipe, "model_name", None) != model_name:
        pipe = load_models(model_name, download_source)
    
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        if image is None:
            raise ValueError("请上传初始图片")
        img = Image.open(image).convert("RGB")

        # 处理随机种子
        actual_seed = int(seed)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)

        params = {
            "prompt": prompt or "默认提示词",
            "negative_prompt": negative_prompt or "",
            "input_image": img,
            "input_video": None,
            "denoising_strength": float(denoising_strength),
            "seed": actual_seed,  # 使用实际种子
            "rand_device": rand_device,
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "cfg_scale": float(cfg_scale),
            "num_inference_steps": int(num_inference_steps),
            "sigma_shift": float(sigma_shift),
            "tiled": bool(tiled),
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "tea_cache_l1_thresh": float(tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None,
            "tea_cache_model_id": model_name,
            "progress_bar_cmd": progress_bar_cmd,
            "progress_bar_st": progress_bar_st
        }
        
        video = pipe(**params)

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"wan_video_i2v_{int(time.time())}.mp4")
        
        disk_space = psutil.disk_usage(output_dir).free // (1024 ** 3)
        if disk_space < 1:
            raise Exception("磁盘空间不足，请清理后再试")
        
        save_video(video, output_path, fps=int(fps), quality=5)

        mem_info = ""
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() // (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 3)
            mem_info = f"显存使用：{mem_used}GB / 峰值保留：{mem_reserved}GB\n"

        time_cost = time.time() - start_time
        info = f"""{pipe.hardware_info}
生成信息：
- 分辨率：{params['width']}x{params['height']}
- 总帧数：{params['num_frames']}
- 推理步数：{params['num_inference_steps']}
- 随机种子：{actual_seed} {'(随机生成)' if seed == -1 else ''}
- 总耗时：{time_cost:.2f}秒
- 帧率：{fps} FPS
- 视频时长：{params['num_frames'] / int(fps):.1f}秒
{mem_info}
- 模型版本：{model_name}
- 下载源：{download_source}
- 使用Tiled：{'是' if params['tiled'] else '否'}
- TeaCache L1阈值：{params['tea_cache_l1_thresh'] if params['tea_cache_l1_thresh'] is not None else '未使用'}
"""
        return output_path, info
    except Exception as e:
        return None, f"生成失败: {str(e)}"

# 创建界面
def create_wan_video_tab():
    with gr.Blocks(analytics_enabled=False) as wan_interface:
        gr.Markdown("## Wan2.1 文本/图生成视频")
        
        with gr.Tabs():
            with gr.Tab("Text-to-Video"):
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(label="正向提示词", lines=3, placeholder="输入描述视频内容的提示词")
                        negative_prompt = gr.Textbox(
                            label="负向提示词",
                            lines=3,
                            value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
                        )
                        
                        with gr.Accordion("基础参数", open=True):
                            width = gr.Slider(label="宽度", minimum=256, maximum=1920, value=832, step=8)
                            height = gr.Slider(label="高度", minimum=256, maximum=1080, value=480, step=8)
                            num_frames = gr.Number(label="帧数", value=81, minimum=1, precision=0)
                            fps = gr.Slider(label="输出帧率 (FPS)", minimum=1, maximum=60, value=15, step=1)

                        with gr.Accordion("高级参数", open=False):
                            num_inference_steps = gr.Slider(label="推理步数", minimum=20, maximum=100, value=50, step=1)
                            cfg_scale = gr.Number(label="CFG Scale", value=5.0)
                            sigma_shift = gr.Number(label="Sigma Shift", value=5.0)
                            seed = gr.Number(label="随机种子 (-1为随机)", value=-1, precision=0)
                            denoising_strength = gr.Slider(label="降噪强度", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                            download_source = gr.Dropdown(
                                label="模型下载源",
                                choices=["ModelScope", "Hugging Face"],
                                value="ModelScope"
                            )
                            model_name = gr.Dropdown(
                                label="选择模型",
                                choices=["Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B"],
                                value="Wan2.1-T2V-1.3B"
                            )
                        
                        with gr.Accordion("TeaCache 参数", open=False):
                            tea_cache_l1_thresh = gr.Number(label="TeaCache L1阈值 (越大越快但质量下降)", value=0.07)
                            with gr.Accordion("TeaCache 建议值", open=False):
                                tea_cache_suggestion = gr.Textbox(label="建议值", value="根据模型选择自动更新", interactive=False)

                        generate_btn = gr.Button("生成视频")

                    with gr.Column():
                        output_video = gr.Video(label="生成结果")
                        info_output = gr.Textbox(label="系统信息", interactive=False, lines=16)

                model_name.change(
                    fn=update_teacache_suggestion,
                    inputs=[model_name],
                    outputs=[tea_cache_l1_thresh, tea_cache_suggestion]
                )

                generate_btn.click(
                    fn=generate_t2v,
                    inputs=[
                        prompt,
                        negative_prompt,
                        num_inference_steps,
                        seed,
                        height,
                        width,
                        num_frames,
                        cfg_scale,
                        sigma_shift,
                        tea_cache_l1_thresh,
                        download_source,
                        model_name,
                        fps,
                        denoising_strength
                    ],
                    outputs=[output_video, info_output]
                )

            with gr.Tab("Image-to-Video"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="上传初始图片", type="filepath")
                        adapt_resolution_btn = gr.Button("自适应图片分辨率")
                        prompt_i2v = gr.Textbox(label="正向提示词", lines=3, placeholder="输入描述视频内容的提示词")
                        negative_prompt_i2v = gr.Textbox(
                            label="负向提示词",
                            lines=3,
                            value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
                        )
                        
                        with gr.Accordion("基础参数", open=True):
                            width_i2v = gr.Slider(label="宽度", minimum=256, maximum=1920, value=832, step=8)
                            height_i2v = gr.Slider(label="高度", minimum=256, maximum=1080, value=480, step=8)
                            num_frames_i2v = gr.Number(label="帧数", value=81, minimum=1, precision=0)
                            fps_i2v = gr.Slider(label="输出帧率 (FPS)", minimum=1, maximum=60, value=15, step=1)

                        with gr.Accordion("高级参数", open=False):
                            num_inference_steps_i2v = gr.Slider(label="推理步数", minimum=20, maximum=100, value=50, step=1)
                            cfg_scale_i2v = gr.Number(label="CFG Scale", value=5.0)
                            sigma_shift_i2v = gr.Number(label="Sigma Shift", value=5.0)
                            seed_i2v = gr.Number(label="随机种子 (-1为随机)", value=-1, precision=0)
                            denoising_strength_i2v = gr.Slider(label="降噪强度", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                            download_source_i2v = gr.Dropdown(
                                label="模型下载源",
                                choices=["ModelScope", "Hugging Face"],
                                value="ModelScope"
                            )
                            model_name_i2v = gr.Dropdown(
                                label="选择模型",
                                choices=["Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-720P"],
                                value="Wan2.1-I2V-14B-480P"
                            )
                        
                        with gr.Accordion("TeaCache 参数", open=False):
                            tea_cache_l1_thresh_i2v = gr.Number(label="TeaCache L1阈值 (越大越快但质量下降)", value=0.19)
                            with gr.Accordion("TeaCache 建议值", open=False):
                                tea_cache_suggestion_i2v = gr.Textbox(label="建议值", value="根据模型选择自动更新", interactive=False)

                        generate_i2v_btn = gr.Button("生成视频")

                    with gr.Column():
                        output_video_i2v = gr.Video(label="生成结果")
                        info_output_i2v = gr.Textbox(label="系统信息", interactive=False, lines=16)

                adapt_resolution_btn.click(
                    fn=adaptive_resolution,
                    inputs=[image_input],
                    outputs=[width_i2v, height_i2v]
                )

                model_name_i2v.change(
                    fn=update_teacache_suggestion,
                    inputs=[model_name_i2v],
                    outputs=[tea_cache_l1_thresh_i2v, tea_cache_suggestion_i2v]
                )

                generate_i2v_btn.click(
                    fn=generate_i2v,
                    inputs=[
                        image_input,
                        prompt_i2v,
                        negative_prompt_i2v,
                        num_inference_steps_i2v,
                        seed_i2v,
                        height_i2v,
                        width_i2v,
                        num_frames_i2v,
                        cfg_scale_i2v,
                        sigma_shift_i2v,
                        tea_cache_l1_thresh_i2v,
                        download_source_i2v,
                        model_name_i2v,
                        fps_i2v,
                        denoising_strength_i2v
                    ],
                    outputs=[output_video_i2v, info_output_i2v]
                )
    
    return wan_interface

if IN_WEBUI:
    script_callbacks.on_ui_tabs(lambda: [(create_wan_video_tab(), "Wan Video", "wan_video_tab")])
else:
    if __name__ == "__main__":
        interface = create_wan_video_tab()
        interface.launch(
            allowed_paths=["outputs"]
        )
