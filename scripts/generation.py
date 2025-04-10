# F:\sd-reforge\webui\extensions\sd-webui-wanvideo\scripts\generation.py
import gradio as gr
import torch
import time
import psutil
import os
import numpy as np
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from PIL import Image
from tqdm import tqdm
import random
import logging
import re
from modelscope import snapshot_download, dataset_snapshot_download

# 设置日志
logging.basicConfig(level=logging.INFO)

# 检查是否在 WebUI 环境中运行
try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'opts': type('Opts', (), {'outdir_samples': '', 'outdir_txt2img_samples': ''})})()

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

# 获取指定目录中的模型文件列表
def get_model_files(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return ["无模型文件"]
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and not f.endswith('.txt') and not f.endswith('.json')
    ]
    return files if files else ["无模型文件"]

# 从提示词中提取 LoRA 信息
def extract_lora_from_prompt(prompt):
    lora_pattern = r"<lora:([^:]+):([\d\.]+)>"
    matches = re.findall(lora_pattern, prompt)
    loras = [(name, float(weight)) for name, weight in matches]
    cleaned_prompt = re.sub(lora_pattern, "", prompt).strip()
    return loras, cleaned_prompt

# 加载模型和 LoRA
def load_models(dit_models, t5_model, vae_model, image_encoder_model=None, lora_prompt="", 
                torch_dtype="bfloat16", image_encoder_torch_dtype="float32", use_usp=False, 
                num_persistent_param_in_dit=None):
    # 定义模型目录
    base_dir = "models/wan2.1"
    dit_dir = os.path.join(base_dir, "dit")
    t5_dir = os.path.join(base_dir, "t5")
    vae_dir = os.path.join(base_dir, "vae")
    lora_dir = os.path.join(base_dir, "lora")
    image_encoder_dir = os.path.join(base_dir, "image_encoder") if image_encoder_model else None
    
    # 自动创建目录
    os.makedirs(dit_dir, exist_ok=True)
    os.makedirs(t5_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)
    if image_encoder_dir:
        os.makedirs(image_encoder_dir, exist_ok=True)
    
    # 记录目录信息以便调试
    logging.info(f"DIT 模型目录: {os.path.abspath(dit_dir)}")
    logging.info(f"T5 模型目录: {os.path.abspath(t5_dir)}")
    logging.info(f"VAE 模型目录: {os.path.abspath(vae_dir)}")
    logging.info(f"LoRA 模型目录: {os.path.abspath(lora_dir)}")
    if image_encoder_dir:
        logging.info(f"Image Encoder 模型目录: {os.path.abspath(image_encoder_dir)}")
    
    # 检查模型文件是否存在
    if not dit_models or "无模型文件" in dit_models or t5_model == "无模型文件" or vae_model == "无模型文件":
        raise Exception("请确保所有模型文件夹中都有有效的模型文件：DIT、T5 和 VAE 模型不可为空")
    if image_encoder_model in ["无模型文件", "无"]:
        image_encoder_model = None
    
    # 将多个 DIT 模型文件视为一个整体
    dit_model_paths = [os.path.join(dit_dir, dit_model) for dit_model in dit_models if dit_model != "无模型文件"]
    if not dit_model_paths:
        raise Exception("未选择有效的 DIT 模型文件")
    
    # 组织 model_list，DIT 模型作为一个嵌套列表
    model_list = [
        dit_model_paths,  # 多个 DIT 文件合并加载
        os.path.join(t5_dir, t5_model),
        os.path.join(vae_dir, vae_model)
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 支持 FP16、BF16 和 FP8
    if torch_dtype == "float16":
        torch_dtype = torch.float16
    elif torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float8_e4m3fn
    
    # Image Encoder 的数据类型支持
    if image_encoder_torch_dtype == "float16":
        image_encoder_torch_dtype = torch.float16
    elif image_encoder_torch_dtype == "float32":
        image_encoder_torch_dtype = torch.float32
    else:
        image_encoder_torch_dtype = torch.bfloat16
    
    model_manager = ModelManager(device="cpu", torch_dtype=torch_dtype)
    
    # 检查文件路径
    for item in model_list:
        if isinstance(item, list):
            for path in item:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"DIT 模型文件 {path} 不存在，请检查路径")
        elif not os.path.exists(item):
            raise FileNotFoundError(f"模型文件 {item} 不存在，请检查路径")
    
    # 加载 Image Encoder（若存在）
    if image_encoder_model:
        image_encoder_path = os.path.join(image_encoder_dir, image_encoder_model)
        if not os.path.exists(image_encoder_path):
            raise FileNotFoundError(f"Image Encoder 文件 {image_encoder_path} 不存在，请检查路径")
        logging.info(f"加载 Image Encoder: {image_encoder_path} (使用 {image_encoder_torch_dtype})")
        model_manager.load_models([image_encoder_path], torch_dtype=image_encoder_torch_dtype)
        model_list.insert(0, image_encoder_path)
    
    # 加载基础模型
    logging.info(f"开始加载基础模型: {model_list} (使用 {torch_dtype})")
    model_manager.load_models(model_list, torch_dtype=torch_dtype)
    logging.info(f"基础模型加载完成: {model_manager.model_name if model_manager.model_name else '未识别到模型'}")
    
    # 从提示词中提取 LoRA 信息并加载
    loras, _ = extract_lora_from_prompt(lora_prompt)
    loaded_loras = {}
    if loras:
        for lora_name, lora_weight in loras:
            lora_path = os.path.join(lora_dir, lora_name)
            if not os.path.exists(lora_path):
                logging.warning(f"LoRA 文件 {lora_path} 不存在，跳过加载")
                continue
            logging.info(f"加载 LoRA: {lora_path} (alpha={lora_weight})")
            model_manager.load_lora(lora_path, lora_alpha=lora_weight)
            loaded_loras[lora_name] = lora_weight
    
    # 检查 USP 环境
    if use_usp and not torch.distributed.is_initialized():
        logging.warning("USP 启用失败：分布式环境未初始化，将禁用 USP")
        use_usp = False
    
    # 创建管道
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch_dtype, device=device, use_usp=use_usp)
    if device == "cuda":
        pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)
    
    # 设置管道信息
    pipe.hardware_info = get_hardware_info()
    pipe.model_name = f"DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}" + (f", Image Encoder: {image_encoder_model}" if image_encoder_model else "")
    pipe.lora_info = ", ".join([f"{name} ({weight})" for name, weight in loaded_loras.items()]) if loaded_loras else "无"
    pipe.torch_dtype_info = f"DIT/T5/VAE: {torch_dtype}, Image Encoder: {image_encoder_torch_dtype if image_encoder_model else '未使用'}"
    pipe.num_persistent_param_in_dit = num_persistent_param_in_dit  # 新增属性以便显示
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

# 生成文生视频
def generate_t2v(prompt, negative_prompt, num_inference_steps, seed, height, width, 
                 num_frames, cfg_scale, sigma_shift, tea_cache_l1_thresh, tea_cache_model_id, 
                 dit_models, t5_model, vae_model, image_encoder_model, fps, denoising_strength, 
                 rand_device, tiled, tile_size_x, tile_size_y, tile_stride_x, tile_stride_y, 
                 torch_dtype, image_encoder_torch_dtype, use_usp, enable_num_persistent=None, 
                 num_persistent_param_in_dit=None, progress_bar_cmd=tqdm, progress_bar_st=None):
    # 处理 num_persistent_param_in_dit 的开关逻辑
    if not enable_num_persistent:
        num_persistent_param_in_dit = None
    
    # 每次生成都创建新的 pipe
    pipe = load_models(dit_models, t5_model, vae_model, image_encoder_model, prompt, torch_dtype, 
                      image_encoder_torch_dtype, use_usp, num_persistent_param_in_dit)
    
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        # 处理随机种子
        actual_seed = int(seed)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)

        # 从提示词中提取 LoRA 并清理提示词
        _, cleaned_prompt = extract_lora_from_prompt(prompt)

        # 调用 WanVideoPipeline
        frames = pipe(
            prompt=cleaned_prompt or "默认提示词",
            negative_prompt=negative_prompt or "",
            input_image=None,
            input_video=None,
            denoising_strength=float(denoising_strength),
            seed=actual_seed,
            rand_device=rand_device,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            cfg_scale=float(cfg_scale),
            num_inference_steps=int(num_inference_steps),
            sigma_shift=float(sigma_shift),
            tiled=bool(tiled),
            tile_size=(int(tile_size_x), int(tile_size_y)),
            tile_stride=(int(tile_stride_x), int(tile_stride_y)),
            tea_cache_l1_thresh=float(tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None,
            tea_cache_model_id=tea_cache_model_id,
            progress_bar_cmd=progress_bar_cmd,
            progress_bar_st=progress_bar_st
        )

        output_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples or "outputs"
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"输出目录 {output_dir} 不可写，请检查权限")
        output_path = os.path.join(output_dir, f"wan_video_t2v_{int(time.time())}.mp4")
        
        disk_space = psutil.disk_usage(output_dir).free // (1024 ** 3)
        if disk_space < 1:
            raise Exception("磁盘空间不足，请清理后再试")
        
        save_video(frames, output_path, fps=int(fps), quality=5)

        mem_info = ""
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() // (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 3)
            mem_info = f"显存使用：{mem_used}GB / 峰值保留：{mem_reserved}GB\n"

        time_cost = time.time() - start_time
        info = f"""{pipe.hardware_info}
生成信息：
- 分辨率：{width}x{height}
- 总帧数：{num_frames}
- 推理步数：{num_inference_steps}
- 随机种子：{actual_seed} {'(随机生成)' if seed == -1 else ''}
- 总耗时：{time_cost:.2f}秒
- 帧率：{fps} FPS
- 视频时长：{num_frames / int(fps):.1f}秒
{mem_info}
- 模型版本：DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}{', Image Encoder: ' + image_encoder_model if image_encoder_model else ''}
- 使用Tiled：{'是' if tiled else '否'}
- Tile Size：({tile_size_x}, {tile_size_y})
- Tile Stride：({tile_stride_x}, {tile_stride_y})
- TeaCache L1阈值：{tea_cache_l1_thresh if tea_cache_l1_thresh is not None else '未使用'}
- TeaCache Model ID：{tea_cache_model_id}
- Torch 数据类型：{pipe.torch_dtype_info}
- 使用USP：{'是' if use_usp else '否'}
- 显存管理参数 (num_persistent_param_in_dit)：{num_persistent_param_in_dit if num_persistent_param_in_dit is not None else '未限制'}
- 已加载LoRA：{pipe.lora_info}
"""
        return output_path, info
    except Exception as e:
        return None, f"生成失败: {str(e)}"
    finally:
        del pipe  # 确保每次生成后清理 pipe

# 生成图生视频（新增 end_image 参数）
def generate_i2v(image, end_image, prompt, negative_prompt, num_inference_steps, seed, height, width, 
                num_frames, cfg_scale, sigma_shift, tea_cache_l1_thresh, tea_cache_model_id, 
                dit_models, t5_model, vae_model, image_encoder_model, fps, denoising_strength, 
                rand_device, tiled, tile_size_x, tile_size_y, tile_stride_x, tile_stride_y, 
                torch_dtype, image_encoder_torch_dtype, use_usp, enable_num_persistent=None, 
                num_persistent_param_in_dit=None, progress_bar_cmd=tqdm, progress_bar_st=None):
    # 处理 num_persistent_param_in_dit 的开关逻辑
    if not enable_num_persistent:
        num_persistent_param_in_dit = None
    
    # 每次生成都创建新的 pipe
    pipe = load_models(dit_models, t5_model, vae_model, image_encoder_model, prompt, torch_dtype, 
                      image_encoder_torch_dtype, use_usp, num_persistent_param_in_dit)
    
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        if image is None:
            raise ValueError("请上传初始图片")
        img = Image.open(image).convert("RGB")
        # 如果提供了结束图片，则加载它
        end_img = Image.open(end_image).convert("RGB") if end_image else None

        # 处理随机种子
        actual_seed = int(seed)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)

        # 从提示词中提取 LoRA 并清理提示词
        _, cleaned_prompt = extract_lora_from_prompt(prompt)

        # 调用 WanVideoPipeline，新增 end_image 参数
        frames = pipe(
            prompt=cleaned_prompt or "默认提示词",
            negative_prompt=negative_prompt or "",
            input_image=img,
            end_image=end_img,  # 新增的可选结束图片参数
            input_video=None,
            denoising_strength=float(denoising_strength),
            seed=actual_seed,
            rand_device=rand_device,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            cfg_scale=float(cfg_scale),
            num_inference_steps=int(num_inference_steps),
            sigma_shift=float(sigma_shift),
            tiled=bool(tiled),
            tile_size=(int(tile_size_x), int(tile_size_y)),
            tile_stride=(int(tile_stride_x), int(tile_stride_y)),
            tea_cache_l1_thresh=float(tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None,
            tea_cache_model_id=tea_cache_model_id,
            progress_bar_cmd=progress_bar_cmd,
            progress_bar_st=progress_bar_st
        )

        output_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples or "outputs"
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"输出目录 {output_dir} 不可写，请检查权限")
        output_path = os.path.join(output_dir, f"wan_video_i2v_{int(time.time())}.mp4")
        
        disk_space = psutil.disk_usage(output_dir).free // (1024 ** 3)
        if disk_space < 1:
            raise Exception("磁盘空间不足，请清理后再试")
        
        save_video(frames, output_path, fps=int(fps), quality=5)

        mem_info = ""
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() // (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 3)
            mem_info = f"显存使用：{mem_used}GB / 峰值保留：{mem_reserved}GB\n"

        time_cost = time.time() - start_time
        info = f"""{pipe.hardware_info}
生成信息：
- 分辨率：{width}x{height}
- 总帧数：{num_frames}
- 推理步数：{num_inference_steps}
- 随机种子：{actual_seed} {'(随机生成)' if seed == -1 else ''}
- 总耗时：{time_cost:.2f}秒
- 帧率：{fps} FPS
- 视频时长：{num_frames / int(fps):.1f}秒
{mem_info}
- 模型版本：DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}{', Image Encoder: ' + image_encoder_model if image_encoder_model else ''}
- 使用Tiled：{'是' if tiled else '否'}
- Tile Size：({tile_size_x}, {tile_size_y})
- Tile Stride：({tile_stride_x}, {tile_stride_y})
- TeaCache L1阈值：{tea_cache_l1_thresh if tea_cache_l1_thresh is not None else '未使用'}
- TeaCache Model ID：{tea_cache_model_id}
- Torch 数据类型：{pipe.torch_dtype_info}
- 使用USP：{'是' if use_usp else '否'}
- 显存管理参数 (num_persistent_param_in_dit)：{num_persistent_param_in_dit if num_persistent_param_in_dit is not None else '未限制'}
- 已加载LoRA：{pipe.lora_info}
- 是否使用结束图片：{'是' if end_image else '否'}
"""
        return output_path, info
    except Exception as e:
        return None, f"生成失败: {str(e)}"
    finally:
        del pipe  # 确保每次生成后清理 pipe

# 生成视频生视频（新增 control_video 参数）
def generate_v2v(video, control_video, prompt, negative_prompt, num_inference_steps, seed, height, width, 
                num_frames, cfg_scale, sigma_shift, dit_models, t5_model, vae_model, 
                image_encoder_model, fps, denoising_strength, rand_device, tiled, 
                tile_size_x, tile_size_y, tile_stride_x, tile_stride_y, torch_dtype, 
                image_encoder_torch_dtype, use_usp, enable_num_persistent=None, 
                num_persistent_param_in_dit=None, progress_bar_cmd=tqdm, progress_bar_st=None):
    # 处理 num_persistent_param_in_dit 的开关逻辑
    if not enable_num_persistent:
        num_persistent_param_in_dit = None
    
    # 每次生成都创建新的 pipe
    pipe = load_models(dit_models, t5_model, vae_model, image_encoder_model, prompt, torch_dtype, 
                      image_encoder_torch_dtype, use_usp, num_persistent_param_in_dit)
    
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        if video is None and control_video is None:
            raise ValueError("请至少上传初始视频或控制视频")
        video_data = VideoData(video, height=int(height), width=int(width)) if video else None
        control_video_data = VideoData(control_video, height=int(height), width=int(width)) if control_video else None

        # 处理随机种子
        actual_seed = int(seed)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)

        # 从提示词中提取 LoRA 并清理提示词
        _, cleaned_prompt = extract_lora_from_prompt(prompt)

        # 调用 WanVideoPipeline，新增 control_video 参数
        frames = pipe(
            prompt=cleaned_prompt or "默认提示词",
            negative_prompt=negative_prompt or "",
            input_image=None,
            input_video=video_data,
            control_video=control_video_data,  # 新增的可选控制视频参数
            denoising_strength=float(denoising_strength),
            seed=actual_seed,
            rand_device=rand_device,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            cfg_scale=float(cfg_scale),
            num_inference_steps=int(num_inference_steps),
            sigma_shift=float(sigma_shift),
            tiled=bool(tiled),
            tile_size=(int(tile_size_x), int(tile_size_y)),
            tile_stride=(int(tile_stride_x), int(tile_stride_y)),
            tea_cache_l1_thresh=None,  # TeaCache 不支持视频生视频
            tea_cache_model_id="",
            progress_bar_cmd=progress_bar_cmd,
            progress_bar_st=progress_bar_st
        )

        output_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples or "outputs"
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"输出目录 {output_dir} 不可写，请检查权限")
        output_path = os.path.join(output_dir, f"wan_video_v2v_{int(time.time())}.mp4")
        
        disk_space = psutil.disk_usage(output_dir).free // (1024 ** 3)
        if disk_space < 1:
            raise Exception("磁盘空间不足，请清理后再试")
        
        save_video(frames, output_path, fps=int(fps), quality=5)

        mem_info = ""
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() // (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 3)
            mem_info = f"显存使用：{mem_used}GB / 峰值保留：{mem_reserved}GB\n"

        time_cost = time.time() - start_time
        info = f"""{pipe.hardware_info}
生成信息：
- 分辨率：{width}x{height}
- 总帧数：{num_frames}
- 推理步数：{num_inference_steps}
- 随机种子：{actual_seed} {'(随机生成)' if seed == -1 else ''}
- 总耗时：{time_cost:.2f}秒
- 帧率：{fps} FPS
- 视频时长：{num_frames / int(fps):.1f}秒
{mem_info}
- 模型版本：DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}{', Image Encoder: ' + image_encoder_model if image_encoder_model else ''}
- 使用Tiled：{'是' if tiled else '否'}
- Tile Size：({tile_size_x}, {tile_size_y})
- Tile Stride：({tile_stride_x}, {tile_stride_y})
- Torch 数据类型：{pipe.torch_dtype_info}
- 使用USP：{'是' if use_usp else '否'}
- 显存管理参数 (num_persistent_param_in_dit)：{num_persistent_param_in_dit if num_persistent_param_in_dit is not None else '未限制'}
- 已加载LoRA：{pipe.lora_info}
- 是否使用控制视频：{'是' if control_video else '否'}
"""
        return output_path, info
    except Exception as e:
        return None, f"生成失败: {str(e)}"
    finally:
        del pipe  # 确保每次生成后清理 pipe

# 创建界面
def create_wan_video_tab():
    # 定义模型目录
    base_dir = "models/wan2.1"
    dit_dir = os.path.join(base_dir, "dit")
    t5_dir = os.path.join(base_dir, "t5")
    vae_dir = os.path.join(base_dir, "vae")
    image_encoder_dir = os.path.join(base_dir, "image_encoder")
    lora_dir = os.path.join(base_dir, "lora")
    
    # 获取模型文件列表
    dit_models = get_model_files(dit_dir)
    t5_models = get_model_files(t5_dir)
    vae_models = get_model_files(vae_dir)
    image_encoder_models = get_model_files(image_encoder_dir)
    lora_models = get_model_files(lora_dir)

    with gr.Blocks(analytics_enabled=False) as wan_interface:
        gr.Markdown("## Wan2.1 文本/图/视频生成视频")
        gr.Markdown("提示：在提示词中添加 `<lora:模型文件名:权重>` 来加载 LoRA，例如 `<lora:example_lora.ckpt:1.0>`")
        
        # 顶部模型选择
        with gr.Row():
            dit_model = gr.Dropdown(
                label="选择 DIT 模型 (可多选，多个文件将合并为一个模型)",
                choices=dit_models,
                value=[dit_models[0]],  # 默认选中第一个模型
                multiselect=True
            )
            t5_model = gr.Dropdown(label="选择 T5 模型", choices=t5_models, value=t5_models[0])
            vae_model = gr.Dropdown(label="选择 VAE 模型", choices=vae_models, value=vae_models[0])
            image_encoder_model = gr.Dropdown(label="选择 Image Encoder 模型 (图生视频必选)", choices=["无"] + image_encoder_models, value="无")
        
        with gr.Tabs():
            with gr.Tab("文本生成视频"):
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(label="正向提示词", lines=3, placeholder="输入描述视频内容的提示词，可包含 <lora:模型文件名:权重>")
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
                            num_inference_steps = gr.Slider(label="推理步数", minimum=1, maximum=100, value=15, step=1)
                            cfg_scale = gr.Number(label="CFG Scale", value=5.0)
                            sigma_shift = gr.Number(label="Sigma Shift", value=5.0)
                            seed = gr.Number(label="随机种子 (-1为随机)", value=-1, precision=0)
                            denoising_strength = gr.Slider(label="降噪强度", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                            rand_device = gr.Dropdown(label="随机设备", choices=["cpu", "cuda"], value="cpu")
                            tiled = gr.Checkbox(label="使用Tiled", value=True)
                            tile_size_x = gr.Number(label="Tile Size X", value=30, precision=0)
                            tile_size_y = gr.Number(label="Tile Size Y", value=52, precision=0)
                            tile_stride_x = gr.Number(label="Tile Stride X", value=15, precision=0)
                            tile_stride_y = gr.Number(label="Tile Stride Y", value=26, precision=0)
                            torch_dtype = gr.Dropdown(label="DIT/T5/VAE 数据类型", choices=["float16", "bfloat16", "float8_e4m3fn"], value="bfloat16")
                            image_encoder_torch_dtype = gr.Dropdown(label="Image Encoder 数据类型", choices=["float16", "float32", "bfloat16"], value="float32")
                            use_usp = gr.Checkbox(label="使用USP (Unified Sequence Parallel)", value=False)
                            nproc_per_node = gr.Number(label="USP 每节点进程数 (需要 torchrun 运行)", value=1, minimum=1, precision=0, visible=False)
                            enable_num_persistent = gr.Checkbox(label="启用显存优化参数 (num_persistent_param_in_dit)", value=False)
                            num_persistent_param_in_dit = gr.Slider(
                                label="显存管理参数值 (值越小显存需求越少,但需要时间变长)",
                                minimum=0,
                                maximum=10**10,
                                value=7*10**9,
                                step=10**8,
                                visible=False,
                                info="启用后调整此值，0 表示最低显存需求"
                            )

                            # 动态显示 nproc_per_node 和 num_persistent_param_in_dit
                            def toggle_nproc_visibility(use_usp):
                                return gr.update(visible=use_usp)
                            use_usp.change(fn=toggle_nproc_visibility, inputs=use_usp, outputs=nproc_per_node)

                            def toggle_num_persistent_visibility(enable):
                                return gr.update(visible=enable)
                            enable_num_persistent.change(fn=toggle_num_persistent_visibility, inputs=enable_num_persistent, outputs=num_persistent_param_in_dit)

                        with gr.Accordion("TeaCache 参数", open=False):
                            tea_cache_l1_thresh = gr.Number(label="TeaCache L1阈值 (越大越快但质量下降)", value=0.07)
                            tea_cache_model_id = gr.Dropdown(label="TeaCache Model ID", choices=["Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B", "Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-720P"], value="Wan2.1-T2V-1.3B")

                        generate_btn = gr.Button("生成视频")

                    with gr.Column():
                        output_video = gr.Video(label="生成结果")
                        info_output = gr.Textbox(label="系统信息", interactive=False, lines=16)

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
                        tea_cache_model_id,
                        dit_model,
                        t5_model,
                        vae_model,
                        image_encoder_model,
                        fps,
                        denoising_strength,
                        rand_device,
                        tiled,
                        tile_size_x,
                        tile_size_y,
                        tile_stride_x,
                        tile_stride_y,
                        torch_dtype,
                        image_encoder_torch_dtype,
                        use_usp,
                        enable_num_persistent,
                        num_persistent_param_in_dit
                    ],
                    outputs=[output_video, info_output]
                )

            with gr.Tab("图片生成视频"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="上传初始图片", type="filepath")
                        end_image_input = gr.Image(label="上传结束图片 (可选)", type="filepath")  # 新增结束图片输入
                        adapt_resolution_btn = gr.Button("自适应图片分辨率")
                        prompt_i2v = gr.Textbox(label="正向提示词", lines=3, placeholder="输入描述视频内容的提示词，可包含 <lora:模型文件名:权重>")
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
                            num_inference_steps_i2v = gr.Slider(label="推理步数", minimum=1, maximum=100, value=15, step=1)
                            cfg_scale_i2v = gr.Number(label="CFG Scale", value=5.0)
                            sigma_shift_i2v = gr.Number(label="Sigma Shift", value=5.0)
                            seed_i2v = gr.Number(label="随机种子 (-1为随机)", value=-1, precision=0)
                            denoising_strength_i2v = gr.Slider(label="降噪强度", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                            rand_device_i2v = gr.Dropdown(label="随机设备", choices=["cpu", "cuda"], value="cpu")
                            tiled_i2v = gr.Checkbox(label="使用Tiled", value=True)
                            tile_size_x_i2v = gr.Number(label="Tile Size X", value=30, precision=0)
                            tile_size_y_i2v = gr.Number(label="Tile Size Y", value=52, precision=0)
                            tile_stride_x_i2v = gr.Number(label="Tile Stride X", value=15, precision=0)
                            tile_stride_y_i2v = gr.Number(label="Tile Stride Y", value=26, precision=0)
                            torch_dtype_i2v = gr.Dropdown(label="DIT/T5/VAE 数据类型", choices=["float16", "bfloat16", "float8_e4m3fn"], value="bfloat16")
                            image_encoder_torch_dtype_i2v = gr.Dropdown(label="Image Encoder 数据类型", choices=["float16", "float32", "bfloat16"], value="float32")
                            use_usp_i2v = gr.Checkbox(label="使用USP (Unified Sequence Parallel)", value=False)
                            nproc_per_node_i2v = gr.Number(label="USP 每节点进程数 (需要 torchrun 运行)", value=1, minimum=1, precision=0, visible=False)
                            enable_num_persistent_i2v = gr.Checkbox(label="启用显存优化参数 (num_persistent_param_in_dit)", value=False)
                            num_persistent_param_in_dit_i2v = gr.Slider(
                                label="显存管理参数值 (值越小显存需求越少,但需要时间变长)",
                                minimum=0,
                                maximum=10**10,
                                value=7*10**9,
                                step=10**8,
                                visible=False,
                                info="启用后调整此值，0 表示最低显存需求"
                            )

                            # 动态显示 nproc_per_node_i2v 和 num_persistent_param_in_dit_i2v
                            def toggle_nproc_visibility(use_usp):
                                return gr.update(visible=use_usp)
                            use_usp_i2v.change(fn=toggle_nproc_visibility, inputs=use_usp_i2v, outputs=nproc_per_node_i2v)

                            def toggle_num_persistent_visibility(enable):
                                return gr.update(visible=enable)
                            enable_num_persistent_i2v.change(fn=toggle_num_persistent_visibility, inputs=enable_num_persistent_i2v, outputs=num_persistent_param_in_dit_i2v)

                        with gr.Accordion("TeaCache 参数", open=False):
                            tea_cache_l1_thresh_i2v = gr.Number(label="TeaCache L1阈值 (越大越快但质量下降)", value=0.19)
                            tea_cache_model_id_i2v = gr.Dropdown(label="TeaCache Model ID", choices=["Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B", "Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-720P"], value="Wan2.1-I2V-14B-480P")

                        generate_i2v_btn = gr.Button("生成视频")

                    with gr.Column():
                        output_video_i2v = gr.Video(label="生成结果")
                        info_output_i2v = gr.Textbox(label="系统信息", interactive=False, lines=16)

                adapt_resolution_btn.click(
                    fn=adaptive_resolution,
                    inputs=[image_input],
                    outputs=[width_i2v, height_i2v]
                )

                generate_i2v_btn.click(
                    fn=generate_i2v,
                    inputs=[
                        image_input,
                        end_image_input,  # 新增结束图片输入
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
                        tea_cache_model_id_i2v,
                        dit_model,
                        t5_model,
                        vae_model,
                        image_encoder_model,
                        fps_i2v,
                        denoising_strength_i2v,
                        rand_device_i2v,
                        tiled_i2v,
                        tile_size_x_i2v,
                        tile_size_y_i2v,
                        tile_stride_x_i2v,
                        tile_stride_y_i2v,
                        torch_dtype_i2v,
                        image_encoder_torch_dtype_i2v,
                        use_usp_i2v,
                        enable_num_persistent_i2v,
                        num_persistent_param_in_dit_i2v
                    ],
                    outputs=[output_video_i2v, info_output_i2v]
                )

            with gr.Tab("视频生成视频"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="上传初始视频", format="mp4")
                        control_video_input = gr.Video(label="上传控制视频 (可选)", format="mp4")  # 新增控制视频输入
                        prompt_v2v = gr.Textbox(label="正向提示词", lines=3, placeholder="输入描述视频内容的提示词，可包含 <lora:模型文件名:权重>")
                        negative_prompt_v2v = gr.Textbox(
                            label="负向提示词",
                            lines=3,
                            value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
                        )
                        
                        with gr.Accordion("基础参数", open=True):
                            width_v2v = gr.Slider(label="宽度", minimum=256, maximum=1920, value=832, step=8)
                            height_v2v = gr.Slider(label="高度", minimum=256, maximum=1080, value=480, step=8)
                            num_frames_v2v = gr.Number(label="帧数", value=81, minimum=1, precision=0)
                            fps_v2v = gr.Slider(label="输出帧率 (FPS)", minimum=1, maximum=60, value=15, step=1)

                        with gr.Accordion("高级参数", open=False):
                            num_inference_steps_v2v = gr.Slider(label="推理步数", minimum=1, maximum=100, value=15, step=1)
                            cfg_scale_v2v = gr.Number(label="CFG Scale", value=5.0)
                            sigma_shift_v2v = gr.Number(label="Sigma Shift", value=5.0)
                            seed_v2v = gr.Number(label="随机种子 (-1为随机)", value=-1, precision=0)
                            denoising_strength_v2v = gr.Slider(label="降噪强度", minimum=0.0, maximum=1.0, value=0.7, step=0.01)
                            rand_device_v2v = gr.Dropdown(label="随机设备", choices=["cpu", "cuda"], value="cpu")
                            tiled_v2v = gr.Checkbox(label="使用Tiled", value=True)
                            tile_size_x_v2v = gr.Number(label="Tile Size X", value=30, precision=0)
                            tile_size_y_v2v = gr.Number(label="Tile Size Y", value=52, precision=0)
                            tile_stride_x_v2v = gr.Number(label="Tile Stride X", value=15, precision=0)
                            tile_stride_y_v2v = gr.Number(label="Tile Stride Y", value=26, precision=0)
                            torch_dtype_v2v = gr.Dropdown(label="DIT/T5/VAE 数据类型", choices=["float16", "bfloat16", "float8_e4m3fn"], value="bfloat16")
                            image_encoder_torch_dtype_v2v = gr.Dropdown(label="Image Encoder 数据类型", choices=["float16", "float32", "bfloat16"], value="float32")
                            use_usp_v2v = gr.Checkbox(label="使用USP (Unified Sequence Parallel)", value=False)
                            nproc_per_node_v2v = gr.Number(label="USP 每节点进程数 (需要 torchrun 运行)", value=1, minimum=1, precision=0, visible=False)
                            enable_num_persistent_v2v = gr.Checkbox(label="启用显存优化参数 (num_persistent_param_in_dit)", value=False)
                            num_persistent_param_in_dit_v2v = gr.Slider(
                                label="显存管理参数值 (值越小显存需求越少,但需要时间变长)",
                                minimum=0,
                                maximum=10**10,
                                value=7*10**9,
                                step=10**8,
                                visible=False,
                                info="启用后调整此值，0 表示最低显存需求"
                            )

                            # 动态显示 nproc_per_node_v2v 和 num_persistent_param_in_dit_v2v
                            def toggle_nproc_visibility(use_usp):
                                return gr.update(visible=use_usp)
                            use_usp_v2v.change(fn=toggle_nproc_visibility, inputs=use_usp_v2v, outputs=nproc_per_node_v2v)

                            def toggle_num_persistent_visibility(enable):
                                return gr.update(visible=enable)
                            enable_num_persistent_v2v.change(fn=toggle_num_persistent_visibility, inputs=enable_num_persistent_v2v, outputs=num_persistent_param_in_dit_v2v)

                        generate_v2v_btn = gr.Button("生成视频")

                    with gr.Column():
                        output_video_v2v = gr.Video(label="生成结果")
                        info_output_v2v = gr.Textbox(label="系统信息", interactive=False, lines=16)

                generate_v2v_btn.click(
                    fn=generate_v2v,
                    inputs=[
                        video_input,
                        control_video_input,  # 新增控制视频输入
                        prompt_v2v,
                        negative_prompt_v2v,
                        num_inference_steps_v2v,
                        seed_v2v,
                        height_v2v,
                        width_v2v,
                        num_frames_v2v,
                        cfg_scale_v2v,
                        sigma_shift_v2v,
                        dit_model,
                        t5_model,
                        vae_model,
                        image_encoder_model,
                        fps_v2v,
                        denoising_strength_v2v,
                        rand_device_v2v,
                        tiled_v2v,
                        tile_size_x_v2v,
                        tile_size_y_v2v,
                        tile_stride_x_v2v,
                        tile_stride_y_v2v,
                        torch_dtype_v2v,
                        image_encoder_torch_dtype_v2v,
                        use_usp_v2v,
                        enable_num_persistent_v2v,
                        num_persistent_param_in_dit_v2v
                    ],
                    outputs=[output_video_v2v, info_output_v2v]
                )
    
    return wan_interface

if IN_WEBUI:
    script_callbacks.on_ui_tabs(lambda: [(create_wan_video_tab(), "Wan Video", "wan_video_tab")])
else:
    if __name__ == "__main__":
        interface = create_wan_video_tab()
        print("提示：若启用 USP，需使用以下命令运行：")
        print("torchrun --standalone --nproc_per_node=<进程数> generation.py")
        interface.launch(
            allowed_paths=["outputs"]
        )
