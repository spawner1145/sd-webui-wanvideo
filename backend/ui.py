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
from backend.inferrence import *

# 设置日志
logging.basicConfig(level=logging.INFO)
try:
    from scripts.gradio_patch import money_patch_gradio
    if money_patch_gradio():
        logging.info("成功应用gradio补丁")
    else:
        logging.warning("gradio补丁导入失败")
except Exception as e:
    logging.error(e)

# 检查是否在 WebUI 环境中运行
try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'opts': type('Opts', (), {'outdir_samples': '', 'outdir_txt2img_samples': ''})})()

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
                        image_input = gr.Image(label="上传首帧", type="filepath")
                        end_image_input = gr.Image(label="上传尾帧 (可选)", type="filepath")  # 新增尾帧输入
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
                        end_image_input,  # 新增尾帧输入
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