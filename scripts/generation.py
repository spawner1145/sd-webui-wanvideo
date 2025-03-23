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
import threading
import signal
import sys
import subprocess
import pkg_resources

# 定义模型路径和输出路径
models_dir = "webui/models/wan"
outputs_dir = "webui/outputs/videos"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

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

# 全局变量用于控制任务终止
stop_event = threading.Event()

def check_and_update_diffusers():
    try:
        diffusers_version = pkg_resources.get_distribution("diffusers").version
        if pkg_resources.parse_version(diffusers_version) <= pkg_resources.parse_version("0.31.0"):
            print("检测到 diffusers 版本不大于 0.31.0，开始从源码下载最新版本...")
            subprocess.run(["pip", "install", "git+https://github.com/huggingface/diffusers.git"])
            print("diffusers 版本更新完成。")
    except pkg_resources.DistributionNotFound:
        print("未找到 diffusers 包，开始从源码下载...")
        subprocess.run(["pip", "install", "git+https://github.com/huggingface/diffusers.git"])
        print("diffusers 包下载完成。")


def download_model(model_id, source, update_button):
    model_path = os.path.join(models_dir, model_id.split("/")[-1])
    if os.path.exists(model_path):
        print(f"模型 {model_id} 已存在于 {model_path}，无需再次下载。")
        return model_path
    print(f"开始从 {source} 下载模型 {model_id}...")
    try:
        if source == "Hugging Face":
            snapshot_download(repo_id=model_id, local_dir=model_path)
        elif source == "ModelScope":
            modelscope_snapshot_download(model_id, cache_dir=models_dir)
        print(f"模型 {model_id} 下载完成，保存路径为 {model_path}。")
        update_button.update(value="生成视频")
        return model_path
    except Exception as e:
        print(f"下载模型时出错: {e}")
        update_button.update(value="生成视频")
        return None


def text_to_video(prompt, negative_prompt, model_id, source, height, width, num_frames, guidance_scale, flow_shift, fps, update_button, output_video):
    global stop_event
    stop_event.clear()
    check_and_update_diffusers()
    model_path = os.path.join(models_dir, model_id.split("/")[-1])
    if not os.path.exists(model_path):
        update_button.update(value="下载中")
        model_path = download_model(model_id, source, update_button)
        if model_path is None:
            return None
    update_button.update(value="生成中")

    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000,
                                        flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.to("cuda")

    intermediate_frames = []
    for i in range(num_frames):
        if stop_event.is_set():
            print("文生视频任务已终止")
            if intermediate_frames:
                output_path = os.path.join(outputs_dir, "text_to_video_output_intermediate.mp4")
                export_to_video(intermediate_frames, output_path, fps=fps)
                output_video.update(value=output_path)
            update_button.update(value="生成视频")
            return None
        frame = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=1,
            guidance_scale=guidance_scale,
        ).frames[0]
        intermediate_frames.append(frame)

    output_path = os.path.join(outputs_dir, "text_to_video_output.mp4")
    export_to_video(intermediate_frames, output_path, fps=fps)
    update_button.update(value="生成视频")
    return output_path


def image_to_video(image, prompt, negative_prompt, model_id, source, num_frames, guidance_scale, fps, update_button, output_video):
    global stop_event
    stop_event.clear()
    check_and_update_diffusers()
    model_path = os.path.join(models_dir, model_id.split("/")[-1])
    if not os.path.exists(model_path):
        update_button.update(value="下载中")
        model_path = download_model(model_id, source, update_button)
        if model_path is None:
            return None
    update_button.update(value="生成中")

    image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_path, vae=vae, image_encoder=image_encoder,
                                                   torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    max_area = 720 * 1280
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))

    intermediate_frames = []
    for i in range(num_frames):
        if stop_event.is_set():
            print("图生视频任务已终止")
            if intermediate_frames:
                output_path = os.path.join(outputs_dir, "image_to_video_output_intermediate.mp4")
                export_to_video(intermediate_frames, output_path, fps=fps)
                output_video.update(value=output_path)
            update_button.update(value="生成视频")
            return None
        frame = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height, width=width,
            num_frames=1,
            guidance_scale=guidance_scale
        ).frames[0]
        intermediate_frames.append(frame)

    output_path = os.path.join(outputs_dir, "image_to_video_output.mp4")
    export_to_video(intermediate_frames, output_path, fps=fps)
    update_button.update(value="生成视频")
    return output_path


def stop_task():
    global stop_event
    stop_event.set()


def on_ui_tabs():
    with gr.Blocks() as demo:
        gr.Markdown("## 文生视频和图生视频工具")
        with gr.Tab("文生视频"):
            with gr.Row():
                with gr.Column(scale=2):  # 增大输入区域列的比例
                    prompt_text = gr.Textbox(label="提示词", lines=5, max_lines=10,
                                             value="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.")
                    negative_prompt_text = gr.Textbox(label="负提示词", lines=5, max_lines=10,
                                                      value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
                    model_choice_text = gr.Dropdown(choices=text_to_video_models, label="选择模型",
                                                    value=text_to_video_models[0])
                    source_choice_text = gr.Dropdown(choices=["Hugging Face", "ModelScope"], label="下载源",
                                                     value="Hugging Face")
                    height_slider_text = gr.Slider(minimum=256, maximum=1080, step=16, label="高度", value=720)
                    width_slider_text = gr.Slider(minimum=256, maximum=1920, step=16, label="宽度", value=1280)
                    num_frames_slider_text = gr.Slider(minimum=16, maximum=128, step=1, label="帧数", value=81)
                    guidance_scale_slider_text = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, label="引导系数",
                                                           value=5.0)
                    flow_shift_slider_text = gr.Slider(minimum=3.0, maximum=5.0, step=0.1, label="Flow Shift",
                                                       value=5.0)
                    fps_slider_text = gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=16)
                    generate_button_text = gr.Button("生成视频")
                    stop_button_text = gr.Button("终止任务", visible=False)
                with gr.Column(scale=1):
                    output_video_text = gr.Video(label="输出视频")

            def start_text_to_video():
                stop_button_text.update(visible=True)
                return gr.Button.update(visible=False)

            def end_text_to_video():
                stop_button_text.update(visible=False)
                return gr.Button.update(visible=True)

            generate_button_text.click(
                fn=start_text_to_video,
                outputs=[generate_button_text]
            ).then(
                fn=text_to_video,
                inputs=[prompt_text, negative_prompt_text, model_choice_text, source_choice_text, height_slider_text,
                        width_slider_text, num_frames_slider_text, guidance_scale_slider_text, flow_shift_slider_text, fps_slider_text, generate_button_text, output_video_text],
                outputs=output_video_text
            ).then(
                fn=end_text_to_video,
                outputs=[generate_button_text]
            )

            stop_button_text.click(
                fn=stop_task,
                inputs=[],
                outputs=[]
            )

        with gr.Tab("图生视频"):
            with gr.Row():
                with gr.Column(scale=2):  # 增大输入区域列的比例
                    input_image = gr.Image(type="pil", label="输入图片")
                    prompt_image = gr.Textbox(label="提示词", lines=5, max_lines=10,
                                              value="An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.")
                    negative_prompt_image = gr.Textbox(label="负提示词", lines=5, max_lines=10,
                                                       value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
                    model_choice_image = gr.Dropdown(choices=image_to_video_models, label="选择模型",
                                                     value=image_to_video_models[0])
                    source_choice_image = gr.Dropdown(choices=["Hugging Face", "ModelScope"], label="下载源",
                                                      value="Hugging Face")
                    num_frames_slider_image = gr.Slider(minimum=16, maximum=128, step=1, label="帧数", value=81)
                    guidance_scale_slider_image = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, label="引导系数",
                                                            value=5.0)
                    fps_slider_image = gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=16)
                    generate_button_image = gr.Button("生成视频")
                    stop_button_image = gr.Button("终止任务", visible=False)
                with gr.Column(scale=1):
                    output_video_image = gr.Video(label="输出视频")

            def start_image_to_video():
                stop_button_image.update(visible=True)
                return gr.Button.update(visible=False)

            def end_image_to_video():
                stop_button_image.update(visible=False)
                return gr.Button.update(visible=True)

            generate_button_image.click(
                fn=start_image_to_video,
                outputs=[generate_button_image]
            ).then(
                fn=image_to_video,
                inputs=[input_image, prompt_image, negative_prompt_image, model_choice_image, source_choice_image,
                        num_frames_slider_image, guidance_scale_slider_image, fps_slider_image, generate_button_image, output_video_image],
                outputs=output_video_image
            ).then(
                fn=end_image_to_video,
                outputs=[generate_button_image]
            )

            stop_button_image.click(
                fn=stop_task,
                inputs=[],
                outputs=[]
            )

    return [(demo, "wan video", "video_generation_tab")]


try:
    from modules import script_callbacks

    script_callbacks.on_ui_tabs(on_ui_tabs)
except ImportError:
    print("未找到 script_callbacks 模块，可能不是在 Stable Diffusion WebUI 环境中运行。")