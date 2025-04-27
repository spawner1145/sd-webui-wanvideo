import requests
import base64
import os
import json
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API 配置
API_URL = "http://127.0.0.1:7870/wanvideo/v1/v2v" # 替换为你的实际API地址
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 视频转 Base64
def video_to_base64(video_path):
    try:
        with open(video_path, "rb") as f:
            video_data = f.read()
        return base64.b64encode(video_data).decode("utf-8")
    except Exception as e:
        logging.error(f"视频编码失败: {str(e)}")
        return None

# 请求参数
VIDEO_PATH = "input_video.mp4"  # 替换为实际视频路径
PAYLOAD = {
    "video": video_to_base64(VIDEO_PATH),  # Base64 编码的初始视频
    "control_video": None,  # 可选控制视频
    "prompt": "A futuristic transformation of the input video",
    "negative_prompt": "low quality, blurry, static",
    "dit_models": ["diffusion_pytorch_model.safetensors"],  # 替换为实际的 DIT 模型文件名，分片模型记得全加上
    "t5_model": "models_t5_umt5-xxl-enc-bf16.pth",        # 替换为实际的 T5 模型文件名
    "vae_model": "Wan2.1_VAE.pth",      # 替换为实际的 VAE 模型文件名
    "image_encoder_model": None,
    "num_inference_steps": 15,
    "seed": -1,
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "cfg_scale": 5.0,
    "sigma_shift": 5.0,
    "fps": 15,
    "denoising_strength": 0.7,
    "rand_device": "cpu",
    "tiled": True,
    "tile_size_x": 30,
    "tile_size_y": 52,
    "tile_stride_x": 15,
    "tile_stride_y": 26,
    "torch_dtype": "bfloat16",
    "image_encoder_torch_dtype": "float32",
    "use_usp": False,
    "enable_num_persistent": False,
    "num_persistent_param_in_dit": None
}

def request_v2v(payload):
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, json=payload, headers=headers, timeout=None)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API 请求失败: {str(e)}")
        return None

def save_video(video_base64, output_dir):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"v2v_output_{timestamp}.mp4")
        video_data = base64.b64decode(video_base64)
        with open(output_path, "wb") as f:
            f.write(video_data)
        logging.info(f"视频已保存到: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"保存视频失败: {str(e)}")
        return None

def main():
    if not PAYLOAD["video"]:
        logging.error("视频编码失败，退出")
        return
    logging.info("开始发送 v2v 请求")
    response = request_v2v(PAYLOAD)
    if not response:
        logging.error("请求失败，退出")
        return
    if "video" not in response or "info" not in response:
        logging.error(f"响应格式错误: {json.dumps(response, indent=2)}")
        return
    video_base64 = response["video"]
    info = response["info"]
    output_path = save_video(video_base64, OUTPUT_DIR)
    if output_path:
        logging.info(f"生成信息:\n{info}")
    else:
        logging.error("视频保存失败")

if __name__ == "__main__":
    main()