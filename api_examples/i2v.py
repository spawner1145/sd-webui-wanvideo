import requests
import base64
import os
import json
import logging
from datetime import datetime
from PIL import Image
from io import BytesIO

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API 配置
API_URL = "http://127.0.0.1:7870/wanvideo/v1/i2v" # 替换为你的实际API地址
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 图片转 Base64
def image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error(f"图片编码失败: {str(e)}")
        return None

# 请求参数
IMAGE_PATH = "start_image.png"  # 替换为实际图片路径
PAYLOAD = {
    "image": image_to_base64(IMAGE_PATH),  # Base64 编码的首帧
    "end_image": None,  # 可选尾帧
    "prompt": "A serene beach transitioning to night",
    "negative_prompt": "low quality, blurry, static",
    "dit_models": ["diffusion_pytorch_model.safetensors"],  # 替换为实际的 DIT 模型文件名，分片模型记得全加上
    "t5_model": "models_t5_umt5-xxl-enc-bf16.pth",        # 替换为实际的 T5 模型文件名
    "vae_model": "Wan2.1_VAE.pth",      # 替换为实际的 VAE 模型文件名
    "image_encoder_model": "image_encoder.ckpt",  # 必须提供
    "num_inference_steps": 15,
    "seed": -1,
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "cfg_scale": 5.0,
    "sigma_shift": 5.0,
    "tea_cache_l1_thresh": 0.19,
    "tea_cache_model_id": "Wan2.1-I2V-14B-480P",
    "fps": 15,
    "denoising_strength": 1.0,
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

def request_i2v(payload):
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
        output_path = os.path.join(output_dir, f"i2v_output_{timestamp}.mp4")
        video_data = base64.b64decode(video_base64)
        with open(output_path, "wb") as f:
            f.write(video_data)
        logging.info(f"视频已保存到: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"保存视频失败: {str(e)}")
        return None

def main():
    if not PAYLOAD["image"]:
        logging.error("图片编码失败，退出")
        return
    logging.info("开始发送 i2v 请求")
    response = request_i2v(PAYLOAD)
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