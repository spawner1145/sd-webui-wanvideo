import logging
import threading
from threading import Lock
from fastapi import FastAPI
from backend.inferrence import *
from backend.ui import *
from backend.api import Api
import uvicorn

# 设置日志
logging.basicConfig(level=logging.INFO)
try:
    from scripts.gradio_patch import money_patch_gradio
    if money_patch_gradio():
        logging.info("成功应用gradio补丁")
    else:
        logging.warning("gradio补丁导入失败")
except Exception as e:
    logging.error(f"Gradio补丁加载失败: {e}")

# 检查是否在 WebUI 环境中运行
try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'opts': type('Opts', (), {'outdir_samples': '', 'outdir_txt2img_samples': ''})})()

# 硬编码配置
HOST = "127.0.0.1"
PORT_API = 7870     # 独立的 FastAPI 端口
NPROC_PER_NODE = 1  # 默认 USP 进程数

if IN_WEBUI:
    # WebUI 环境下，注册 UI 和 API 回调
    from backend.api import on_app_started
    script_callbacks.on_ui_tabs(lambda: [(create_wan_video_tab(), "Wan Video", "wan_video_tab")])
    script_callbacks.on_app_started(on_app_started)
else:
    # 非 WebUI 环境下，分别启动 Gradio UI 和 FastAPI
    if __name__ == "__main__":
        # 创建 Gradio 界面
        interface = create_wan_video_tab()
        logging.info("Gradio 界面已创建")

        # 创建独立的 FastAPI 实例
        app = FastAPI(docs_url="/docs", openapi_url="/openapi.json")
        queue_lock = Lock()  # 为非 WebUI 环境提供线程锁
        api = Api(app, queue_lock, prefix="/wanvideo/v1")
        logging.info("API 路由已挂载到独立的 FastAPI 实例")

        # 打印 USP 和 API 文档提示
        if NPROC_PER_NODE > 1:
            print("提示：已启用 USP，需使用以下命令运行：")
            print(f"torchrun --standalone --nproc_per_node={NPROC_PER_NODE} generation.py")
        else:
            print("提示：若需启用 USP，需使用以下命令运行（修改 NPROC_PER_NODE）：")
            print("torchrun --standalone --nproc_per_node=<进程数> generation.py")
        print(f"API 文档可用：http://{HOST}:{PORT_API}/docs")

        # 在单独线程中启动 Gradio
        def run_gradio():
            try:
                interface.launch(
                    server_name=HOST,
                    allowed_paths=["outputs"],
                    prevent_thread_lock=True
                )
            except Exception as e:
                logging.error(f"Gradio 启动失败: {str(e)}")

        gradio_thread = threading.Thread(target=run_gradio)
        gradio_thread.start()

        # 启动独立的 FastAPI 服务器
        try:
            uvicorn.run(
                app,
                host=HOST,
                port=PORT_API,
                log_level="info"
            )
        except Exception as e:
            logging.error(f"FastAPI 启动失败: {str(e)}")
        finally:
            interface.close()
            gradio_thread.join()
