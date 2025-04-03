# install.py
import launch

def install():
    if not launch.is_installed("modelscope"):
        launch.run_pip("install modelscope", "requirements for Wan Video Generator")
    
    if not launch.is_installed("diffsynth"):
        launch.run_pip("install git+https://github.com/modelscope/DiffSynth-Studio", 
                      "requirements for Wan Video Generator - DiffSynth-Studio")

if __name__ == "__main__":
    install()
