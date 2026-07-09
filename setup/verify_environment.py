import os
import shutil

import torch
from dotenv import load_dotenv

load_dotenv()

def success(message):
    print(f"success {message}")

def error(message):
    print(f"error {message}")

def info(message):
    print(f"info {message}")

def main():
    info("Starting environment verification...")
    info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        success("CUDA is available")
    else:
        error("CUDA is NOT available")
        return
    gpu_name = torch.cuda.get_device_name(0)
    success(f"GPU: {gpu_name}")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / (1024 ** 3)
    success(f"GPU Memory: {total_memory_gb:.1f} GB")


if __name__ == "__main__":
    main()

