"""
system_test.py

This file contains a function to test and display various system information including
Python version, hostname, current time, operating system details, CPU information, RAM usage,
PyTorch information, disk usage, and all installed packages and their versions.

This information can be used to compare against the specifications listed in INSTALL.md
for compatibility purposes.

Author: Luc Sträter
"""

import socket
import datetime
import sys
import platform
import psutil
import os
from importlib.metadata import distributions
import torchcodec
from torchcodec.decoders import VideoDecoder
import torch

def test_system():
    # System Information
    print(f"Python version: {sys.version}")
    print(f"Hello from {socket.gethostname()}!")
    print(f"Current time: {datetime.datetime.now()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"Number of CPU cores: {os.cpu_count()}")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")

    # PyTorch Information
    print("\nPyTorch Information:")
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("Number of CUDA devices:", torch.cuda.device_count())
    else:
        print("PyTorch is not using GPU")

    # Disk Usage
    print("\nDisk Usage:")
    disk = psutil.disk_usage('/')
    print(f"Total disk space: {disk.total / (1024**3):.2f} GB")
    print(f"Used disk space: {disk.used / (1024**3):.2f} GB")
    print(f"Free disk space: {disk.free / (1024**3):.2f} GB")

    # All Installed Packages and Versions
    print("\nAll Installed Packages and Versions:")
    installed_packages = sorted([(d.metadata['Name'], d.version) for d in distributions()])
    for name, version in installed_packages:
        print(f"{name}=={version}")

    print(f"TorchCodec Version: {torchcodec.__version__}")

    # Test the new parameter
    try:
        # Replace with any small video file path
        test_path = "example.mov" 
        decoder = VideoDecoder(test_path, seek_mode='approximate')
        print("Success: Approximate seek is enabled and stable.")
    except TypeError:
        print("Error: seek_mode is not recognized. Check version.")
    except Exception as e:
        print(f"Success: Decoder recognized 'seek_mode' but failed on file: {e}")