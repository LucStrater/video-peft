# Installation

This codebase is built on the following sources:

- Python
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.lightning.ai/)
- [Weights and Biases](https://www.wandb.com/)
- [TorchCodec](https://pytorch.org/blog/torchcodec/)

## Setup

The experiments in the paper use Python 3.10, PyTorch 2.6 and CUDA 12.6.

```bash
conda create -n video-peft python=3.10
conda activate video-peft
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

TorchCodec decodes the videos with FFmpeg, so a system FFmpeg (we use 7.0.2) must be available. The same FFmpeg is
used to re-encode the datasets to DNxHR HQ (see [DATASET.md](DATASET.md)).

Logging uses [Weights and Biases](https://wandb.ai). Log in once with `wandb login` (or set `WANDB_API_KEY`), or
pass `--wandb_mode offline` / `--wandb_mode disabled` to run without an account. Test metrics are also printed to
the log at the end of every run.

## System test

The system test prints the Python, PyTorch, CUDA and hardware information and all installed packages, and checks
that TorchCodec can be loaded:

```bash
python main.py --state_type test_system --config config/classification.yaml
```

Our reference output is below. Some of the listed libraries are not needed for the codebase to run.

```
xFormers not available
Running system test...
Python version: 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]
Hello from <anonymous hostname>
Current time: 2026-07-25 15:16:43.994039
Operating System: Linux 5.14.0-687.29.1.el9_8.x86_64
CPU: x86_64
Number of CPU cores: 72
Total RAM: 503.50 GB
Available RAM: 429.93 GB

PyTorch Information:
PyTorch version: 2.6.0+cu126
PyTorch CUDA available: True
CUDA version: 12.6
Current CUDA device: 0
CUDA device name: NVIDIA A100-SXM4-40GB
Number of CUDA devices: 1

Disk Usage:
Total disk space: 251.63 GB
Used disk space: 3.81 GB
Free disk space: 247.81 GB

All Installed Packages and Versions:
Deprecated==1.3.1
GitPython==3.1.46
ImageIO==2.37.2
Jinja2==3.1.6
MarkupSafe==2.1.5
PyYAML==6.0.3
TorchCodec==0.2.1+cu126
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
annotated-types==0.7.0
anyio==4.12.1
async-lru==2.0.5
async-timeout==5.0.1
asyncio-atexit==1.0.1
attrs==25.4.0
autocommand==2.2.2
backports.tarfile==1.2.0
certifi==2026.1.4
charset-normalizer==3.4.4
click==8.3.1
contourpy==1.3.2
cycler==0.12.1
deno==2.6.10
einops==0.8.2
exceptiongroup==1.3.1
filelock==3.20.0
flash_attn==2.8.3
fonttools==4.61.1
frozenlist==1.8.0
fsspec==2025.12.0
gitdb==4.0.12
googleapis-common-protos==1.75.0
h11==0.16.0
h5py==3.16.0
hdf5plugin==6.0.0
hf-xet==1.2.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.3.7
idna==3.11
importlib_metadata==8.7.1
jaraco.context==6.1.0
jaraco.functools==4.4.0
jaraco.text==4.0.0
kiwisolver==1.4.9
lightning==2.6.1
lightning-utilities==0.15.2
matplotlib==3.10.8
more-itertools==10.8.0
mpmath==1.3.0
multidict==6.7.1
nest-asyncio==1.6.0
networkx==3.4.2
numpy==2.2.6
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvrtc-cu12==12.6.77
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-cufft-cu12==11.3.0.4
nvidia-curand-cu12==10.3.7.77
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-cusparselt-cu12==0.6.3
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.6.85
nvidia-nvtx-cu12==12.6.77
opencv-python==4.13.0.90
opentelemetry-api==1.43.0
opentelemetry-exporter-otlp-proto-common==1.43.0
opentelemetry-exporter-otlp-proto-http==1.43.0
opentelemetry-instrumentation==0.64b0
opentelemetry-instrumentation-httpx==0.64b0
opentelemetry-instrumentation-requests==0.64b0
opentelemetry-instrumentation-threading==0.64b0
opentelemetry-instrumentation-urllib==0.64b0
opentelemetry-proto==1.43.0
opentelemetry-sdk==1.43.0
opentelemetry-semantic-conventions==0.64b0
opentelemetry-util-http==0.64b0
packaging==25.0
packaging==25.0
pandas==2.3.3
pillow==12.0.0
pip==25.3
platformdirs==4.4.0
platformdirs==4.5.1
propcache==0.4.1
protobuf==6.33.5
psutil==7.2.2
pyarrow==25.0.0
pydantic==2.12.5
pydantic_core==2.41.5
pyparsing==3.3.2
pypng==0.20220715.0
python-dateutil==2.9.0.post0
pytorch-lightning==2.6.1
pytz==2025.2
regex==2026.1.15
requests==2.34.2
safetensors==0.7.0
sentry-sdk==2.51.0
setuptools==80.10.1
shellingham==1.5.4
six==1.17.0
smmap==5.0.2
sympy==1.13.1
synapseclient==4.13.0
timm==1.0.24
tokenizers==0.22.2
tomli==2.4.0
torch==2.6.0+cu126
torchaudio==2.6.0+cu126
torchmetrics==1.8.2
torchvision==0.21.0+cu126
tqdm==4.67.2
transformers==5.0.0
triton==3.2.0
typer-slim==0.21.1
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.3
urllib3==2.6.3
wandb==0.24.1
wheel==0.45.1
wheel==0.46.3
wrapt==2.2.2
yarl==1.22.0
yt-dlp==2026.3.17
zipp==3.23.0
TorchCodec Version: 0.2.1+cu126
Success: Decoder recognized 'seek_mode' but failed on file: Could not open input file: example.mov No such file or directory
```


