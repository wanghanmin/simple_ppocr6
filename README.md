# simple-ppocr6

[English](#english) | [中文](#中文)

---

## 中文

### 简介

基于百度 PaddleOCR **PP-OCRv6** 模型的简易 Python OCR 库，使用 ONNXRuntime 纯推理（不依赖 PaddleOCR 框架），支持 50 语种文本检测与识别，几行代码即可完成图片文字提取。

### 特性

- **轻量封装**：一个类即可完成加载模型、检测、方向分类与识别
- **ONNXRuntime 推理**：支持 CPU / GPU（CUDAExecutionProvider）
- **检测+识别一体化**：自动完成文本框检测、角度分类与文本识别
- **简易可视化**：内置 `displaybox` 方法，可直接画出检测框查看效果
- **易于集成**：既可作为脚本运行，也可在其他 Python 项目中 `import simple_ppocr6` 作为库使用
- **HTTP 服务**：提供生产级 HTTP API 服务（可选）
- **三种规格模型**：`model_size='medium'`（默认，随仓库）· `'small'` / `'tiny'`（可选，需自行下载）

### 性能参考

默认 **`model_size='medium'`**，克隆仓库即可使用；更快或低显存可选用 **`small`**（~800MB）或 **`tiny`**（~550MB），需先运行 `download_models.py`。

**1080p 全量 OCR**（1920×1080，GPU，HEURISTIC，无 cuDNN Fallback）：

| 规格 | Linux A4000 | Linux 2080Ti | Windows RTX 3070 |
|------|---------------|--------------|------------------|
| **small** | **~86 ms** | ~96 ms | ~294 ms |
| medium | ~202 ms | ~244 ms | ~618 ms |

**GPU 显存**（det + rec + ori 三 Session 常驻）：tiny ~550 MiB · small ~800 MiB · medium ~1.7 GiB

> **cuDNN**：ORT 1.20+ 的 `DEFAULT` 会触发 Fallback 极慢；本库默认 **HEURISTIC**（`SIMPLE_PPOCR6_CUDNN_CONV_ALGO` / `_REC`）。
### 环境与依赖

- **Python**: 3.7+
- **依赖**:
  - `onnxruntime` 或 `onnxruntime-gpu`
  - `opencv-python`

> 说明：`opencv-python` 依赖 `numpy`，通常会自动一并安装，无需单独声明。

安装依赖：

```bash
pip install -r requirements.txt
```

### 模型文件

本项目使用 **PaddleOCR PP-OCRv6** 的 ONNX 导出模型。通过 `model_size` 选择规格（默认 `medium`）：

| 规格 | 检测 | 识别 | 字典 | 说明 |
|------|------|------|------|------|
| `medium`（默认） | `ppocr6_m_det.onnx` | `ppocr6_m_rec.onnx` | `ppocr6_dict.txt`（18708 字符，50 语种） | **随仓库提供**，克隆即用 |
| `small` | `ppocr6_s_det.onnx` | `ppocr6_s_rec.onnx` | `ppocr6_s_dict.txt` | 可选，`download_models.py --size small` |
| `tiny` | `ppocr6_t_det.onnx` | `ppocr6_t_rec.onnx` | `ppocr6_t_dict.txt`（6904 字符） | 可选，`download_models.py --size tiny` |

共用模型：

- `models/ppocr6_textline_ori.onnx` — 文本行方向分类

> **说明**：模型文件来源于 [PaddlePaddle HuggingFace](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6)。使用模型时请遵守其原始许可证和使用条款。

将上述文件放到仓库根目录下的 `models/` 目录，保持默认路径，即可正常使用。克隆仓库已包含 **medium** 全套；若需 tiny/small 或重新下载 medium，可运行：

```bash
pip install huggingface_hub pyyaml
python download_models.py --size tiny    # 或 small / medium
```

### 安装与使用

克隆仓库并安装依赖：

```bash
git clone https://github.com/wanghanmin/simple_ppocr6.git
cd simple_ppocr6
pip install -r requirements.txt
```


### 快速上手

下面是一个最小示例，演示如何对单张图片进行 OCR：

```python
from simple_ppocr6 import simple_ppocr6

ocr = simple_ppocr6(use_gpu=True)  # 默认 medium，克隆即用
ocr.run("path/to/your/image.jpg")

# 可选：仅 det 定位 / 局部 rec（默认全量识别，不截断框）
# ocr.run(img, rec=False)
# ocr.run(img, focus_rect=(x1, y1, x2, y2))

for item in ocr.results:
    print(item['text'], item['rec_pos'])
```
更多示例可以参考仓库根目录中的 `ocr_image.py` 和 `winclip_ocr.py`。

### 示例

#### 1. 命令行 OCR

- **`ocr_image.py`**  
  从本地图片进行 OCR，打印全部文本与框位置、性能统计，并可视化检测框；支持命令行参数传入图片路径（无参数时默认使用 `example.jpg`）
  
- **`winclip_ocr.py`**  
  Windows 截图/剪贴板 OCR 示例

运行示例：

```bash
python ocr_image.py                    # example.jpg，默认 medium
python ocr_image.py your.jpg --gpu     # 指定图片 + GPU
python winclip_ocr.py                  # 剪贴板 OCR（默认 medium + GPU）
python winclip_ocr.py --model-size small   # 需先 download_models.py --size small
```
#### 2. HTTP API 服务

- **`ocr-http-service.py`**  
  提供生产级 HTTP API 服务，支持接收 base64 编码的图片，返回 JSON 格式的 OCR 结果

**安装额外依赖：**

```bash
pip install flask waitress
```

**启动服务：**

```bash
# 使用默认配置（0.0.0.0:11005，CPU 模式）
python ocr-http-service.py

# 指定端口
python ocr-http-service.py --port 8080

# 指定 IP 和端口
python ocr-http-service.py --host 127.0.0.1 --port 8080

# 启用 GPU 模式
python ocr-http-service.py --gpu

# 完整配置
python ocr-http-service.py --host 0.0.0.0 --port 8080 --threads 8 --gpu

# 查看帮助
python ocr-http-service.py --help
```

**启动参数：**

- `--host`: 绑定的 IP 地址（默认：`0.0.0.0`）
- `--port`: 绑定的端口号（默认：`11005`）
- `--threads`: 工作线程数（默认：`4`）
- `--gpu`: 启用 GPU 加速（默认：CPU 模式）

**API 调用示例：**

```python
import requests
import base64

# 读取图片并转换为 base64
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# 发送 POST 请求
response = requests.post(
    "http://localhost:11005/ocr",
    json={"image": image_base64}
)

# 查看结果
result = response.json()
print(f"OCR 处理时间: {result['processing_time']:.3f}s")
for item in result['results']:
    print(f"文本: {item['text']}")
    print(f"位置: {item['bounding_box']}")
```

**返回格式：**

```json
{
  "processing_time": 0.523,
  "decode_time": 0.012,
  "total_time": 0.535,
  "results": [
    {
      "text": "识别的文本",
      "bounding_box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ]
}
```

### 注意事项

- 仓库自带 **medium** 模型（约 133MB）；tiny/small 为可选，运行 `python download_models.py --size tiny|small`
- GPU 需安装 `onnxruntime-gpu` 与匹配的 CUDA/cuDNN；Linux 服务器需将 cuDNN 库加入 `LD_LIBRARY_PATH`
- ORT 1.20+ 请勿使用 `DEFAULT` cuDNN 算法（本库已默认 HEURISTIC）
- `medium` 约需 1.7GB 显存；8GB 显卡建议 `small`（~800MB）
### 协议与致谢

本项目推荐使用 **Apache-2.0 License** 开源（与 PaddleOCR 一致）。

本项目基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 的 PP-OCRv6 模型与相关技术实现，特此致谢 PaddleOCR 及其开发团队。

---

## English

### Introduction

A simple Python OCR library based on Baidu PaddleOCR **PP-OCRv6** models, running on ONNXRuntime only (no PaddleOCR framework). Supports 50-language text detection and recognition with a lightweight API.

### Features

- **Lightweight Wrapper**: One class handles model loading, detection, orientation classification, and recognition
- **ONNXRuntime Inference**: Supports both CPU and GPU (CUDAExecutionProvider)
- **All-in-One Detection & Recognition**: Automatically performs text box detection, angle classification, and text recognition
- **Easy Visualization**: Built-in `displaybox` method to draw detection boxes
- **Easy Integration**: Can be used as a script or imported as a library in other Python projects
- **HTTP Service**: Production-ready HTTP API service (optional)
- **Three model sizes**: `model_size='medium'` (default, bundled), `'small'` / `'tiny'` (optional, download separately)

### Performance

Default **`model_size='medium'`** — clone and run. For faster inference or lower VRAM, use **`small`** (~800MB) or **`tiny`** (~550MB) after `download_models.py`.

**1080p full OCR** (1920×1080, GPU, HEURISTIC, no cuDNN Fallback):

| Size | Linux A4000 | Linux 2080Ti | Windows RTX 3070 |
|------|-------------|--------------|------------------|
| **small** | **~86 ms** | ~96 ms | ~294 ms |
| medium | ~202 ms | ~244 ms | ~618 ms |

**GPU VRAM** (det + rec + ori loaded): tiny ~550 MiB · small ~800 MiB · medium ~1.7 GiB

> **cuDNN**: ORT 1.20+ `DEFAULT` triggers slow Fallback; this library defaults to **HEURISTIC**.
### Environment & Dependencies

- **Python**: 3.7+
- **Dependencies**:
  - `onnxruntime` or `onnxruntime-gpu`
  - `opencv-python`

> Note: `opencv-python` depends on `numpy`, which is usually installed automatically.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Model Files

This project uses **PaddleOCR PP-OCRv6** ONNX exported models. Choose size via `model_size` (default `medium`):

| Size | Detection | Recognition | Dictionary | Notes |
|------|-----------|-------------|------------|-------|
| `medium` (default) | `ppocr6_m_det.onnx` | `ppocr6_m_rec.onnx` | `ppocr6_dict.txt` (18708 chars, 50 languages) | **Bundled** — clone and run |
| `small` | `ppocr6_s_det.onnx` | `ppocr6_s_rec.onnx` | `ppocr6_s_dict.txt` | Optional: `download_models.py --size small` |
| `tiny` | `ppocr6_t_det.onnx` | `ppocr6_t_rec.onnx` | `ppocr6_t_dict.txt` (6904 chars) | Optional: `download_models.py --size tiny` |

Shared:

- `models/ppocr6_textline_ori.onnx` — Text line orientation

> **Note**: Model files are from the [PaddlePaddle HuggingFace](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6) collection. Please comply with the original license and terms of use.

Place these files in the `models/` directory under the repository root. **Medium** is included in the repo; for tiny/small or to re-download medium:

> ```bash
> pip install huggingface_hub pyyaml
> python download_models.py --size tiny    # or small / medium
> ```

### Installation & Usage

Clone the repository and install dependencies:

```bash
git clone https://github.com/wanghanmin/simple_ppocr6.git
cd simple_ppocr6
pip install -r requirements.txt
```

> If published to PyPI in the future, you can use:
> ```bash
> pip install simple-ppocr6
> ```

### Quick Start

Here is a minimal example to run OCR on a single image:

```python
from simple_ppocr6 import simple_ppocr6

ocr = simple_ppocr6(use_gpu=True)  # default medium, bundled in repo
ocr.run("path/to/your/image.jpg")

for item in ocr.results:
    print(item['text'], item['rec_pos'])
```
For more examples, please refer to `ocr_image.py` and `winclip_ocr.py` in the repository root.

### Examples

#### 1. Command Line OCR

- **`ocr_image.py`**  
  Perform OCR on local images, print all text and box positions with performance statistics, and visualize detection boxes. Supports command line arguments to specify image path (defaults to `example.jpg` if not specified)
  
- **`winclip_ocr.py`**  
  Windows screenshot/clipboard OCR example

Run examples:

```bash
python ocr_image.py                    # example.jpg, default medium
python ocr_image.py your.jpg --gpu
python winclip_ocr.py                  # clipboard OCR, default medium + GPU
python winclip_ocr.py --model-size small   # requires download_models.py --size small
```
#### 2. HTTP API Service

- **`ocr-http-service.py`**  
  Production-ready HTTP API service that accepts base64-encoded images and returns OCR results in JSON format

**Install additional dependencies:**

```bash
pip install flask waitress
```

**Start the service:**

```bash
# Use default settings (0.0.0.0:11005, CPU mode)
python ocr-http-service.py

# Specify port
python ocr-http-service.py --port 8080

# Specify host and port
python ocr-http-service.py --host 127.0.0.1 --port 8080

# Enable GPU mode
python ocr-http-service.py --gpu

# Full configuration
python ocr-http-service.py --host 0.0.0.0 --port 8080 --threads 8 --gpu

# Show help
python ocr-http-service.py --help
```

**Command line options:**

- `--host`: Host address to bind (default: `0.0.0.0`)
- `--port`: Port number to bind (default: `11005`)
- `--threads`: Number of worker threads (default: `4`)
- `--gpu`: Enable GPU acceleration (default: CPU mode)

**API usage example:**

```python
import requests
import base64

# Read image and convert to base64
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Send POST request
response = requests.post(
    "http://localhost:11005/ocr",
    json={"image": image_base64}
)

# View results
result = response.json()
print(f"OCR processing time: {result['processing_time']:.3f}s")
for item in result['results']:
    print(f"Text: {item['text']}")
    print(f"Position: {item['bounding_box']}")
```

**Response format:**

```json
{
  "processing_time": 0.523,
  "decode_time": 0.012,
  "total_time": 0.535,
  "results": [
    {
      "text": "Recognized text",
      "bounding_box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ]
}
```

### Notes

- **Medium** models are bundled (~133MB); tiny/small are optional via `python download_models.py --size tiny|small`
- GPU requires `onnxruntime-gpu` and CUDA/cuDNN; on Linux add cuDNN to `LD_LIBRARY_PATH`
- Do not use ORT `DEFAULT` cuDNN algo on 1.20+ (library defaults to HEURISTIC)
- `medium` needs ~1.7GB VRAM; on 8GB GPUs prefer `small` (~800MB)
### License & Acknowledgements

This project is recommended to use **Apache-2.0 License** (consistent with PaddleOCR).

This project is based on the PP-OCRv6 models and related technologies from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). Special thanks to PaddleOCR and its development team.
