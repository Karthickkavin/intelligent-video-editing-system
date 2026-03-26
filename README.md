# 🎬 Intelligent Video Editing System

> **Autonomous AI video editor** – transforms raw footage into polished content using machine learning. Fully compatible with Google Colab (free tier).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthickkavin/intelligent-video-editing-system/blob/main/colab_notebook.ipynb)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎬 Scene detection | Automatically finds shot boundaries using optical flow |
| 🔇 Silence removal | Removes silent gaps from speech/commentary videos |
| ✨ Smart transitions | Crossfades, fades, and dissolves between clips |
| ⚡ Speed adjustment | Slows down action shots, speeds up static scenes |
| 🔍 Auto-zoom | Centres the frame on detected faces or objects |
| 🎨 Colour correction | Normalises brightness and contrast across the video |
| 🔊 Audio normalisation | Levels audio to a consistent loudness target |
| 📦 Quality presets | 480p / 720p / 1080p with optimised codec settings |
| 🔁 Batch processing | Edit multiple videos in one command |
| 🖥️ Gradio Web UI | Interactive interface that runs inside Colab |

---

## 📁 Project Structure

```
intelligent-video-editing-system/
├── colab_notebook.ipynb      ← One-click Colab notebook
├── ai_engine/
│   ├── __init__.py
│   ├── analyzer.py           ← Scene/audio/face analysis
│   ├── editor.py             ← Edit-plan creation
│   ├── renderer.py           ← Final video composition
│   └── models.py             ← ML model management
├── utils/
│   ├── __init__.py
│   ├── config.py             ← All tuneable parameters
│   ├── logger.py             ← Rotating file + console logging
│   └── helpers.py            ← Shared utilities
├── requirements.txt
├── example_usage.py          ← CLI + importable pipeline function
└── README.md
```

---

## 🚀 Quick Start

### Option A – Google Colab (recommended for beginners)

1. Click the **Open in Colab** badge above.
2. Run **Cell 1** – installs all dependencies automatically.
3. Run **Cell 2** – mounts your Google Drive.
4. Run **Cell 3** – launches the Gradio web UI.
5. Upload a video, choose settings, click **Edit Video**.
6. Download the result.

### Option B – Local installation

```bash
# 1. Clone the repository
git clone https://github.com/Karthickkavin/intelligent-video-editing-system.git
cd intelligent-video-editing-system

# 2. Install dependencies (Python 3.9+)
pip install -r requirements.txt

# 3. Make sure FFmpeg is installed
# Ubuntu/Debian:  sudo apt install ffmpeg
# macOS:          brew install ffmpeg
# Windows:        https://ffmpeg.org/download.html

# 4. Edit a video
python example_usage.py --input my_video.mp4 --quality 720p --intensity medium
```

---

## 🐍 Python API

```python
from example_usage import run_pipeline

# Simple one-liner
output = run_pipeline("raw_footage.mp4", quality="720p", intensity="medium")
print(f"Edited video saved to: {output}")
```

### Fine-grained control

```python
from utils.config import Config
from ai_engine import Analyzer, VideoEditor, Renderer

cfg = Config(
    quality="1080p",
    intensity="heavy",
    enable_object_detection=False,   # set True for YOLOv5 (slower)
    enable_auto_zoom=True,
)

analysis  = Analyzer(cfg).analyze("input.mp4")
edit_plan = VideoEditor(cfg).create_edit_plan(analysis)
Renderer(cfg).render("input.mp4", edit_plan, "output.mp4")
```

### Batch processing

```python
from example_usage import run_batch

outputs = run_batch(
    ["clip1.mp4", "clip2.mp4", "clip3.mp4"],
    output_dir="edited",
    quality="720p",
    intensity="light",
)
```

---

## ⚙️ Configuration

`Config` accepts the following key parameters:

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `quality` | `480p` `720p` `1080p` | `720p` | Output resolution & bitrate |
| `intensity` | `light` `medium` `heavy` | `medium` | How aggressively to edit |
| `enable_scene_detection` | bool | `True` | Detect and use scene boundaries |
| `enable_silence_removal` | bool | `True` | Cut silent segments |
| `enable_transitions` | bool | `True` | Add crossfades between clips |
| `enable_speed_adjustment` | bool | `True` | Vary speed based on motion |
| `enable_auto_zoom` | bool | `True` | Zoom in on faces/objects |
| `enable_color_correction` | bool | `True` | Normalise brightness |
| `enable_audio_normalization` | bool | `True` | Level audio loudness |
| `enable_face_detection` | bool | `True` | Run Haar-cascade or MediaPipe |
| `enable_object_detection` | bool | `False` | Run YOLOv5 (heavier) |
| `use_gpu` | bool | `True` | Use CUDA when available |

### Intensity presets

| Setting | light | medium | heavy |
|---------|-------|--------|-------|
| Scene threshold | 0.60 | 0.45 | 0.30 |
| Min scene duration | 2.0 s | 1.5 s | 0.8 s |
| Silence threshold | −45 dBFS | −40 dBFS | −35 dBFS |
| Transition length | 0.5 s | 0.75 s | 1.0 s |
| Max auto-zoom | 1.10× | 1.20× | 1.40× |

---

## 🛠️ Supported Formats

**Input:** MP4, MOV, AVI, MKV, WebM, FLV  
**Output:** MP4 (H.264 + AAC)

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ffmpeg not found` | Install FFmpeg – see Quick Start above |
| `No module named 'cv2'` | `pip install opencv-python` |
| `No module named 'moviepy'` | `pip install moviepy` |
| Out of memory in Colab | Use `quality="480p"` and `intensity="light"` |
| YOLOv5 download fails | Set `enable_object_detection=False` |
| Black output video | Ensure input video has at least one scene > min duration |

---

## 📊 Data Flow

```
Input Video
    │
    ▼
┌─────────┐
│Analyzer │  Scene detection · Audio analysis · Face/object detection
└────┬────┘
     │  AnalysisResult (dict)
     ▼
┌─────────┐
│ Editor  │  Build EditPlan: cuts · transitions · speed · zoom · colour
└────┬────┘
     │  EditPlan (dict)
     ▼
┌──────────┐
│ Renderer │  Apply plan via MoviePy → FFmpeg → output.mp4
└──────────┘
```

---

## 📄 License

MIT – free for personal and commercial use.
