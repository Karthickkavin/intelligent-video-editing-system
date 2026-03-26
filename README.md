# Intelligent Video Editing System

AI-powered video editing system that transforms raw footage into polished content using machine learning. It automates scene detection, trimming, transitions, and audio synchronisation, enabling fast, intelligent, and high-quality video production.

## Features

- **Scene Detection** — Automatic shot-boundary detection using histogram analysis
- **Object & Face Detection** — MediaPipe / OpenCV Haar-cascade face detection with IoU-based multi-frame tracking
- **Motion Analysis** — Dense optical-flow motion intensity scoring and highlight identification
- **Auto-Editing Engine** — Combines all signals to generate cut suggestions, highlight reels, and video summaries
- **Configurable Profiles** — `fast_paced`, `documentary`, `highlights`, and `summary` presets

## Project Structure

```
intelligent-video-editing-system/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration & editing profiles
│   ├── utils.py           # Shared utility functions
│   ├── video_processor.py # Video loading, frame extraction, clip export
│   ├── scene_detector.py  # Scene-change & key-frame detection
│   ├── object_detector.py # Face/object detection & tracking
│   ├── motion_analyzer.py # Optical-flow motion analysis
│   └── auto_editor.py     # Auto-editing engine & plan generation
├── tests/
│   ├── test_video_processor.py
│   ├── test_scene_detector.py
│   └── test_object_detector.py
├── examples/
│   └── example_usage.py
├── requirements.txt
├── setup.py
└── README.md
```

## Requirements

- Python 3.8+
- OpenCV (`opencv-python >= 4.8`)
- NumPy (`numpy >= 1.24`)
- MediaPipe (`mediapipe >= 0.10`, optional — falls back to Haar cascades)

## Installation

```bash
pip install -r requirements.txt
# or install as a package
pip install -e .
```

## Quick Start

```python
from src.config import Config, EditingProfile
from src.video_processor import VideoProcessor
from src.auto_editor import AutoEditor

config = Config()
config.apply_profile(EditingProfile.HIGHLIGHTS)

with VideoProcessor(config) as processor:
    metadata = processor.load("my_video.mp4")
    frames = list(processor.frames(skip=2))

with AutoEditor(config) as editor:
    plan = editor.generate_plan(frames, fps=metadata.fps)

print(plan)
for seg in plan.summary_segments:
    print(seg)
```

## Command-Line Example

```bash
python examples/example_usage.py \
    --video path/to/video.mp4 \
    --output output/ \
    --profile highlights
```

## Running Tests

```bash
python -m pytest tests/ -v
```
