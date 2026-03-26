# Autonomous Smart Video Editor (Shorts Generator)

> A production-ready system that automatically converts long videos into high-quality short clips by detecting silence, low activity, and important segments.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Folder Structure](#folder-structure)
4. [Quick Start – Google Colab](#quick-start--google-colab)
5. [Backend Setup](#backend-setup)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Future Enhancements](#future-enhancements)

---

## Overview

The system analyses a long-form video, automatically detects silence and low-activity regions, scores the remaining segments by importance, and renders a concise edited clip.

**Key capabilities:**

| Feature | Description |
|---|---|
| Silence detection | Audio-threshold-based removal of silent passages |
| Activity scoring | RMS energy analysis to rank segment importance |
| Smart merging | Adjacent high-score segments merged into coherent clips |
| FFmpeg rendering | Stream-copy cutting + H.264/AAC encoding with `faststart` |
| REST API | Upload → Process → Poll → Download workflow |
| Web UI | Drag-and-drop interface for browser-based usage |
| Colab notebook | Single-file version for rapid experimentation |

---

## System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────┐
│ Input Layer │────▶│ Video Processor  │────▶│ Decision Engine  │────▶│ Renderer │
│  (upload)   │     │ silence detect   │     │ score & filter   │     │ FFmpeg   │
└─────────────┘     │ activity score   │     │ merge segments   │     └────┬─────┘
                    └──────────────────┘     └──────────────────┘          │
                                                                            ▼
                                                                    ┌──────────────┐
                                                                    │ Output Video │
                                                                    └──────────────┘
```

### Processing Pipeline

1. **FFprobe** probes the input file for duration, streams, and bitrate.
2. **Audio extraction** — mono 16 kHz PCM WAV via FFmpeg.
3. **Silence detection** — `silencedetect` filter identifies silent intervals.
4. **Activity scoring** — per-second RMS energy → normalised [0, 1] score.
5. **Segment building** — complement of silent regions, split at `MAX_SEGMENT_DURATION`.
6. **Decision Engine** — weighted scoring, threshold filtering, gap merging.
7. **Renderer** — stream-copy cut → concat demuxer → H.264 encode.

---

## Folder Structure

```
intelligent-video-editing-system/
├── colab_version/
│   └── autonomous_video_editor_colab.ipynb   # Self-contained Colab notebook
├── backend/
│   ├── app.py                                # Flask application factory
│   ├── config.py                             # Configuration classes
│   ├── run.py                                # Entry point (CLI)
│   ├── requirements.txt                      # Python dependencies
│   ├── core/
│   │   ├── __init__.py
│   │   ├── video_processor.py               # Analysis pipeline
│   │   ├── decision_engine.py               # Scoring & filtering
│   │   └── renderer.py                      # FFmpeg rendering
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                        # Structured logging
│   │   ├── validators.py                    # Input validation
│   │   └── helpers.py                       # Utility functions
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                        # REST API endpoints
│   ├── static/
│   │   └── index.html                       # Web interface
│   ├── uploads/                             # Temporary upload storage
│   └── outputs/                             # Rendered output videos
├── README.md
└── SETUP.md
```

---

## Quick Start – Google Colab

1. Open `colab_version/autonomous_video_editor_colab.ipynb` in Google Colab.
2. Run **all cells** from top to bottom.
3. Upload your video when prompted.
4. Wait for processing to complete.
5. Download the edited output.

> The notebook installs FFmpeg automatically and requires no local setup.

---

## Backend Setup

See [SETUP.md](SETUP.md) for full installation instructions.

### Minimal quick start

```bash
# 1. Install FFmpeg
# macOS:  brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg

# 2. Install Python dependencies
cd backend
pip install -r requirements.txt

# 3. Start the server
python run.py --env development --port 5000
```

Open `http://localhost:5000` in your browser to use the web UI.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a video file (multipart/form-data, field `file`) |
| `POST` | `/api/process/<job_id>` | Start processing (optional JSON body with params) |
| `GET`  | `/api/status/<job_id>` | Poll job status and progress |
| `GET`  | `/api/download/<job_id>` | Download the completed output video |
| `DELETE` | `/api/job/<job_id>` | Delete job files and record |

### Processing parameters (JSON body for `/api/process`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `silence_threshold` | float | -40 | Silence threshold in dB (–80 to 0) |
| `min_segment_score` | float | 0.3 | Minimum importance score (0–1) |
| `min_segment_duration` | float | 2.0 | Minimum segment duration in seconds |

### Example workflow

```bash
# Upload
curl -X POST http://localhost:5000/api/upload \
     -F "file=@my_video.mp4"
# → {"job_id": "abc123", ...}

# Process
curl -X POST http://localhost:5000/api/process/abc123 \
     -H "Content-Type: application/json" \
     -d '{"silence_threshold": -35, "min_segment_score": 0.4}'

# Poll
curl http://localhost:5000/api/status/abc123
# → {"status": "processing", "progress": 60, ...}

# Download (when status = "completed")
curl -O http://localhost:5000/api/download/abc123
```

---

## Configuration

Edit `backend/config.py` or set environment variables:

| Config key | Default | Description |
|---|---|---|
| `SILENCE_THRESHOLD` | -40 dB | Audio level below which audio is considered silent |
| `SILENCE_MIN_DURATION` | 1.5 s | Minimum silence length to be removed |
| `MIN_SEGMENT_SCORE` | 0.3 | Discard segments scoring below this |
| `MIN_SEGMENT_DURATION` | 2.0 s | Minimum kept segment length |
| `MAX_SEGMENT_DURATION` | 60.0 s | Maximum segment length before splitting |
| `GAP_FILL_THRESHOLD` | 0.5 s | Merge segments with gap smaller than this |
| `VIDEO_PRESET` | fast | FFmpeg encoding speed preset |
| `VIDEO_CRF` | 23 | Constant Rate Factor (lower = higher quality) |
| `AUDIO_BITRATE` | 128k | Output audio bitrate |
| `FFMPEG_THREADS` | 4 | FFmpeg thread count |
| `OUTPUT_RESOLUTION` | None | Scale output, e.g. `"1280x720"` |

---

## Future Enhancements

### Phase 2 – Optimisation
- [ ] Parallel segment processing with `concurrent.futures`
- [ ] Redis-based job queue (Celery)
- [ ] GPU-accelerated encoding (`h264_nvenc`)
- [ ] LRU cache for repeated analysis of the same file

### Phase 3 – Intelligence
- [ ] **Speech-to-text** (Whisper) for transcript-based scoring
- [ ] **Emotion detection** in audio (valence/arousal scoring)
- [ ] **Scene classification** with lightweight CNN
- [ ] **Face detection** weighting (segments with faces score higher)
- [ ] **Highlight detection** with pre-trained sports/action models

### Phase 4 – Platform
- [ ] User accounts & persistent job history
- [ ] S3/GCS storage backend
- [ ] Docker image + Kubernetes deployment
- [ ] Webhook notifications on job completion
