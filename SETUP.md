# Setup Instructions

This guide walks you through setting up the **Autonomous Smart Video Editor** backend on your local machine or a Linux server.

---

## Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.8 | 3.10+ recommended |
| FFmpeg | 4.0 | Must include `ffprobe` |
| pip | 22+ | |

### Install FFmpeg

**macOS (Homebrew)**
```bash
brew install ffmpeg
```

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

**Windows**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add `ffmpeg.exe` + `ffprobe.exe` to your `PATH`.

Verify:
```bash
ffmpeg -version
ffprobe -version
```

---

## Installation

```bash
# Clone the repository (if not already done)
git clone https://github.com/Karthickkavin/intelligent-video-editing-system.git
cd intelligent-video-editing-system/backend

# (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

---

## Running the Server

```bash
# Development (auto-reload, debug logs)
python run.py --env development --host 127.0.0.1 --port 5000

# Production
python run.py --env production --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000` in your browser.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `development` | `development` or `production` |
| `SECRET_KEY` | `dev-secret-key-…` | **Change in production** |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Running with Gunicorn (production)

```bash
pip install gunicorn
gunicorn "app:create_app('production')" \
    --bind 0.0.0.0:5000 \
    --workers 4 \
    --timeout 600
```

---

## Google Colab

1. Open `colab_version/autonomous_video_editor_colab.ipynb` in Colab.
2. Click **Runtime → Run all**.
3. Upload your video when the file picker appears.
4. Download the edited output from the last cell.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ffmpeg not found` | Ensure FFmpeg is on your `PATH` (`which ffmpeg`) |
| `No suitable segments found` | Lower `min_segment_score` or `silence_threshold` |
| Large upload fails | Increase `MAX_CONTENT_LENGTH` in `config.py` |
| Slow processing | Reduce `VIDEO_CRF` range or use `--preset ultrafast` |
