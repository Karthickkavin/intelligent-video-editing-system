# 🎬 AutoEdit AI — Intelligent Video Editing System

AutoEdit AI is a full-stack autonomous video editor that uses computer vision and audio processing to automatically edit your videos — detecting scenes, trimming dead space, adding transitions, and syncing audio.

## ✨ Features

- 🤖 **AI Scene Detection** — Automatically identifies scene changes using frame-difference analysis
- ✂️ **Smart Trimming** — Removes dark, silent, or low-quality segments
- 🎞️ **Smooth Transitions** — Concatenates clips with seamless composition
- 🎵 **Audio Synchronization** — Normalizes and fades audio for professional output
- 📊 **Real-time Progress** — Live dashboard showing every processing step
- 🌐 **Modern UI** — Dark-themed, glass-morphism React frontend

## 🛠️ Tech Stack

| Layer     | Technology                              |
|-----------|-----------------------------------------|
| Frontend  | React 18, Vite 5, Axios                 |
| Backend   | Python 3.11, Flask 3, MoviePy, OpenCV   |
| Processing| FFmpeg, libx264, AAC                    |
| Deployment| Docker, Docker Compose, Nginx, Gunicorn |

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
git clone <repo-url>
cd intelligent-video-editing-system
docker-compose up --build
```

- Frontend: http://localhost:3000  
- Backend API: http://localhost:5000

### Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 📡 API Reference

| Method | Endpoint                  | Description                    |
|--------|---------------------------|--------------------------------|
| POST   | `/api/upload`             | Upload video for processing    |
| GET    | `/api/status/<job_id>`    | Get job status and progress    |
| GET    | `/api/download/<job_id>`  | Download the edited video      |
| GET    | `/api/jobs`               | List all processing jobs       |

### Upload Request
```
POST /api/upload
Content-Type: multipart/form-data
Body: video=<file>
```

### Status Response
```json
{
  "id": "uuid",
  "status": "processing",
  "progress": 65,
  "message": "Adding transitions...",
  "result_filename": null,
  "created_at": "2024-01-01T00:00:00"
}
```

## 📸 Screenshots

> Upload → AI Processing Dashboard → Download Edited Video

## 📁 Project Structure

```
├── backend/          # Flask API + video processing pipeline
│   ├── app/
│   │   ├── routes/   # API endpoints
│   │   └── services/ # Scene detection, video processing, audio sync
│   └── run.py
├── frontend/         # React + Vite UI
│   └── src/
│       └── components/
└── docker-compose.yml
```

## 📝 License

MIT
AI-powered video editing system that transforms raw footage into polished content using machine learning. It automates scene detection, trimming, transitions, and audio synchronization, enabling fast, intelligent, and high-quality video production.
