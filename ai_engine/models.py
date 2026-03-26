"""
ML model management for the Intelligent Video Editing System.

This module handles lazy loading and caching of pre-trained models so that
heavy imports (torch, transformers, etc.) only happen when they are actually
needed.  Each model is loaded at most once per process.
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("video_editor.models")


class ModelManager:
    """Central registry for pre-trained ML models.

    Models are loaded lazily on first access and then cached in memory.

    Supported models
    ----------------
    face_detector   : OpenCV Haar-cascade or MediaPipe face detection.
    object_detector : YOLOv5-nano via torch.hub (optional, off by default).
    audio_classifier: librosa-based feature extractor (no large model needed).
    """

    def __init__(self, use_gpu: bool = True) -> None:
        self.use_gpu = use_gpu
        self._cache: Dict[str, Any] = {}
        self._device: Optional[str] = None

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch  # type: ignore

                self._device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
            except ImportError:
                self._device = "cpu"
        return self._device

    # ------------------------------------------------------------------
    # Face detection
    # ------------------------------------------------------------------

    def get_face_detector(self, backend: str = "haarcascade") -> Any:
        """Return a face-detection callable.

        Parameters
        ----------
        backend:
            ``"haarcascade"`` (OpenCV, always available) or
            ``"mediapipe"`` (optional, more accurate).

        Returns
        -------
        A detector object with a ``detect(frame)`` method that returns a list
        of ``(x, y, w, h)`` bounding boxes.
        """
        key = f"face_{backend}"
        if key in self._cache:
            return self._cache[key]

        if backend == "mediapipe":
            detector = self._load_mediapipe_face()
        else:
            detector = self._load_haar_face()

        self._cache[key] = detector
        return detector

    def _load_haar_face(self) -> Any:
        try:
            import cv2  # type: ignore

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)

            class HaarDetector:
                def __init__(self, clf):
                    self._clf = clf

                def detect(self, frame):
                    import cv2 as _cv2

                    gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
                    faces = self._clf.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    if len(faces) == 0:
                        return []
                    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

            logger.info("Loaded Haar-cascade face detector.")
            return HaarDetector(cascade)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load Haar-cascade detector: %s", exc)
            return _NullDetector()

    def _load_mediapipe_face(self) -> Any:
        try:
            import mediapipe as mp  # type: ignore

            mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )

            class MediaPipeDetector:
                def __init__(self, detector):
                    self._det = detector

                def detect(self, frame):
                    import cv2 as _cv2

                    rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                    result = self._det.process(rgb)
                    boxes = []
                    if result.detections:
                        h, w = frame.shape[:2]
                        for det in result.detections:
                            bb = det.location_data.relative_bounding_box
                            x = int(bb.xmin * w)
                            y = int(bb.ymin * h)
                            bw = int(bb.width * w)
                            bh = int(bb.height * h)
                            boxes.append((x, y, bw, bh))
                    return boxes

            logger.info("Loaded MediaPipe face detector.")
            return MediaPipeDetector(mp_face)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MediaPipe unavailable, falling back to Haar: %s", exc)
            return self._load_haar_face()

    # ------------------------------------------------------------------
    # Object detection (optional)
    # ------------------------------------------------------------------

    def get_object_detector(self) -> Any:
        """Return a YOLOv5-nano detector (loads from torch.hub).

        Returns a ``_NullDetector`` if torch or the model is unavailable.
        """
        key = "object_yolo"
        if key in self._cache:
            return self._cache[key]

        detector = self._load_yolo()
        self._cache[key] = detector
        return detector

    def _load_yolo(self) -> Any:
        try:
            import torch  # type: ignore

            model = torch.hub.load(
                "ultralytics/yolov5",
                "yolov5n",
                pretrained=True,
                verbose=False,
                trust_repo=True,
            )
            model.to(self.device)
            model.eval()

            class YoloDetector:
                def __init__(self, m):
                    self._model = m

                def detect(self, frame):
                    results = self._model(frame)
                    detections = []
                    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                        x1, y1, x2, y2 = map(int, xyxy)
                        detections.append(
                            {
                                "bbox": (x1, y1, x2 - x1, y2 - y1),
                                "confidence": float(conf),
                                "class": int(cls),
                                "label": results.names[int(cls)],
                            }
                        )
                    return detections

            logger.info("Loaded YOLOv5-nano object detector (device=%s).", self.device)
            return YoloDetector(model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLOv5 unavailable: %s", exc)
            return _NullDetector()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Release all cached models to free memory."""
        self._cache.clear()
        try:
            import torch  # type: ignore
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Model cache cleared.")


# ---------------------------------------------------------------------------
# Null detector (safe fallback)
# ---------------------------------------------------------------------------


class _NullDetector:
    """Drop-in detector that always returns an empty list."""

    def detect(self, frame) -> list:  # noqa: ARG002
        return []
