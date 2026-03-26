"""
Object Detection module.
Detects objects, people, and faces in video frames using MediaPipe and OpenCV.
Tracks object movement across frames and generates editing suggestions.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .utils import draw_bounding_box, frame_to_timestamp

logger = logging.getLogger("video_editing_system.object_detector")


@dataclass
class Detection:
    """A single detected object in a frame."""

    frame_number: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)

    def __str__(self) -> str:
        x, y, w, h = self.bbox
        return (
            f"Detection(frame={self.frame_number}, label={self.label}, "
            f"conf={self.confidence:.2f}, bbox=({x},{y},{w},{h}))"
        )


@dataclass
class Track:
    """Tracked object across multiple frames."""

    track_id: int
    label: str
    detections: List[Detection] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_number if self.detections else 0

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_number if self.detections else 0

    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        return [d.center for d in self.detections]

    def __str__(self) -> str:
        return (
            f"Track(id={self.track_id}, label={self.label}, "
            f"frames {self.start_frame}-{self.end_frame}, "
            f"{len(self.detections)} detections)"
        )


class ObjectDetector:
    """
    Detect objects, people, and faces in video frames.

    Attempts to use MediaPipe for face detection when available, and falls back
    to OpenCV's Haar cascades. Object tracking is performed using a simple IoU-
    based tracker to maintain consistent IDs across frames.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        self._mp_face_detection = None
        self._tracks: Dict[int, Track] = {}
        self._next_track_id: int = 0
        self._init_detectors()

    def _init_detectors(self) -> None:
        """Initialise available detection backends."""
        # Try MediaPipe face detection first
        if self.config.face_detection_enabled:
            try:
                import mediapipe as mp  # noqa: PLC0415

                mp_face = mp.solutions.face_detection
                self._mp_face_detection = mp_face.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self.config.object_confidence_threshold,
                )
                logger.debug("MediaPipe face detection initialised")
            except Exception:
                logger.debug("MediaPipe not available; falling back to Haar cascade")

        # Haar cascade fallback
        if self._mp_face_detection is None and self.config.face_detection_enabled:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self._face_cascade = cascade
                logger.debug("Haar cascade face detector loaded")
            else:
                logger.warning("Haar cascade not found; face detection disabled")

    def detect_frame(self, frame: np.ndarray, frame_number: int = 0) -> List[Detection]:
        """
        Run all enabled detectors on a single frame.

        Returns
        -------
        list[Detection]
            All detections found in this frame.
        """
        detections: List[Detection] = []

        if self.config.face_detection_enabled:
            detections.extend(self._detect_faces(frame, frame_number))

        return detections

    def _detect_faces(self, frame: np.ndarray, frame_number: int) -> List[Detection]:
        """Detect faces using MediaPipe or Haar cascade."""
        detections: List[Detection] = []
        h, w = frame.shape[:2]

        if self._mp_face_detection is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._mp_face_detection.process(rgb)
                if results.detections:
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        bw = int(bbox.width * w)
                        bh = int(bbox.height * h)
                        score = det.score[0] if det.score else 0.0
                        detections.append(
                            Detection(
                                frame_number=frame_number,
                                label="face",
                                confidence=float(score),
                                bbox=(max(0, x), max(0, y), bw, bh),
                            )
                        )
                return detections
            except Exception as exc:
                logger.debug("MediaPipe detection error: %s", exc)

        if self._face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for x, y, fw, fh in (faces if len(faces) else []):
                detections.append(
                    Detection(
                        frame_number=frame_number,
                        label="face",
                        confidence=1.0,
                        bbox=(int(x), int(y), int(fw), int(fh)),
                    )
                )

        return detections

    def detect_sequence(
        self, frames: List[Tuple[int, np.ndarray]]
    ) -> Dict[int, List[Detection]]:
        """
        Run detection on a sequence of frames.

        Returns
        -------
        dict[int, list[Detection]]
            Mapping from frame_number to list of detections.
        """
        results: Dict[int, List[Detection]] = {}
        for frame_number, frame in frames:
            dets = self.detect_frame(frame, frame_number)
            if dets:
                results[frame_number] = dets
        logger.info(
            "Object detection complete: %d frames with detections", len(results)
        )
        return results

    def track_objects(
        self, frame_detections: Dict[int, List[Detection]]
    ) -> List[Track]:
        """
        Associate detections across frames into tracks using IoU matching.

        Parameters
        ----------
        frame_detections : dict[int, list[Detection]]
            Output of ``detect_sequence()``.

        Returns
        -------
        list[Track]
            All tracks with at least one detection.
        """
        self._tracks = {}
        self._next_track_id = 0
        active_tracks: List[Track] = []

        for frame_number in sorted(frame_detections):
            detections = frame_detections[frame_number]
            matched_track_ids = set()

            for det in detections:
                best_iou = 0.0
                best_track: Optional[Track] = None

                for track in active_tracks:
                    if track.track_id in matched_track_ids:
                        continue
                    last_det = track.detections[-1]
                    iou = self._iou(det.bbox, last_det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_track = track

                iou_threshold = 0.3
                if best_track is not None and best_iou >= iou_threshold:
                    det.track_id = best_track.track_id
                    best_track.detections.append(det)
                    matched_track_ids.add(best_track.track_id)
                else:
                    new_track = Track(
                        track_id=self._next_track_id, label=det.label
                    )
                    self._next_track_id += 1
                    det.track_id = new_track.track_id
                    new_track.detections.append(det)
                    active_tracks.append(new_track)
                    self._tracks[new_track.track_id] = new_track

        logger.info("Tracking complete: %d tracks", len(self._tracks))
        return list(self._tracks.values())

    @staticmethod
    def _iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union for two (x, y, w, h) boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    def generate_suggestions(
        self,
        tracks: List[Track],
        fps: float = 25.0,
    ) -> List[str]:
        """
        Generate editing suggestions based on detected objects and tracks.

        Parameters
        ----------
        tracks : list[Track]
            Tracks from ``track_objects()``.
        fps : float
            Video frame rate for timestamp calculation.

        Returns
        -------
        list[str]
            Human-readable editing suggestions.
        """
        suggestions: List[str] = []

        face_tracks = [t for t in tracks if t.label == "face"]
        if face_tracks:
            longest = max(face_tracks, key=lambda t: len(t.detections))
            start_ts = frame_to_timestamp(longest.start_frame, fps)
            end_ts = frame_to_timestamp(longest.end_frame, fps)
            suggestions.append(
                f"Face detected from {start_ts:.2f}s to {end_ts:.2f}s — "
                f"consider keeping this segment."
            )

        if len(face_tracks) > 1:
            suggestions.append(
                f"{len(face_tracks)} distinct face tracks detected — "
                f"possible interview or multi-person scene."
            )

        return suggestions

    def annotate_frame(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """Draw detection bounding boxes onto a copy of the frame."""
        annotated = frame.copy()
        for det in detections:
            label = f"{det.label} {det.confidence:.0%}"
            color = (0, 255, 0) if det.label == "face" else (255, 128, 0)
            annotated = draw_bounding_box(annotated, det.bbox, label, color)
        return annotated

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._mp_face_detection is not None:
            try:
                self._mp_face_detection.close()
            except Exception:
                pass
            self._mp_face_detection = None

    def __enter__(self) -> "ObjectDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
