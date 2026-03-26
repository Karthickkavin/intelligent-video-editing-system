"""
Video Processor module.
Handles video loading, frame extraction, metadata extraction, and output generation.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .utils import ensure_dir, frame_to_timestamp, setup_logging

logger = logging.getLogger("video_editing_system.video_processor")


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""

    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str = ""
    file_size_mb: float = 0.0

    def __str__(self) -> str:
        return (
            f"VideoMetadata(path={os.path.basename(self.path)}, "
            f"{self.width}x{self.height}, {self.fps:.2f}fps, "
            f"{self.total_frames} frames, {self.duration:.2f}s)"
        )


@dataclass
class VideoClip:
    """Represents a video clip segment."""

    start_frame: int
    end_frame: int
    fps: float
    label: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def start_time(self) -> float:
        return frame_to_timestamp(self.start_frame, self.fps)

    @property
    def end_time(self) -> float:
        return frame_to_timestamp(self.end_frame, self.fps)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __str__(self) -> str:
        return (
            f"VideoClip({self.label}, "
            f"frames {self.start_frame}-{self.end_frame}, "
            f"{self.start_time:.2f}s-{self.end_time:.2f}s, "
            f"score={self.score:.2f})"
        )


class VideoProcessor:
    """
    Core video processing component.

    Handles loading, frame extraction, metadata extraction, and exporting clips.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    def load(self, path: str) -> VideoMetadata:
        """Open a video file and return its metadata."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> (i * 8)) & 0xFF) for i in range(4)]).strip()
        file_size_mb = os.path.getsize(path) / (1024 * 1024)

        if self._cap is not None:
            self._cap.release()

        self._cap = cap
        self._metadata = VideoMetadata(
            path=path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec,
            file_size_mb=file_size_mb,
        )
        logger.info("Loaded video: %s", self._metadata)
        return self._metadata

    @property
    def metadata(self) -> Optional[VideoMetadata]:
        return self._metadata

    def frames(self, skip: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yield (frame_number, frame) tuples from the loaded video.

        Parameters
        ----------
        skip : int
            Yield every ``skip``-th frame (1 = all frames).
        """
        if self._cap is None:
            raise RuntimeError("No video loaded. Call load() first.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        max_frames = self.config.max_frames

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if frame_idx % skip == 0:
                yield frame_idx, frame
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

    def read_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Read a specific frame by its index."""
        if self._cap is None:
            raise RuntimeError("No video loaded. Call load() first.")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()
        return frame if ret else None

    def extract_frames(
        self,
        output_dir: str,
        frame_numbers: Optional[List[int]] = None,
        skip: int = 1,
    ) -> List[str]:
        """
        Extract frames to disk as JPEG images.

        Parameters
        ----------
        output_dir : str
            Destination directory for extracted frames.
        frame_numbers : list[int] | None
            Specific frames to extract; if None all frames are extracted.
        skip : int
            Extract every ``skip``-th frame when ``frame_numbers`` is None.

        Returns
        -------
        list[str]
            Paths of saved images.
        """
        ensure_dir(output_dir)
        saved: List[str] = []

        if frame_numbers is not None:
            for fn in frame_numbers:
                frame = self.read_frame(fn)
                if frame is not None:
                    path = os.path.join(output_dir, f"frame_{fn:06d}.jpg")
                    cv2.imwrite(path, frame)
                    saved.append(path)
        else:
            for idx, frame in self.frames(skip=skip):
                path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)

        logger.info("Extracted %d frames to %s", len(saved), output_dir)
        return saved

    def save_clip(
        self,
        clip: VideoClip,
        output_path: str,
    ) -> bool:
        """
        Save a VideoClip segment to a new video file.

        Parameters
        ----------
        clip : VideoClip
            The clip segment to export.
        output_path : str
            Path for the output video file.

        Returns
        -------
        bool
            True if the clip was saved successfully.
        """
        if self._cap is None or self._metadata is None:
            raise RuntimeError("No video loaded. Call load() first.")

        ensure_dir(os.path.dirname(output_path)) if os.path.dirname(output_path) else None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self._metadata.fps,
            (self._metadata.width, self._metadata.height),
        )
        if not writer.isOpened():
            logger.error("Cannot open VideoWriter for %s", output_path)
            return False

        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)
            for _ in range(clip.end_frame - clip.start_frame):
                ret, frame = self._cap.read()
                if not ret:
                    break
                writer.write(frame)
        finally:
            writer.release()

        logger.info("Saved clip to %s", output_path)
        return True

    def save_clips(self, clips: List[VideoClip], output_dir: str) -> List[str]:
        """Save multiple VideoClip objects to a directory."""
        ensure_dir(output_dir)
        paths: List[str] = []
        for i, clip in enumerate(clips):
            label = clip.label.replace(" ", "_") or f"clip_{i:03d}"
            out_path = os.path.join(output_dir, f"{label}.mp4")
            if self.save_clip(clip, out_path):
                paths.append(out_path)
        return paths

    def release(self) -> None:
        """Release the video capture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoProcessor":
        return self

    def __exit__(self, *_) -> None:
        self.release()
