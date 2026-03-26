"""
Renderer – FFmpeg-based video cutting and merging.

Responsibilities:
  - Cut individual video segments from the source file
  - Concatenate segments into a single output video
  - Apply encoding optimizations (preset, CRF, threading)
  - Optionally scale output resolution
"""

import os
import subprocess
import tempfile
from typing import List, Optional

from core.video_processor import VideoSegment
from utils.logger import get_logger
from utils.helpers import ensure_directory

logger = get_logger(__name__)


class Renderer:
    """
    Render a final video from a list of VideoSegment objects.

    Parameters
    ----------
    config : object
        Application config (VIDEO_PRESET, VIDEO_CRF, AUDIO_BITRATE,
        FFMPEG_THREADS, OUTPUT_RESOLUTION, OUTPUT_FORMAT).
    """

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        source_path: str,
        segments: List[VideoSegment],
        output_path: str,
    ) -> str:
        """
        Cut *segments* from *source_path* and concatenate them into
        *output_path*.  Returns the path to the finished file.
        """
        if not segments:
            raise ValueError("No segments provided to renderer")

        ensure_directory(os.path.dirname(output_path))

        logger.info(
            "Rendering %d segments from %s → %s",
            len(segments),
            source_path,
            output_path,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            clip_paths = self._cut_segments(source_path, segments, tmpdir)
            if len(clip_paths) == 1:
                self._encode(clip_paths[0], output_path)
            else:
                concat_list = self._write_concat_list(clip_paths, tmpdir)
                self._concat_and_encode(concat_list, output_path)

        logger.info("Rendering complete: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cut_segments(
        self,
        source_path: str,
        segments: List[VideoSegment],
        tmpdir: str,
    ) -> List[str]:
        """Cut each segment and save as a lossless copy in *tmpdir*."""
        clip_paths: List[str] = []
        for idx, seg in enumerate(segments):
            clip_path = os.path.join(tmpdir, f"clip_{idx:04d}.mp4")
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(seg.start),
                "-to", str(seg.end),
                "-i", source_path,
                "-c", "copy",            # stream-copy: fast and lossless
                "-avoid_negative_ts", "make_zero",
                clip_path,
            ]
            self._run(cmd, f"cutting segment {idx}")
            clip_paths.append(clip_path)
            logger.debug(
                "Cut segment %d [%.2f – %.2f] → %s",
                idx, seg.start, seg.end, clip_path,
            )
        return clip_paths

    @staticmethod
    def _write_concat_list(clip_paths: List[str], tmpdir: str) -> str:
        """Write an FFmpeg concat demuxer list file and return its path."""
        list_path = os.path.join(tmpdir, "concat.txt")
        with open(list_path, "w", encoding="utf-8") as fh:
            for path in clip_paths:
                fh.write(f"file '{path}'\n")
        return list_path

    def _concat_and_encode(self, concat_list: str, output_path: str) -> None:
        """Concatenate clips and encode to *output_path*."""
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            *self._encode_flags(),
            output_path,
        ]
        self._run(cmd, "concatenating & encoding")

    def _encode(self, input_path: str, output_path: str) -> None:
        """Re-encode a single clip to *output_path*."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            *self._encode_flags(),
            output_path,
        ]
        self._run(cmd, "encoding single clip")

    def _encode_flags(self) -> List[str]:
        """Build the FFmpeg encoding flags from config."""
        flags = [
            "-c:v", "libx264",
            "-preset", self.config.VIDEO_PRESET,
            "-crf", str(self.config.VIDEO_CRF),
            "-c:a", "aac",
            "-b:a", self.config.AUDIO_BITRATE,
            "-threads", str(self.config.FFMPEG_THREADS),
            "-movflags", "+faststart",
        ]

        resolution: Optional[str] = getattr(self.config, "OUTPUT_RESOLUTION", None)
        if resolution:
            flags += ["-vf", f"scale={resolution}"]

        return flags

    @staticmethod
    def _run(cmd: List[str], step: str) -> None:
        """Execute an FFmpeg command, raising RuntimeError on failure."""
        logger.debug("FFmpeg [%s]: %s", step, " ".join(cmd))
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"FFmpeg failed at step '{step}' (code {result.returncode}): {stderr[-500:]}"
            )
