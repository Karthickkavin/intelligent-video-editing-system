"""
Video Processor – audio/video stream extraction and analysis.

Responsibilities:
  - Probe video metadata via FFprobe
  - Extract raw PCM audio for analysis
  - Detect silence intervals
  - Estimate per-second activity scores
  - Produce a list of candidate VideoSegment objects
"""

import json
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoSegment:
    """Represents a candidate video segment."""

    start: float                    # start time in seconds
    end: float                      # end time in seconds
    score: float = 0.0              # importance score 0.0 – 1.0
    has_audio: bool = True
    has_motion: bool = True
    tags: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class VideoProcessor:
    """
    Analyses a video file and returns scored candidate segments.

    Parameters
    ----------
    config : object
        Application config object (must expose SILENCE_THRESHOLD,
        SILENCE_MIN_DURATION, MIN_SEGMENT_DURATION, MAX_SEGMENT_DURATION).
    """

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, video_path: str) -> List[VideoSegment]:
        """
        Full analysis pipeline.

        Returns a list of *candidate* VideoSegment objects **before** scoring
        and filtering (that is the Decision Engine's job).
        """
        logger.info("Starting analysis of: %s", video_path)

        metadata = self._probe_metadata(video_path)
        duration = metadata.get("duration", 0.0)
        logger.info("Video duration: %.2f s", duration)

        audio_path = self._extract_audio(video_path)
        try:
            silence_intervals = self._detect_silence(audio_path, duration)
            activity_scores = self._estimate_activity(audio_path, duration)
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

        segments = self._build_segments(duration, silence_intervals, activity_scores)
        logger.info("Produced %d candidate segments", len(segments))
        return segments

    def get_metadata(self, video_path: str) -> dict:
        """Return basic metadata dict for *video_path*."""
        return self._probe_metadata(video_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_metadata(self, video_path: str) -> dict:
        """Run ffprobe and return a metadata dictionary."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            data = json.loads(result.stdout)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"FFprobe failed for {video_path}: {exc.stderr.decode()}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("FFprobe returned invalid JSON") from exc

        fmt = data.get("format", {})
        metadata = {
            "duration": float(fmt.get("duration", 0)),
            "size_bytes": int(fmt.get("size", 0)),
            "bit_rate": int(fmt.get("bit_rate", 0)),
            "format_name": fmt.get("format_name", ""),
            "streams": [],
        }

        for stream in data.get("streams", []):
            metadata["streams"].append(
                {
                    "codec_type": stream.get("codec_type"),
                    "codec_name": stream.get("codec_name"),
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "r_frame_rate": stream.get("r_frame_rate"),
                    "sample_rate": stream.get("sample_rate"),
                    "channels": stream.get("channels"),
                }
            )

        return metadata

    def _extract_audio(self, video_path: str) -> Optional[str]:
        """
        Extract mono 16 kHz PCM audio to a temporary WAV file.

        Returns the path to the temp WAV, or None if there is no audio stream.
        """
        # Check for audio stream
        metadata = self._probe_metadata(video_path)
        has_audio = any(
            s.get("codec_type") == "audio" for s in metadata.get("streams", [])
        )
        if not has_audio:
            logger.warning("No audio stream found in %s", video_path)
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            tmp_path,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Audio extraction failed: %s", exc)
            return None

        return tmp_path

    def _detect_silence(
        self, audio_path: Optional[str], duration: float
    ) -> List[Tuple[float, float]]:
        """
        Use FFmpeg's silencedetect filter to find silence intervals.

        Returns list of (start, end) tuples in seconds.
        """
        if audio_path is None:
            return [(0.0, duration)]

        threshold = self.config.SILENCE_THRESHOLD
        min_dur = self.config.SILENCE_MIN_DURATION

        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-af", f"silencedetect=noise={threshold}dB:d={min_dur}",
            "-f", "null",
            "-",
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error("ffmpeg not found")
            return []

        stderr = result.stderr.decode("utf-8", errors="replace")
        intervals: List[Tuple[float, float]] = []
        start: Optional[float] = None

        for line in stderr.splitlines():
            if "silence_start" in line:
                try:
                    start = float(line.split("silence_start:")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif "silence_end" in line and start is not None:
                try:
                    end = float(line.split("silence_end:")[1].split("|")[0].strip())
                    intervals.append((start, end))
                    start = None
                except (IndexError, ValueError):
                    pass

        # If silence started but never ended, the tail is silent
        if start is not None:
            intervals.append((start, duration))

        logger.debug("Detected %d silence interval(s)", len(intervals))
        return intervals

    def _estimate_activity(
        self, audio_path: Optional[str], duration: float
    ) -> List[float]:
        """
        Estimate per-second audio energy scores.

        Returns a list of floats (one per second) in the range [0.0, 1.0].
        """
        if audio_path is None or duration <= 0:
            return [0.0] * max(1, int(duration))

        num_seconds = max(1, int(duration) + 1)
        scores = [0.0] * num_seconds

        try:
            samples = self._read_pcm_samples(audio_path)
            if not samples:
                return scores

            sample_rate = 16000
            for sec in range(num_seconds):
                s = sec * sample_rate
                e = s + sample_rate
                chunk = samples[s:e]
                if chunk:
                    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
                    # Normalise to [0, 1] – 32767 is the max for int16
                    scores[sec] = min(1.0, rms / 3276.7)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Activity estimation failed: %s", exc)

        return scores

    @staticmethod
    def _read_pcm_samples(wav_path: str) -> List[int]:
        """
        Read 16-bit PCM samples from a WAV file without external dependencies.

        Returns a flat list of integer sample values.
        """
        with open(wav_path, "rb") as fh:
            header = fh.read(44)
            if len(header) < 44 or header[:4] != b"RIFF":
                return []
            data_size = struct.unpack_from("<I", header, 40)[0]
            raw = fh.read(data_size)

        num_samples = len(raw) // 2
        return list(struct.unpack(f"<{num_samples}h", raw[: num_samples * 2]))

    def _build_segments(
        self,
        duration: float,
        silence_intervals: List[Tuple[float, float]],
        activity_scores: List[float],
    ) -> List[VideoSegment]:
        """
        Build candidate segments from the complement of silence intervals.

        Non-silent regions are split into segments respecting MAX_SEGMENT_DURATION.
        """
        min_dur = self.config.MIN_SEGMENT_DURATION
        max_dur = self.config.MAX_SEGMENT_DURATION

        # Build a sorted set of silent time ranges
        silent_set: List[Tuple[float, float]] = sorted(silence_intervals)

        # Candidate regions = complement of silence
        candidate_regions: List[Tuple[float, float]] = []
        prev_end = 0.0
        for s_start, s_end in silent_set:
            if s_start > prev_end:
                candidate_regions.append((prev_end, s_start))
            prev_end = max(prev_end, s_end)
        if prev_end < duration:
            candidate_regions.append((prev_end, duration))

        segments: List[VideoSegment] = []
        for region_start, region_end in candidate_regions:
            region_dur = region_end - region_start
            if region_dur < min_dur:
                continue

            # Split long regions
            cursor = region_start
            while cursor < region_end:
                seg_end = min(cursor + max_dur, region_end)
                if seg_end - cursor < min_dur:
                    break

                score = self._score_region(cursor, seg_end, activity_scores)
                seg = VideoSegment(
                    start=cursor,
                    end=seg_end,
                    score=score,
                    has_audio=True,
                    has_motion=score > 0.1,
                )
                segments.append(seg)
                cursor = seg_end

        return segments

    @staticmethod
    def _score_region(
        start: float, end: float, activity_scores: List[float]
    ) -> float:
        """Compute an average activity score for the region [start, end]."""
        s_idx = int(start)
        e_idx = min(int(end) + 1, len(activity_scores))
        if s_idx >= e_idx:
            return 0.0
        region_scores = activity_scores[s_idx:e_idx]
        return sum(region_scores) / max(1, len(region_scores))
