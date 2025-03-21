"""
Mock implementations of core interfaces for testing.

This module provides mock implementations of core interfaces that can be used in tests
to avoid dependencies on external systems and make tests more deterministic.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydub import AudioSegment

from pyhearingai.core.domain.api_constraints import ApiProvider, ApiSizeLimit
from pyhearingai.core.domain.audio_quality import AudioQualitySpecification
from pyhearingai.core.ports import AudioFormatService, ChunkingService, Diarizer, Transcriber

logger = logging.getLogger(__name__)


class MockAudioFormatService(AudioFormatService):
    """
    Mock implementation of AudioFormatService for testing.

    Provides a simplified in-memory implementation that doesn't require FFmpeg.
    """

    def get_audio_metadata(self, audio_path: Path) -> Dict[str, Any]:
        """Return predefined metadata for the audio file."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # If it's a real audio file, try to get some actual metadata
        if audio_path.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]:
            try:
                audio = AudioSegment.from_file(str(audio_path))
                return {
                    "duration": len(audio) / 1000,  # milliseconds to seconds
                    "size_bytes": os.path.getsize(audio_path),
                    "format": audio_path.suffix.lower().lstrip("."),
                    "sample_rate": audio.frame_rate,
                    "channels": audio.channels,
                    "codec": "pcm_s16le" if audio_path.suffix.lower() == ".wav" else "unknown",
                    "bit_rate": None,
                    "bits_per_sample": audio.sample_width * 8,
                }
            except Exception as e:
                logger.warning(f"Error getting real metadata: {e}")

        # Return mock metadata
        return {
            "duration": 10.0,
            "size_bytes": 1024 * 1024,  # 1MB
            "format": audio_path.suffix.lower().lstrip("."),
            "sample_rate": 16000,
            "channels": 1,
            "codec": "pcm_s16le",
            "bit_rate": 256000,
            "bits_per_sample": 16,
        }

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get the duration of an audio file in seconds."""
        try:
            metadata = self.get_audio_metadata(audio_path)
            return metadata["duration"]
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            raise RuntimeError(f"Failed to get audio duration: {str(e)}") from e

    def extract_audio_segment(
        self,
        audio_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
        quality_spec: Optional[AudioQualitySpecification] = None,
    ) -> Path:
        """Extract a segment of audio and save to a new file."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if start_time >= end_time:
            raise ValueError(f"Invalid time range: start ({start_time}) >= end ({end_time})")

        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)

        try:
            # If it's a real audio file, try to extract a real segment
            if audio_path.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]:
                try:
                    audio = AudioSegment.from_file(str(audio_path))
                    segment = audio[int(start_time * 1000) : int(end_time * 1000)]
                    segment.export(str(output_path), format=output_path.suffix.lower().lstrip("."))
                    return output_path
                except Exception as e:
                    logger.warning(f"Error extracting real segment: {e}")

            # Create a simple synthetic audio segment
            duration_ms = int((end_time - start_time) * 1000)
            sample_rate = 16000 if quality_spec is None else quality_spec.sample_rate
            channels = 1 if quality_spec is None else quality_spec.channels

            # Generate a simple sine wave
            t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000))
            sine_wave = np.sin(2 * np.pi * 440 * t) * 32767  # 440 Hz
            sine_wave = sine_wave.astype(np.int16)

            # Convert to mono or stereo
            if channels == 2:
                sine_wave = np.column_stack((sine_wave, sine_wave))

            # Create audio segment
            segment = AudioSegment(
                sine_wave.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit
                channels=channels,
            )

            # Export to file
            segment.export(str(output_path), format=output_path.suffix.lower().lstrip("."))
            return output_path

        except Exception as e:
            logger.error(f"Error extracting audio segment: {str(e)}")
            raise RuntimeError(f"Failed to extract audio segment: {str(e)}") from e

    def detect_silence(
        self, audio_path: Path, min_silence_duration: float = 0.5, silence_threshold: float = -40
    ) -> List[Dict[str, float]]:
        """Detect silence regions in an audio file."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Return mock silence regions
        return [{"start": 1.0, "end": 1.5}, {"start": 3.0, "end": 3.8}, {"start": 5.5, "end": 6.0}]


class MockChunkingService(ChunkingService):
    """
    Mock implementation of ChunkingService for testing.

    Provides a simplified implementation that doesn't require actual audio processing.
    """

    def __init__(self, audio_format_service: Optional[AudioFormatService] = None):
        """Initialize with an optional audio format service."""
        self.audio_format_service = audio_format_service or MockAudioFormatService()

    def calculate_chunk_boundaries(
        self,
        audio_duration: float,
        chunk_duration: float,
        overlap_duration: float = 0.0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Calculate chunk boundaries for audio."""
        # Validate chunk parameters
        if chunk_duration <= 0:
            raise ValueError("Chunk duration must be positive")

        if overlap_duration < 0 or overlap_duration >= chunk_duration:
            raise ValueError("Overlap duration must be non-negative and less than chunk duration")

        # Apply time range constraints if specified
        effective_start = start_time if start_time is not None else 0.0
        effective_end = end_time if end_time is not None else audio_duration

        # Validate time range
        if effective_start < 0:
            effective_start = 0.0

        if effective_end > audio_duration:
            effective_end = audio_duration

        if effective_start >= effective_end:
            raise ValueError(
                f"Invalid time range: start ({effective_start}) >= end ({effective_end})"
            )

        # Calculate effective duration for chunking
        effective_duration = effective_end - effective_start

        # Handle very short audio files (shorter than chunk duration)
        if effective_duration <= chunk_duration:
            return [(effective_start, effective_end)]

        # Calculate chunk boundaries with overlap
        chunk_boundaries = []
        current_pos = effective_start

        while current_pos < effective_end:
            chunk_end = min(current_pos + chunk_duration, effective_end)
            chunk_boundaries.append((current_pos, chunk_end))

            if chunk_end >= effective_end:
                break

            current_pos = chunk_end - overlap_duration

        return chunk_boundaries

    def create_audio_chunks(
        self,
        audio_path: Path,
        output_dir: Path,
        chunk_boundaries: List[Tuple[float, float]],
        quality_spec: AudioQualitySpecification,
        api_provider: Optional[ApiProvider] = None,
        job_id: Optional[str] = None,
    ) -> List[Path]:
        """Create audio chunks from chunk boundaries."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        chunk_paths = []
        for i, (start, end) in enumerate(chunk_boundaries):
            # Create chunk filename
            chunk_filename = f"chunk_{i:03d}_{start:.2f}_{end:.2f}{audio_path.suffix}"
            chunk_path = output_dir / chunk_filename

            # Extract audio segment
            self.audio_format_service.extract_audio_segment(
                audio_path, chunk_path, start, end, quality_spec
            )

            chunk_paths.append(chunk_path)

        return chunk_paths

    def detect_silence(
        self, audio_path: Path, min_silence_duration: float = 0.5, silence_threshold: float = -40
    ) -> List[Tuple[float, float]]:
        """Detect silence regions in an audio file."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use the audio format service to detect silence
        silence_regions = self.audio_format_service.detect_silence(
            audio_path, min_silence_duration, silence_threshold
        )

        # Convert to list of tuples
        return [(region["start"], region["end"]) for region in silence_regions]


@dataclass
class MockTranscriptSegment:
    """Mock transcript segment for testing."""

    text: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: Optional[int] = None


class MockTranscriber(Transcriber):
    """
    Mock implementation of Transcriber for testing.

    Returns predetermined transcript segments.
    """

    def __init__(self, segments: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize with optional predefined segments.

        Args:
            segments: Optional list of segment dictionaries with text, start, end keys
        """
        self.segments = segments or [
            {"text": "This is a test transcript.", "start": 0.0, "end": 2.0},
            {"text": "It has multiple segments.", "start": 2.5, "end": 4.0},
            {"text": "To test the transcription service.", "start": 4.5, "end": 6.0},
        ]

    def transcribe(self, audio_path: Path, **kwargs) -> List[MockTranscriptSegment]:
        """Transcribe audio file returning mock segments."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Return mock transcript segments
        return [
            MockTranscriptSegment(
                text=segment["text"],
                start=segment["start"],
                end=segment["end"],
                confidence=segment.get("confidence", 1.0),
                speaker=segment.get("speaker"),
            )
            for segment in self.segments
        ]

    def close(self):
        """Clean up resources."""
        pass


@dataclass
class MockDiarizationSegment:
    """Mock diarization segment for testing."""

    speaker: int
    start: float
    end: float
    confidence: float = 1.0


class MockDiarizer(Diarizer):
    """
    Mock implementation of Diarizer for testing.

    Returns predetermined diarization segments.
    """

    def __init__(self, segments: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize with optional predefined segments.

        Args:
            segments: Optional list of segment dictionaries with speaker, start, end keys
        """
        self.segments = segments or [
            {"speaker": 0, "start": 0.0, "end": 2.0},
            {"speaker": 1, "start": 2.5, "end": 4.0},
            {"speaker": 0, "start": 4.5, "end": 6.0},
        ]

    def diarize(self, audio_path: Path, **kwargs) -> List[MockDiarizationSegment]:
        """Diarize audio file returning mock segments."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Return mock diarization segments
        return [
            MockDiarizationSegment(
                speaker=segment["speaker"],
                start=segment["start"],
                end=segment["end"],
                confidence=segment.get("confidence", 1.0),
            )
            for segment in self.segments
        ]

    def close(self):
        """Clean up resources."""
        pass
