"""
Pyannote diarizer implementation.

This module provides a concrete implementation of the Diarizer port
using the Pyannote.audio library for speaker diarization.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import the real Pyannote pipeline
try:
    import torch  # Added torch import for GPU detection
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

from pyhearingai.core.models import DiarizationSegment
from pyhearingai.core.ports import Diarizer
from pyhearingai.infrastructure.registry import register_diarizer

logger = logging.getLogger(__name__)


@register_diarizer("pyannote")
class PyannoteDiarizer(Diarizer):
    """Diarizer implementation using the Pyannote.audio library."""

    def __init__(self):
        """Initialize the Pyannote diarizer."""
        self._api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self._api_key:
            logger.warning(
                "HUGGINGFACE_API_KEY environment variable not set. "
                "You will need to provide the API key when calling diarize()."
            )

        # Lazily initialized pipeline
        self._pipeline = None

        if not PYANNOTE_AVAILABLE:
            logger.warning(
                "Pyannote is not installed. You need to install it with: "
                "pip install pyannote.audio"
            )

    @property
    def name(self) -> str:
        """Get the name of the diarizer."""
        return "pyannote"

    def _get_pipeline(self, api_key: str = None):
        """
        Get or initialize the Pyannote pipeline.

        Args:
            api_key: Hugging Face API key (overrides environment variable)

        Returns:
            Pyannote pipeline instance
        """
        if self._pipeline is None:
            api_key = api_key or self._api_key
            if not api_key:
                raise ValueError(
                    "Hugging Face API key not provided. "
                    "Set the HUGGINGFACE_API_KEY environment variable or provide it as api_key parameter."
                )

            if not PYANNOTE_AVAILABLE:
                raise ImportError(
                    "Pyannote is not installed. Install it with: pip install pyannote.audio"
                )

            logger.debug("Initializing Pyannote pipeline")
            try:
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",  # Match original version 3.1
                    use_auth_token=api_key,
                )
            except Exception as auth_error:
                error_message = str(auth_error)
                if "401 Client Error" in error_message:
                    raise Exception(
                        "Authentication failed. Please:\n"
                        "1. Verify your HUGGINGFACE_API_KEY is correct\n"
                        "2. Accept the user agreement at: https://hf.co/pyannote/speaker-diarization-3.1\n"
                        "3. Accept the user agreement at: https://hf.co/pyannote/segmentation-3.1"
                    )
                raise

        return self._pipeline

    def diarize(self, audio_path: Path, **kwargs) -> List[DiarizationSegment]:
        """
        Perform speaker diarization on an audio file using Pyannote.

        Args:
            audio_path: Path to the audio file to diarize
            **kwargs: Additional diarization options
                - api_key: Hugging Face API key (overrides environment variable)
                - output_dir: Directory to save output files (default: "content/diarization")

        Returns:
            List of segments with speaker identification and timing information
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Extract options
        api_key = kwargs.get("api_key", self._api_key)
        output_dir = kwargs.get("output_dir", "content/diarization")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        try:
            # If Pyannote is not available, fall back to mock data
            if not PYANNOTE_AVAILABLE:
                logger.warning("Pyannote not available, using mock data instead")
                return self._mock_diarize(audio_path, output_dir=output_dir)

            # Use the real Pyannote pipeline
            logger.debug(f"Diarizing {audio_path} with Pyannote")
            pipeline = self._get_pipeline(api_key)

            # Use GPU if available (matching original implementation)
            device_info = "CPU"
            if PYANNOTE_AVAILABLE and hasattr(torch, "cuda") and torch.cuda.is_available():
                try:
                    pipeline = pipeline.to(torch.device("cuda"))
                    device_info = "GPU (CUDA)"
                    logger.info(f"Using GPU acceleration for diarization")
                except Exception as e:
                    logger.warning(f"Failed to use GPU: {e}")

            # Log pipeline initialization
            with open(os.path.join(output_dir, "pipeline_info.txt"), "w") as f:
                f.write(f"Pipeline initialized at: {datetime.now().isoformat()}\n")
                f.write(f"Model: pyannote/speaker-diarization-3.1\n")
                f.write(f"Input file: {audio_path}\n")
                f.write(f"Processing device: {device_info}\n")

            logger.info(f"Processing audio on {device_info}")

            # Use ProgressHook as in original
            with ProgressHook() as hook:
                diarization = pipeline(str(audio_path), min_speakers=1, max_speakers=5, hook=hook)

            # Save RTTM file (as in original)
            rttm_path = os.path.join(output_dir, "diarization.rttm")
            with open(rttm_path, "w") as rttm:
                diarization.write_rttm(rttm)

            # Convert diarization to DiarizationSegment objects and raw dict format
            segments = []
            raw_segments = []

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Normalize speaker format:
                # 1. Strip any existing "SPEAKER_" prefix to avoid double prefixing
                # 2. Ensure numeric format (01, 02, etc.) with leading zeros

                # Extract the actual speaker identifier (number or string)
                # Handle various cases like "SPEAKER_01", "01", "1", etc.
                if isinstance(speaker, str) and speaker.startswith("SPEAKER_"):
                    speaker_num = speaker[8:]  # Remove "SPEAKER_" prefix
                else:
                    speaker_num = speaker

                # Remove any additional 'SPEAKER' text that might be present
                if isinstance(speaker_num, str) and "SPEAKER" in speaker_num:
                    # Extract just the numeric or identifier part
                    # Try to extract numbers from the string
                    numbers = re.findall(r"\d+", speaker_num)
                    if numbers:
                        speaker_num = numbers[0]  # Use the first number found

                # Then format with leading zeros for single digits
                # Try to convert to int first to handle both numeric and string inputs
                try:
                    speaker_num = int(speaker_num)
                    speaker_id = f"SPEAKER_{speaker_num:02d}"  # Format with leading zero
                except (ValueError, TypeError):
                    # If conversion fails, use as is with prefix
                    speaker_id = f"SPEAKER_{speaker_num}"

                # Create raw segment dict (as in original)
                raw_segment = {"start": turn.start, "end": turn.end, "speaker": speaker_id}
                raw_segments.append(raw_segment)

                # Create a DiarizationSegment (for clean architecture)
                diarization_segment = DiarizationSegment(
                    speaker_id=speaker_id, start=turn.start, end=turn.end, score=1.0
                )
                segments.append(diarization_segment)

            # Save segments as JSON (as in original)
            segments_path = os.path.join(output_dir, "segments.json")
            with open(segments_path, "w") as f:
                json.dump(raw_segments, f, indent=2)

            # Create summary file (as in original)
            summary_path = os.path.join(output_dir, "diarization_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"Diarization completed at: {datetime.now().isoformat()}\n")
                f.write(f"Input file: {audio_path}\n")
                f.write(f"Number of segments: {len(raw_segments)}\n")
                f.write(f"Processing device: {device_info}\n")

                # Calculate some statistics
                total_duration = sum(seg["end"] - seg["start"] for seg in raw_segments)
                unique_speakers = len(set(seg["speaker"] for seg in raw_segments))

                f.write(f"Total audio duration processed: {total_duration:.2f} seconds\n")
                f.write(f"Number of unique speakers detected: {unique_speakers}\n")

            logger.debug(f"Found {len(segments)} diarization segments")
            return segments

        except Exception as e:
            # Log error and create error file (as in original)
            error_log_path = os.path.join(output_dir, "diarization_error.log")
            with open(error_log_path, "w") as f:
                f.write(f"Error in diarization:\n{str(e)}")

            logger.error(f"Error diarizing with Pyannote: {str(e)}")
            raise Exception(f"Diarization error: {str(e)}")

    def _mock_diarize(
        self, audio_path: Path, output_dir: str = "content/diarization"
    ) -> List[DiarizationSegment]:
        """Fallback mock implementation for testing without Pyannote."""
        logger.warning("Using mock diarization data - ONLY FOR TESTING!")

        # Generate mock segments based on file
        if "example_audio" in str(audio_path):
            # Original mock segments for the example audio
            mock_segments = [
                ("0", 0.0, 2.5),
                ("1", 2.7, 5.2),
                ("0", 5.4, 8.1),
                ("1", 8.3, 10.0),
                ("0", 10.2, 12.8),
            ]
        else:
            # Extended mock segments for other audio files
            mock_segments = [
                ("0", 0.0, 5.0),
                ("1", 5.0, 10.0),
                ("0", 10.0, 15.0),
                ("1", 15.0, 20.0),
                ("0", 20.0, 25.0),
                ("1", 25.0, 30.0),
                ("0", 30.0, 35.0),
                ("1", 35.0, 40.0),
            ]

        # Convert to both DiarizationSegment objects and raw dict segments
        segments = []
        raw_segments = []

        for speaker_num, start, end in mock_segments:
            # Extract the actual speaker identifier (number or string)
            # Handle various cases like "SPEAKER_01", "01", "1", etc.
            if isinstance(speaker_num, str) and speaker_num.startswith("SPEAKER_"):
                speaker_num = speaker_num[8:]  # Remove "SPEAKER_" prefix

            # Remove any additional 'SPEAKER' text that might be present
            if isinstance(speaker_num, str) and "SPEAKER" in speaker_num:
                # Try to extract numbers from the string
                numbers = re.findall(r"\d+", speaker_num)
                if numbers:
                    speaker_num = numbers[0]  # Use the first number found

            # Format with leading zeros for single digits
            try:
                speaker_num = int(speaker_num)
                speaker_id = f"SPEAKER_{speaker_num:02d}"  # Format with leading zero
            except (ValueError, TypeError):
                speaker_id = f"SPEAKER_{speaker_num}"

            # Create raw segment dict
            raw_segment = {"start": start, "end": end, "speaker": speaker_id}
            raw_segments.append(raw_segment)

            # Create DiarizationSegment
            segment = DiarizationSegment(speaker_id=speaker_id, start=start, end=end, score=1.0)
            segments.append(segment)

        # Save mock outputs to match real output format
        # Save segments as JSON
        os.makedirs(output_dir, exist_ok=True)
        segments_path = os.path.join(output_dir, "segments.json")
        with open(segments_path, "w") as f:
            json.dump(raw_segments, f, indent=2)

        # Create summary file
        summary_path = os.path.join(output_dir, "diarization_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Mock diarization completed at: {datetime.now().isoformat()}\n")
            f.write(f"Input file: {audio_path}\n")
            f.write(f"Number of segments: {len(raw_segments)}\n")
            f.write(f"Processing device: CPU (mock)\n")

            # Calculate some statistics
            total_duration = sum(seg["end"] - seg["start"] for seg in raw_segments)
            unique_speakers = len(set(seg["speaker"] for seg in raw_segments))

            f.write(f"Total audio duration processed: {total_duration:.2f} seconds\n")
            f.write(f"Number of unique speakers detected: {unique_speakers}\n")
            f.write("NOTE: THIS IS MOCK DATA - NOT REAL DIARIZATION\n")

        logger.debug(f"Created {len(segments)} mock diarization segments")
        return segments
