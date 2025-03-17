"""
Whisper adapter for chunk-aware transcription.

This module provides an adapter for the Whisper transcriber that supports
transcribing individual audio chunks and speaker segments.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from pyhearingai.core.idempotent import AudioChunk, SpeakerSegment
from pyhearingai.core.models import Segment
from pyhearingai.core.ports import Transcriber
from pyhearingai.infrastructure.registry import get_transcriber

logger = logging.getLogger(__name__)


class WhisperAdapter:
    """
    Adapter for the Whisper transcriber to support chunk-aware transcription.

    This adapter wraps the original WhisperOpenAITranscriber and adds support
    for transcribing individual audio chunks and speaker segments.
    """

    def __init__(self, transcriber_name: str = "whisper_openai"):
        """
        Initialize the Whisper adapter.

        Args:
            transcriber_name: Name of the transcriber to use (default: "whisper_openai")
        """
        self.transcriber_name = transcriber_name
        self._transcriber = None  # Lazy-loaded when needed
        logger.debug(f"Initialized WhisperAdapter with transcriber: {transcriber_name}")

    @property
    def transcriber(self) -> Optional[Transcriber]:
        """Lazy-load the transcriber when needed"""
        if self._transcriber is None:
            logger.debug(f"Initializing transcriber: {self.transcriber_name}")
            self._transcriber = get_transcriber(self.transcriber_name)
        return self._transcriber

    def transcribe_chunk(self, chunk: AudioChunk, **kwargs) -> List[Segment]:
        """
        Transcribe an entire audio chunk.

        Args:
            chunk: Audio chunk to transcribe
            **kwargs: Additional arguments for the transcriber

        Returns:
            List of transcription segments with timing information
        """
        if not chunk or not chunk.chunk_path or not Path(chunk.chunk_path).exists():
            logger.warning(f"Cannot transcribe chunk {chunk.id}: Invalid path or file not found")
            return []

        transcriber = self.transcriber
        if not transcriber:
            logger.error(f"Failed to initialize transcriber '{self.transcriber_name}'")
            return []

        try:
            logger.debug(f"Transcribing chunk {chunk.id} with path {chunk.chunk_path}")

            # Add prompt if specified in kwargs
            if not kwargs.get("prompt") and chunk.metadata.get("language"):
                # Use language as a prompt hint if available
                logger.debug(f"Adding language hint to prompt: {chunk.metadata['language']}")
                kwargs["language"] = chunk.metadata["language"]

            # Transcribe the chunk
            segments = transcriber.transcribe(Path(chunk.chunk_path), **kwargs)

            # Adjust segment times based on chunk start time
            adjusted_segments = []
            for segment in segments:
                # Create a new segment with adjusted times
                adjusted_segment = Segment(
                    text=segment.text,
                    start=segment.start + chunk.start_time,
                    end=segment.end + chunk.start_time,
                    speaker_id=segment.speaker_id,
                )
                adjusted_segments.append(adjusted_segment)

            logger.debug(f"Transcribed chunk {chunk.id} - found {len(segments)} segments")
            return adjusted_segments

        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk.id}: {str(e)}")
            return []

    def transcribe_segment(self, segment: SpeakerSegment, audio_path: Path, **kwargs) -> str:
        """
        Transcribe a specific speaker segment.

        Args:
            segment: Speaker segment to transcribe
            audio_path: Path to the audio file for this segment
            **kwargs: Additional arguments for the transcriber

        Returns:
            Transcribed text for the segment
        """
        if not segment or not audio_path or not audio_path.exists():
            logger.warning(
                f"Cannot transcribe segment {segment.id}: Invalid path or file not found"
            )
            return ""

        transcriber = self.transcriber
        if not transcriber:
            logger.error(f"Failed to initialize transcriber '{self.transcriber_name}'")
            return ""

        try:
            logger.debug(f"Transcribing segment {segment.id} with path {audio_path}")

            # Add speaker hint to prompt if available
            if not kwargs.get("prompt") and segment.speaker_id:
                # Use speaker ID as hint in prompt
                kwargs["prompt"] = f"Speaker {segment.speaker_id}"

            # Add language hint if available
            if not kwargs.get("language") and segment.language:
                logger.debug(f"Adding language hint: {segment.language}")
                kwargs["language"] = segment.language

            # Transcribe the segment
            result_segments = transcriber.transcribe(audio_path, **kwargs)

            # Combine all segments into a single text
            text = " ".join(segment.text for segment in result_segments)

            logger.debug(f"Transcribed segment {segment.id} - text length: {len(text)}")
            return text

        except Exception as e:
            logger.error(f"Error transcribing segment {segment.id}: {str(e)}")
            return ""

    def close(self):
        """Clean up resources"""
        # Nothing to do here, but included for consistency with other adapters
        pass
