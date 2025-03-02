"""
Main application service for transcribing audio with speaker diarization.

This module implements the primary use case of the application: transcribing an audio file
with speaker diarization and returning the result.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pyhearingai.core.models import TranscriptionResult
from pyhearingai.infrastructure.registry import (
    get_converter,
    get_diarizer,
    get_speaker_assigner,
    get_transcriber,
)

logger = logging.getLogger(__name__)


def transcribe(
    audio_path: Union[str, Path],
    transcriber: str = "whisper_openai",
    diarizer: str = "pyannote",
    output_format: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    verbose: bool = False,
    **kwargs,
) -> TranscriptionResult:
    """
    Transcribe an audio file with speaker diarization.

    Args:
        audio_path: Path to the audio file to transcribe
        transcriber: Name of the transcriber to use
        diarizer: Name of the diarizer to use
        output_format: Format to save the output (optional)
        progress_callback: Callback function for progress updates
        verbose: Whether to enable verbose logging
        **kwargs: Additional options passed to the transcriber and diarizer

    Returns:
        TranscriptionResult object containing the transcribed segments with speaker information
    """
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Convert audio path to Path object
    if isinstance(audio_path, str):
        audio_path = Path(audio_path)

    # Ensure the audio file exists
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Report progress
    if progress_callback:
        progress_callback(0.0, "Starting transcription process")

    # 1. Convert audio if needed
    logger.debug(f"Converting audio: {audio_path}")
    converter = get_converter()
    converted_path = converter.convert(audio_path, **kwargs)

    if progress_callback:
        progress_callback(0.1, "Audio conversion complete")

    # 2. Transcribe audio
    logger.debug(f"Transcribing with {transcriber}")
    transcriber_instance = get_transcriber(transcriber)
    transcript_segments = transcriber_instance.transcribe(converted_path, **kwargs)

    if progress_callback:
        progress_callback(0.5, "Transcription complete")

    # 3. Diarize audio (identify speakers)
    logger.debug(f"Diarizing with {diarizer}")
    diarizer_instance = get_diarizer(diarizer)
    diarization_segments = diarizer_instance.diarize(converted_path, **kwargs)

    if progress_callback:
        progress_callback(0.8, "Diarization complete")

    # 4. Assign speakers to transcript segments
    logger.debug("Assigning speakers to transcript segments")
    assigner = get_speaker_assigner()
    merged_segments = assigner.assign_speakers(transcript_segments, diarization_segments, **kwargs)

    # 5. Create and return result
    result = TranscriptionResult(
        segments=merged_segments,
        audio_path=audio_path,
        metadata={
            "transcriber": transcriber,
            "diarizer": diarizer,
            "duration": merged_segments[-1].end if merged_segments else 0,
            "options": kwargs,
        },
    )

    if progress_callback:
        progress_callback(1.0, "Processing complete")

    # If output format is specified, save the result
    if output_format:
        from pyhearingai.application.outputs import save_transcript

        save_transcript(result, audio_path.with_suffix(f".{output_format}"), output_format)

    return result
