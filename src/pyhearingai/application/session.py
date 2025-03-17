"""
Session management for PyHearingAI.

This module implements a session manager that allows reusing resources across
multiple transcription jobs, improving performance for batch processing.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pyhearingai.application.orchestrator import WorkflowOrchestrator
from pyhearingai.application.progress import ProgressTracker
from pyhearingai.core.models import TranscriptionResult

logger = logging.getLogger(__name__)


class Session:
    """
    A session for processing multiple audio files with shared resources.

    The Session class maintains loaded models and configurations between
    multiple transcription jobs, improving performance by avoiding
    repeated initialization of resources.
    """

    def __init__(
        self,
        transcriber: Union[str, Dict[str, Any]] = "default",
        diarizer: Union[str, Dict[str, Any]] = "default",
        speaker_assigner: Union[str, Dict[str, Any]] = "default",
        chunk_size_seconds: float = 10.0,
        overlap_seconds: float = 1.5,
        max_workers: Optional[int] = None,
        show_chunks: bool = False,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
        **options,
    ):
        """
        Initialize a session with the specified resources.

        Args:
            transcriber: Transcription model name or config dict
            diarizer: Diarization model name or config dict
            speaker_assigner: Speaker assignment model name or config
            chunk_size_seconds: Size of audio chunks in seconds
            overlap_seconds: Overlap between chunks in seconds
            max_workers: Maximum number of worker threads
            show_chunks: Whether to show chunk progress
            cache_dir: Directory to use for caching
            verbose: Whether to enable verbose logging
            options: Additional options for the pipeline
        """
        logger.info("Initializing PyHearingAI session")

        # Store configuration
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.speaker_assigner = speaker_assigner
        self.chunk_size_seconds = chunk_size_seconds
        self.overlap_seconds = overlap_seconds
        self.verbose = verbose
        self.options = options

        # Create orchestrator with persistent resources
        self.orchestrator = WorkflowOrchestrator(
            transcriber_name=transcriber,
            diarizer_name=diarizer,
            speaker_assigner_name=speaker_assigner,
            max_workers=max_workers,
            show_chunks=show_chunks,
            chunk_size=chunk_size_seconds,
            cache_dir=cache_dir,
            verbose=verbose,
            **options,
        )

        logger.debug("Session initialized with resources")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        format: str = "txt",
        chunk_size_seconds: Optional[float] = None,
        overlap_seconds: Optional[float] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using the session's resources.

        Args:
            audio_path: Path to the audio file
            output_path: Path for the output file
            format: Output format
            chunk_size_seconds: Override the session's chunk size
            overlap_seconds: Override the session's overlap size
            kwargs: Additional options for this specific transcription

        Returns:
            A TranscriptionResult object
        """
        # Convert paths to Path objects
        audio_path = Path(audio_path) if isinstance(audio_path, str) else audio_path
        if output_path:
            output_path = Path(output_path) if isinstance(output_path, str) else output_path

        logger.info(f"Transcribing audio file: {audio_path}")

        # Use session defaults if not specified
        chunk_size = chunk_size_seconds or self.chunk_size_seconds
        overlap = overlap_seconds or self.overlap_seconds

        # Create or resume the job
        job = self.orchestrator.create_or_resume_job(
            audio_path=audio_path, chunk_duration=chunk_size, overlap_duration=overlap, **kwargs
        )

        # Get chunks from the repository
        chunks = self.orchestrator.chunk_repository.get_by_job_id(job.id)

        # Create a progress tracker
        progress_tracker = ProgressTracker(
            job=job, chunks=chunks, show_chunks=self.orchestrator.show_chunks
        )

        # Process the job
        result = self.orchestrator.process_job(job=job, progress_tracker=progress_tracker)

        # Save the result if an output path is provided
        if output_path:
            result.save(output_path, format=format)
            logger.info(f"Saved result to {output_path}")

        return result

    def close(self):
        """
        Close the session and release all resources.

        This method should be called when the session is no longer needed
        to ensure all resources are properly released.
        """
        logger.info("Closing PyHearingAI session")

        # Close the orchestrator's services
        if (
            hasattr(self.orchestrator, "diarization_service")
            and self.orchestrator.diarization_service
        ):
            self.orchestrator.diarization_service.close()

        if (
            hasattr(self.orchestrator, "transcription_service")
            and self.orchestrator.transcription_service
        ):
            self.orchestrator.transcription_service.close()

        if (
            hasattr(self.orchestrator, "reconciliation_service")
            and self.orchestrator.reconciliation_service
        ):
            self.orchestrator.reconciliation_service.close()

        logger.debug("Session resources released")


@contextmanager
def pipeline_session(
    transcriber: Union[str, Dict[str, Any]] = "default",
    diarizer: Union[str, Dict[str, Any]] = "default",
    speaker_assigner: Union[str, Dict[str, Any]] = "default",
    chunk_size_seconds: float = 10.0,
    overlap_seconds: float = 1.5,
    max_workers: Optional[int] = None,
    show_chunks: bool = False,
    cache_dir: Optional[str] = None,
    verbose: bool = False,
    **options,
):
    """
    Create a reusable pipeline session for multiple transcriptions.

    This context manager allows efficient processing of multiple audio files
    by reusing loaded models and resources across transcriptions.

    Args:
        transcriber: Transcription model name or config dict
        diarizer: Diarization model name or config dict
        speaker_assigner: Speaker assignment model name or config
        chunk_size_seconds: Size of audio chunks in seconds
        overlap_seconds: Overlap between chunks in seconds
        max_workers: Maximum number of worker threads
        show_chunks: Whether to show chunk progress
        cache_dir: Directory to use for caching
        verbose: Whether to enable verbose logging
        options: Additional options for the pipeline

    Yields:
        A Session object with a transcribe method

    Example:
        ```python
        from pyhearingai import pipeline_session

        # Process multiple files with resource reuse
        with pipeline_session(verbose=True) as session:
            result1 = session.transcribe("file1.mp3")
            result2 = session.transcribe("file2.mp3")
            # Resources are efficiently managed
        ```
    """
    # Create the session
    session = Session(
        transcriber=transcriber,
        diarizer=diarizer,
        speaker_assigner=speaker_assigner,
        chunk_size_seconds=chunk_size_seconds,
        overlap_seconds=overlap_seconds,
        max_workers=max_workers,
        show_chunks=show_chunks,
        cache_dir=cache_dir,
        verbose=verbose,
        **options,
    )

    try:
        # Yield the session to the caller
        yield session
    finally:
        # Ensure resources are released
        session.close()
