"""
Orchestration layer for PyHearingAI.

This module implements the orchestration layer for PyHearingAI, which coordinates
the execution of the various services and manages the workflow for audio processing jobs.
"""

import logging
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pyhearingai.application.audio_chunking import AudioChunkingService
from pyhearingai.application.progress import ProgressTracker, create_progress_callback
from pyhearingai.application.resource_manager import get_resource_manager
from pyhearingai.core.idempotent import AudioChunk, ProcessingJob, ProcessingStatus, SpeakerSegment
from pyhearingai.core.models import Segment, TranscriptionResult
from pyhearingai.diarization.service import DiarizationService
from pyhearingai.infrastructure.registry import (
    get_converter,
    get_diarizer,
    get_speaker_assigner,
    get_transcriber,
)
from pyhearingai.infrastructure.repositories.json_repositories import (
    JsonChunkRepository,
    JsonJobRepository,
)
from pyhearingai.reconciliation.service import BatchConfig, ReconciliationService
from pyhearingai.transcription.service import TranscriptionService

logger = logging.getLogger(__name__)


class Monitoring:
    """Monitoring utility for tracking performance and system health."""

    def __init__(self, enabled: bool = True):
        """
        Initialize the monitoring utility.

        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self.start_time = time.time()
        self.timings = {}
        self.memory_usage = []
        self.errors = []

    def start_task(self, task_name: str) -> None:
        """
        Start timing a task.

        Args:
            task_name: Name of the task to time
        """
        if not self.enabled:
            return

        self.timings[task_name] = {"start": time.time(), "end": None, "duration": None}
        logger.debug(f"Starting task: {task_name}")

    def end_task(self, task_name: str) -> float:
        """
        End timing a task and return the duration.

        Args:
            task_name: Name of the task

        Returns:
            float: Duration of the task in seconds
        """
        if not self.enabled or task_name not in self.timings:
            return 0.0

        end_time = time.time()
        self.timings[task_name]["end"] = end_time
        duration = end_time - self.timings[task_name]["start"]
        self.timings[task_name]["duration"] = duration

        logger.debug(f"Completed task: {task_name} in {duration:.2f}s")
        return duration

    def log_error(self, task_name: str, error: Exception) -> None:
        """
        Log an error that occurred during processing.

        Args:
            task_name: Name of the task where the error occurred
            error: The exception that was raised
        """
        if not self.enabled:
            return

        error_info = {
            "task": task_name,
            "error": str(error),
            "type": type(error).__name__,
            "timestamp": time.time(),
            "traceback": traceback.format_exc(),
        }

        self.errors.append(error_info)
        logger.error(f"Error in {task_name}: {str(error)}")

    def log_memory_usage(self) -> None:
        """Log current memory usage."""
        if not self.enabled:
            return

        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            self.memory_usage.append({"timestamp": time.time(), "memory_mb": memory_mb})

            logger.debug(f"Memory usage: {memory_mb:.2f} MB")
        except ImportError:
            logger.debug("psutil not available, skipping memory usage logging")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of monitoring information.

        Returns:
            Dict[str, Any]: Monitoring summary
        """
        if not self.enabled:
            return {}

        total_duration = time.time() - self.start_time

        # Calculate task statistics
        task_stats = {}
        for task_name, timing in self.timings.items():
            if timing["duration"] is not None:
                task_stats[task_name] = {
                    "duration": timing["duration"],
                    "percentage": (timing["duration"] / total_duration) * 100
                    if total_duration > 0
                    else 0,
                }

        # Get memory statistics if available
        memory_stats = {}
        if self.memory_usage:
            memory_stats = {
                "max": max(item["memory_mb"] for item in self.memory_usage),
                "min": min(item["memory_mb"] for item in self.memory_usage),
                "avg": sum(item["memory_mb"] for item in self.memory_usage)
                / len(self.memory_usage),
                "samples": len(self.memory_usage),
            }

        return {
            "total_duration": total_duration,
            "tasks": task_stats,
            "memory": memory_stats,
            "errors": len(self.errors),
        }


class Task:
    """
    Context manager for task execution.

    This class provides a simple context manager for executing tasks
    with proper logging and error handling.
    """

    def __init__(self, task_id: str, description: str):
        """
        Initialize the task.

        Args:
            task_id: ID of the task
            description: Description of the task
        """
        self.task_id = task_id
        self.description = description

    def __enter__(self):
        """Enter the context manager."""
        logger.info(f"Starting task: {self.description} ({self.task_id})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if exc_type:
            logger.error(f"Task failed: {self.description} ({self.task_id}): {exc_val}")
        else:
            logger.info(f"Completed task: {self.description} ({self.task_id})")
        return False  # Don't suppress exceptions


class WorkflowOrchestrator:
    """
    Orchestrates the end-to-end audio processing workflow.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        chunk_size: float = 10.0,
        overlap: float = 0.0,
        show_chunks: bool = False,
        verbose: bool = False,
        use_responses_api: bool = False,  # Whether to use the Responses API for reconciliation
        use_batched_reconciliation: bool = False,  # Whether to use batched reconciliation
        show_progress_bars: bool = True,  # Whether to show progress bars in the terminal
        **kwargs,
    ):
        """
        Initialize the workflow orchestrator.

        Args:
            max_workers: Maximum number of parallel workers
            chunk_size: Size of audio chunks in seconds
            overlap: Overlap between chunks in seconds
            show_chunks: Whether to show detailed chunk progress
            verbose: Whether to enable verbose output
            use_responses_api: Whether to use the Responses API for reconciliation
            use_batched_reconciliation: Whether to use batched reconciliation
            show_progress_bars: Whether to show progress bars in the terminal
            **kwargs: Additional options
        """
        # Store options
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.show_chunks = show_chunks
        self.verbose = verbose
        self.use_responses_api = use_responses_api
        self.use_batched_reconciliation = use_batched_reconciliation
        self.show_progress_bars = show_progress_bars
        self.kwargs = kwargs

        # Initialize repositories
        self.job_repository = JsonJobRepository()
        self.chunk_repository = JsonChunkRepository()

        # Initialize services with progress bar support
        self.audio_service = AudioChunkingService()
        self.diarization_service = DiarizationService(
            max_workers=max_workers, show_progress=show_progress_bars
        )
        self.transcription_service = TranscriptionService(max_workers=max_workers)
        self.reconciliation_service = ReconciliationService(use_responses_api=use_responses_api)

        # For backward compatibility
        self.diarization_repository = self.diarization_service
        self.transcription_repository = self.transcription_service
        self.result_repository = self.reconciliation_service

        logger.info(
            f"Initialized WorkflowOrchestrator with max_workers={max_workers}, chunk_size={chunk_size}s, show_chunks={show_chunks}, use_responses_api={use_responses_api}, use_batched_reconciliation={use_batched_reconciliation}, show_progress_bars={show_progress_bars}"
        )

    def process_job(
        self, job: ProcessingJob, progress_tracker=None, force: bool = False
    ) -> TranscriptionResult:
        """
        Process a job from start to finish.

        Args:
            job: The job to process
            progress_tracker: Optional progress tracker for monitoring progress
            force: Whether to force processing even if the job is already completed

        Returns:
            TranscriptionResult object containing the transcription
        """
        try:
            # Update job status
            if job.status != ProcessingStatus.IN_PROGRESS:
                job.status = ProcessingStatus.IN_PROGRESS
                self.job_repository.save(job)

            # Update progress tracker if provided
            if progress_tracker:
                progress_tracker.update_job_progress(0.1, "Starting job")

            # Process audio chunks
            if not self.chunk_repository.has_chunks(job.id) or force:
                with Task(f"job_{job.id}", "Creating audio chunks"):
                    self._process_audio_chunks(job)
                    if progress_tracker:
                        progress_tracker.update_job_progress(0.2, "Audio chunks created")

            # Initialize models (if needed)
            self._preinitialize_models()

            # Process diarization
            with Task(f"diarization_{job.id}", "Processing diarization"):
                # Only process diarization if not already done or force is True
                if not self.diarization_service.has_all_chunk_data(job.id) or force:
                    self._process_diarization(job, progress_tracker)
                else:
                    logger.info(f"Using cached diarization results for job {job.id}")
                    if progress_tracker:
                        progress_tracker.update_job_progress(
                            0.7, "Using cached diarization results"
                        )

            # Process transcription
            with Task(f"transcription_{job.id}", "Processing transcription"):
                # Only process transcription if not already done or force is True
                if not self.transcription_service.has_all_chunk_data(job.id) or force:
                    self._process_transcription(job, progress_tracker)
                else:
                    logger.info(f"Using cached transcription results for job {job.id}")
                    if progress_tracker:
                        progress_tracker.update_job_progress(
                            0.9, "Using cached transcription results"
                        )

            # Reconcile results
            with Task(f"reconciliation_{job.id}", "Reconciling results"):
                # Check if we should use batched reconciliation
                use_batched = self._should_use_batched_reconciliation(job)

                if progress_tracker:
                    progress_tracker.update_job_progress(0.9, "Reconciling results")

                if use_batched:
                    logger.info(f"Using batched reconciliation for job {job.id}")
                    # Create batch config with reasonable defaults
                    config = BatchConfig(
                        batch_size_seconds=180,  # 3 minutes per batch
                        batch_overlap_seconds=10,  # 10 seconds overlap
                        max_tokens_per_batch=2500,  # Reduced tokens per batch
                    )
                    reconciled_segments = self.reconciliation_service.reconcile_batched(job, config)
                else:
                    logger.info(f"Using standard reconciliation for job {job.id}")
                    reconciled_segments = self.reconciliation_service.reconcile(job)

                if progress_tracker:
                    progress_tracker.update_job_progress(0.95, "Results reconciled")

            # Create result
            with Task(f"create_result_{job.id}", "Creating result"):
                result = self._create_result(job, reconciled_segments)

                if progress_tracker:
                    progress_tracker.update_job_progress(1.0, "Processing complete")

            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self.job_repository.save(job)

            return result
        except Exception as e:
            # Log the error and update the job status
            logger.error(f"Error in job_{job.id}: {str(e)}")
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            self.job_repository.save(job)

            # Update progress tracker if provided
            if progress_tracker:
                progress_tracker.error(f"Error: {str(e)}")

            raise

    def _process_audio_chunks(self, job: ProcessingJob) -> None:
        """
        Create audio chunks for a job and save them to the repository.

        Args:
            job: The processing job
        """
        chunks = self._create_chunks(job)

        # Save each chunk to the repository
        for chunk in chunks:
            self.chunk_repository.save(chunk)

        logger.info(f"Created and saved {len(chunks)} chunks for job {job.id}")

    def _preinitialize_models(self) -> None:
        """Initialize models if needed."""
        # This would load models into memory if needed
        pass

    def _process_diarization(self, job: ProcessingJob, progress_tracker=None) -> None:
        """
        Process diarization for all chunks in a job.

        Args:
            job: The processing job
            progress_tracker: Optional progress tracker
        """
        # Get all chunks for the job
        chunks = self.chunk_repository.get_by_job_id(job.id)
        if not chunks:
            logger.warning(f"No chunks found for job {job.id}")
            # Try again with created chunks
            chunks = self._create_chunks(job)
            if not chunks:
                logger.error(f"Failed to create chunks for job {job.id}")
                return

        # Create progress callback function for diarization
        def diarization_progress_callback(completed, total, message):
            """Handle progress updates from the diarization service"""
            if progress_tracker:
                # Convert to percentage for the overall progress tracker
                completion_percent = (completed / total) if total > 0 else 0
                # Diarization is about 50% of the overall process
                overall_progress = 0.2 + (completion_percent * 0.5)  # 20-70% of overall progress
                progress_tracker.update_job_progress(overall_progress, f"Diarization: {message}")

            # Always log critical progress points
            if completed == total:
                logger.info(f"Diarization complete: {message}")
            elif completed > 0 and completed % 5 == 0:  # Log every 5 chunks
                logger.info(
                    f"Diarization progress: {completed}/{total} chunks ({(completed/total)*100:.1f}%)"
                )

            # Display in terminal using logger
            if self.verbose:
                logger.debug(f"Diarization: {completed}/{total} - {message}")

        # Process diarization for all chunks
        logger.info(f"Processing diarization for {len(chunks)} chunks in job {job.id}")
        results = self.diarization_service.process_job(
            job=job,
            chunks=chunks,
            progress_callback=diarization_progress_callback if self.show_progress_bars else None,
            show_progress=self.show_chunks,
        )

        # Check results
        if not results:
            logger.warning(f"No diarization results were generated for job {job.id}")
        else:
            logger.info(f"Completed diarization for {len(results)} chunks in job {job.id}")

    def _process_transcription(self, job: ProcessingJob, progress_tracker=None) -> None:
        """
        Process transcription for all chunks in a job.

        Args:
            job: The processing job
            progress_tracker: Optional progress tracker
        """
        # Get all chunks for the job
        chunks = self.chunk_repository.get_by_job_id(job.id)
        if not chunks:
            logger.warning(f"No chunks found for job {job.id}")
            # Try again with created chunks
            chunks = self._create_chunks(job)
            if not chunks:
                logger.error(f"Failed to create chunks for job {job.id}")
                return

        # Create progress callback function for transcription
        def transcription_progress_callback(completed, total, message):
            """Handle progress updates from the transcription service"""
            if progress_tracker:
                # Convert to percentage for the overall progress tracker
                completion_percent = (completed / total) if total > 0 else 0
                # Transcription is about 20% of the overall process (comes after diarization)
                overall_progress = 0.7 + (completion_percent * 0.2)  # 70-90% of overall progress
                progress_tracker.update_job_progress(overall_progress, f"Transcription: {message}")

            # Log progress
            if self.verbose:
                logger.debug(f"Transcription: {completed}/{total} - {message}")

        # Process transcription for all chunks
        logger.info(f"Processing transcription for {len(chunks)} chunks in job {job.id}")
        results = self.transcription_service.process_job(
            job=job,
            chunks=chunks,
            progress_callback=transcription_progress_callback if self.show_progress_bars else None,
            show_progress=self.show_chunks,
        )

        # Check results
        if not results:
            logger.warning(f"No transcription results were generated for job {job.id}")
        else:
            logger.info(f"Completed transcription for {len(results)} chunks in job {job.id}")

    def _create_result(self, job: ProcessingJob, segments: List[Segment]) -> TranscriptionResult:
        """
        Create a TranscriptionResult object from reconciled segments.

        Args:
            job: The processing job
            segments: List of reconciled segments

        Returns:
            TranscriptionResult object
        """
        # Create metadata with job information
        metadata = {
            "job_id": job.id,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_time": job.start_time,
            "end_time": job.end_time,
            "chunk_duration": job.chunk_duration,
            "overlap_duration": job.overlap_duration,
        }

        # Create the TranscriptionResult object
        result = TranscriptionResult(
            segments=segments, audio_path=Path(job.original_audio_path), metadata=metadata
        )

        return result

    def _should_use_batched_reconciliation(self, job: ProcessingJob) -> bool:
        # Reconcile the results
        # Add debug logging to check the condition values
        logger.info(
            f"Auto-batching check - use_batched_reconciliation: {self.use_batched_reconciliation}"
        )
        logger.info(
            f"Auto-batching check - job.start_time: {job.start_time}, job.end_time: {job.end_time}"
        )
        if job.end_time is not None and job.start_time is not None:
            job_duration = job.end_time - job.start_time
            logger.info(f"Auto-batching check - duration: {job_duration}s (threshold: 300s)")

        use_batched = self.use_batched_reconciliation or (
            job.end_time is not None
            and job.start_time is not None
            and (job.end_time - job.start_time) >= 300  # Changed from > to >=
        )

        logger.info(f"Auto-batching decision: {use_batched}")
        return use_batched

    def create_or_resume_job(
        self,
        audio_path: Path,
        job_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        **kwargs,
    ) -> ProcessingJob:
        """
        Create a new job or resume an existing one.

        Args:
            audio_path: Path to the audio file
            job_id: ID of the job to resume (if any)
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            **kwargs: Additional options

        Returns:
            A ProcessingJob object
        """
        # If job_id is provided, resume the job
        if job_id:
            job = self.job_repository.get_by_id(job_id)
            if not job:
                raise ValueError(f"Job with ID {job_id} not found")

            logger.info(f"Resuming job {job_id} for audio file: {job.original_audio_path}")
            return job

        # Create a new job
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            id=job_id,
            original_audio_path=str(audio_path.resolve()),
            chunk_duration=self.chunk_size,
            overlap_duration=self.overlap,
            start_time=start_time,
            end_time=end_time,
        )

        # Save the job
        self.job_repository.save(job)

        logger.info(f"Created new job {job_id} for audio file: {audio_path}")
        return job

    def _create_chunks(self, job: ProcessingJob) -> List[AudioChunk]:
        """
        Create audio chunks for a job.

        Args:
            job: The job to create chunks for

        Returns:
            List of created audio chunks
        """
        logger.info(f"Creating chunks for job {job.id}")
        chunks = self.audio_service.create_chunks(job)
        logger.info(f"Created {len(chunks)} chunks for job {job.id}")
        return chunks


# Import required for the uuid module used in create_or_resume_job
import uuid
