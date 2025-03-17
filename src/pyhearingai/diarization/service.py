"""
Diarization service for processing audio chunks.

This module provides a service that performs speaker diarization on audio chunks,
with support for idempotent processing and resumability.
"""

import concurrent.futures
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

try:
    # Rich provides better terminal support for progress bars
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        ProgressColumn,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    import tqdm

from pyhearingai.core.idempotent import AudioChunk, ChunkStatus, ProcessingJob
from pyhearingai.core.models import DiarizationSegment
from pyhearingai.core.ports import Diarizer
from pyhearingai.diarization.repositories.diarization_repository import DiarizationRepository
from pyhearingai.infrastructure.diarizers.pyannote import DiarizationTimeoutError
from pyhearingai.infrastructure.registry import get_diarizer

logger = logging.getLogger(__name__)


class DiarizationMonitor:
    """Monitor diarization progress and handle timeouts."""

    def __init__(self, timeout: int = 7200):
        """
        Initialize the monitor.

        Args:
            timeout: Default timeout in seconds (default: 2 hours)
        """
        self.timeout = timeout
        self.start_time = None
        self.current_chunk = None
        self._reset()

    def _reset(self):
        """Reset the monitor state."""
        self.start_time = None
        self.current_chunk = None

    def start_chunk(self, chunk_id: str):
        """Start monitoring a new chunk."""
        self.start_time = datetime.now()
        self.current_chunk = chunk_id
        logger.info(f"Started processing chunk {chunk_id} at {self.start_time}")

    def check_timeout(self) -> bool:
        """Check if current processing has exceeded timeout."""
        if not self.start_time or not self.current_chunk:
            return False

        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > self.timeout:
            logger.warning(f"Chunk {self.current_chunk} processing timed out after {elapsed:.2f}s")
            return True
        return False

    def end_chunk(self):
        """End monitoring for the current chunk."""
        if self.start_time and self.current_chunk:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Completed chunk {self.current_chunk} in {elapsed:.2f}s")
        self._reset()


def _process_chunk_directly(chunk, diarizer_name, timeout=7200, **kwargs):
    """
    Process a chunk directly without a service instance.

    This function is used for parallel processing to avoid issues with
    pickling the service instance.

    Args:
        chunk: Audio chunk to process
        diarizer_name: Name of the diarizer to use
        timeout: Timeout in seconds for diarization
        **kwargs: Additional arguments for the diarizer

    Returns:
        List of diarization segments
    """
    try:
        # Create a new service instance
        from pyhearingai.diarization.repositories.diarization_repository import (
            DiarizationRepository,
        )

        service = DiarizationService(diarizer_name=diarizer_name)

        # Add timeout to kwargs
        kwargs["timeout"] = timeout
        kwargs["disable_progress"] = True

        # Process the chunk
        segments = service.diarize_chunk(chunk, None, **kwargs)

        # Close the service
        service.close()

        return segments
    except Exception as e:
        from traceback import format_exc

        logger.error(f"Error processing chunk directly: {str(e)}")
        logger.error(format_exc())
        return []


class DiarizationService:
    """
    Service for performing speaker diarization on audio chunks.

    This service provides methods for diarizing individual audio chunks
    or entire processing jobs with multiple chunks.
    """

    def __init__(
        self,
        diarizer_name: str = "pyannote",
        repository: Optional[DiarizationRepository] = None,
        max_workers: Optional[int] = None,
        default_timeout: int = 7200,
        show_progress: bool = True,
    ):
        """
        Initialize the diarization service.

        Args:
            diarizer_name: Name of the diarizer to use (default: "pyannote")
            repository: Optional repository for storing diarization results
            max_workers: Maximum number of worker threads for parallel processing
            default_timeout: Default timeout in seconds for diarization (default: 2 hours)
            show_progress: Whether to display progress bars and detailed progress info
        """
        self.diarizer_name = diarizer_name
        self._diarizer = None
        self.default_timeout = default_timeout
        self.monitor = DiarizationMonitor(timeout=default_timeout)
        self.show_progress = show_progress

        # Initialize repository if not provided
        if repository is None:
            logger.debug("No repository provided, creating a default one")
            repository = DiarizationRepository()
        self.repository = repository

        # Set max workers with optimized defaults for different hardware
        if max_workers is None:
            import multiprocessing
            import platform
            import subprocess

            cpu_count = multiprocessing.cpu_count()

            # Detect Apple Silicon (M-series) processors for optimal configuration
            is_apple_silicon = platform.system() == "Darwin" and "arm" in platform.machine()

            # Better detection for M3 Max using system_profiler
            is_high_end = False
            if is_apple_silicon:
                try:
                    # Try to detect M3 Max using system_profiler
                    result = subprocess.run(
                        ["system_profiler", "SPHardwareDataType"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    chip_info = result.stdout.lower()
                    is_high_end = "m3 max" in chip_info or "m3max" in chip_info or cpu_count >= 12
                    if "m3 max" in chip_info or "m3max" in chip_info:
                        logger.info("Detected M3 Max processor")
                except Exception as e:
                    logger.debug(f"Error detecting processor type: {e}")

            if is_high_end:
                # For M3 Max, use more workers to utilize all cores efficiently
                # Keep 2 cores free for system and UI responsiveness
                max_workers = max(1, cpu_count - 2)
                logger.info(
                    f"Detected high-end Apple Silicon with {cpu_count} cores, using {max_workers} workers"
                )
            elif is_apple_silicon:
                # For other M-series chips, use cpu_count - 1
                max_workers = max(1, cpu_count - 1)
                logger.info(
                    f"Detected Apple Silicon with {cpu_count} cores, using {max_workers} workers"
                )
            else:
                # For other systems, use default strategy
                max_workers = max(1, cpu_count - 1)
                logger.info(f"Using {max_workers} workers (system has {cpu_count} cores)")

        self.max_workers = max_workers

        logger.debug(
            f"Initialized DiarizationService with {diarizer_name} diarizer and {max_workers} workers"
        )

    @property
    def diarizer(self):
        """Lazy-load the diarizer when needed"""
        if self._diarizer is None:
            logger.debug(f"Initializing diarizer: {self.diarizer_name}")
            self._diarizer = get_diarizer(self.diarizer_name)
        return self._diarizer

    def diarize_chunk(
        self, chunk: AudioChunk, job: Optional[ProcessingJob] = None, **kwargs
    ) -> List[DiarizationSegment]:
        """
        Diarize a single audio chunk.

        Args:
            chunk: Audio chunk to process
            job: Optional processing job this chunk belongs to
            **kwargs: Additional arguments for the diarizer
                - timeout: Timeout in seconds (default: from service config)

        Returns:
            List of diarization segments
        """
        if not chunk or not chunk.chunk_path or not Path(chunk.chunk_path).exists():
            logger.warning(f"Cannot diarize chunk {chunk.id}: Invalid path or file not found")
            return []

        job_id = job.id if job else chunk.job_id

        # Check if we have cached results
        if job_id and self.repository.exists(job_id, chunk.id):
            segments = self.repository.get(job_id, chunk.id)
            if segments:
                logger.debug(f"Using cached diarization results for chunk {chunk.id}")
                return segments

        # Get appropriate diarizer
        logger.debug(f"Initializing diarizer '{self.diarizer_name}' for chunk {chunk.id}")
        diarizer = self.diarizer
        if not diarizer:
            logger.error(f"Failed to initialize diarizer '{self.diarizer_name}'")
            return []

        # Start monitoring this chunk
        self.monitor.start_chunk(chunk.id)

        # Set timeout
        timeout = kwargs.get("timeout", self.default_timeout)
        kwargs["timeout"] = timeout

        # Perform diarization
        try:
            logger.debug(f"Processing chunk {chunk.id} with path {chunk.chunk_path}")
            segments = diarizer.diarize(chunk.chunk_path, **kwargs)

            # Adjust segment times based on chunk start time
            segments = self._adjust_segment_times(segments, chunk)

            # Save results
            if segments:
                logger.debug(f"Found {len(segments)} segments for chunk {chunk.id}")
                if job_id:
                    self.repository.save(job_id, chunk.id, segments)
            else:
                logger.debug(f"No segments found for chunk {chunk.id}")

            return segments

        except DiarizationTimeoutError as e:
            logger.error(f"Timeout processing chunk {chunk.id}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error diarizing chunk {chunk.id}: {str(e)}")
            return []
        finally:
            self.monitor.end_chunk()

    def _get_chunk_object(self, chunk, job_id, chunk_repo=None):
        """
        Convert a chunk ID string to an AudioChunk object if necessary.

        Args:
            chunk: Either an AudioChunk object or a string chunk ID
            job_id: The job ID this chunk belongs to
            chunk_repo: Optional chunk repository for loading chunks

        Returns:
            AudioChunk: The audio chunk object
        """
        if isinstance(chunk, AudioChunk):
            return chunk

        # If we have a string chunk ID, try to get the chunk from the repository
        try:
            if chunk_repo and hasattr(chunk_repo, "get"):
                logger.debug(f"Loading chunk {chunk} from repository")
                return chunk_repo.get(chunk)
        except Exception as e:
            logger.debug(f"Could not load chunk {chunk} from repository: {str(e)}")

        # If we couldn't get it from the repository, create a stub chunk object
        logger.debug(f"Creating stub chunk object for ID: {chunk}")
        return AudioChunk(
            id=chunk,
            job_id=job_id,
            chunk_index=0,  # We don't know the index, so use 0 as default
            chunk_path=None,  # We don't have the path information
            start_time=0,
            end_time=0,
            status=ChunkStatus.PENDING,
        )

    def diarize_job(
        self,
        job: ProcessingJob,
        chunk_repo: Optional[any] = None,  # Will be ChunkRepository
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs,
    ) -> Dict[str, List[DiarizationSegment]]:
        """
        Diarize all chunks for a job.

        Args:
            job: The processing job to process
            chunk_repo: Optional chunk repository
            progress_callback: Optional callback for reporting progress
            **kwargs: Additional arguments for the diarizer

        Returns:
            Dictionary mapping chunk IDs to lists of diarization segments
        """
        logger.info(f"Starting diarization for job {job.id}")

        # Get chunks from job
        chunks = job.chunks

        if not chunks:
            logger.warning(f"No chunks found for job {job.id}")
            return {}

        # Pass the progress callback to the processing methods
        kwargs["progress_callback"] = progress_callback

        # Check if we're processing in parallel
        if job.parallel and self.max_workers > 1:
            logger.info(f"Using parallel processing with {self.max_workers} workers")
            return self._diarize_job_parallel(
                job=job,
                chunks=chunks,
                force_reprocess=job.force_reprocess,
                max_workers=self.max_workers,
                chunk_repo=chunk_repo,
                **kwargs,
            )
        else:
            logger.info("Using sequential processing")
            return self._diarize_job_sequential(
                job=job,
                chunks=chunks,
                force_reprocess=job.force_reprocess,
                chunk_repo=chunk_repo,
                **kwargs,
            )

    def _adjust_segment_times(
        self, segments: List[DiarizationSegment], chunk: AudioChunk
    ) -> List[DiarizationSegment]:
        """
        Adjust segment times based on chunk start time.

        Args:
            segments: List of diarization segments
            chunk: Audio chunk the segments belong to

        Returns:
            List of adjusted diarization segments
        """
        if not segments:
            return []

        # Get chunk start time
        chunk_start = getattr(chunk, "start_time", 0)

        # Adjust segment times
        adjusted_segments = []
        for segment in segments:
            adjusted_segment = DiarizationSegment(
                speaker_id=segment.speaker_id,
                start=segment.start + chunk_start,
                end=segment.end + chunk_start,
                score=segment.score,
            )
            adjusted_segments.append(adjusted_segment)

        return adjusted_segments

    def _diarize_job_parallel(
        self,
        job: ProcessingJob,
        chunks: List[AudioChunk],
        force_reprocess: bool,
        max_workers: int,
        chunk_repo=None,
        **kwargs,
    ) -> Dict[str, List[DiarizationSegment]]:
        """
        Process chunks in parallel with timeout monitoring.
        """
        start_time = time.time()
        results = {}

        # Extract progress callback from kwargs
        progress_callback = kwargs.pop("progress_callback", None)

        # Set timeout for each chunk
        timeout = kwargs.get("timeout", self.default_timeout)
        kwargs["timeout"] = timeout

        # Get chunks that need processing
        processed_chunks = []
        for chunk in chunks:
            if not force_reprocess and self.repository.exists(job.id, chunk.id):
                # Use cached results
                results[chunk.id] = self.repository.get(job.id, chunk.id)
                logger.debug(f"Using cached results for chunk {chunk.id}")
            else:
                processed_chunks.append(chunk)

        num_chunks = len(processed_chunks)
        if num_chunks == 0:
            logger.info("No chunks need processing")
            return results

        logger.info(f"Processing {num_chunks} chunks in parallel with {max_workers} workers")
        # Print clear separator for progress visibility
        print("\n" + "=" * 50)
        print(f"DIARIZATION PROGRESS: Starting processing of {num_chunks} chunks")
        print("=" * 50)

        # Calculate chunk batches for optimal processing
        # Process chunks in batches to avoid system overload while maintaining parallelism
        import multiprocessing
        import os
        import platform

        # Determine if we're on Apple Silicon for optimization
        is_apple_silicon = platform.system() == "Darwin" and "arm" in platform.machine()
        cpu_count = multiprocessing.cpu_count()

        # Consider it high-end if it has many cores (M3 Max has 16 cores)
        is_high_end = is_apple_silicon and cpu_count >= 12

        # Set optimal batch size based on hardware
        if is_high_end:
            # M3 Max can handle more concurrent work
            batch_size = min(num_chunks, max(max_workers, 8))
        elif is_apple_silicon:
            # Other M-series - slightly more conservative
            batch_size = min(num_chunks, max(max_workers, 6))
        else:
            # Default - be more conservative
            batch_size = min(num_chunks, max_workers)

        logger.info(f"Using batch size of {batch_size} for parallel processing")

        # Track overall progress
        total_completed = 0
        total_batches = (num_chunks + batch_size - 1) // batch_size

        # Create progress display
        # Use Rich if available (better terminal display), otherwise use tqdm
        master_progress = None
        batch_progress = None
        console = None

        if self.show_progress:
            if RICH_AVAILABLE:
                # Create a Rich console for progress display
                console = Console()

                # Print a separator line
                console.print("\n[bold blue]Starting Diarization[/bold blue]")

                # Create a Rich progress display
                master_progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=None),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    expand=True,
                )

                # Create the main task
                master_task_id = master_progress.add_task(
                    f"[cyan]Processing {num_chunks} chunks", total=num_chunks
                )

                # Start the progress display
                master_progress.start()
            else:
                # Fall back to tqdm if Rich is not available
                # Create a progress bar that writes directly to stdout to avoid logging conflicts
                master_pbar = tqdm.tqdm(
                    total=num_chunks,
                    desc="Overall diarization progress",
                    unit="chunk",
                    position=0,
                    file=sys.__stdout__,  # Use sys.__stdout__ to bypass any redirection
                )
                # Force a newline before the progress bar
                print("", file=sys.__stdout__)

        # Process chunks in parallel with timeout monitoring
        for batch_idx in range(0, num_chunks, batch_size):
            batch_chunks = processed_chunks[batch_idx : batch_idx + batch_size]
            batch_size_actual = len(batch_chunks)

            batch_num = batch_idx // batch_size + 1
            logger.info(
                f"Processing batch {batch_num}/{total_batches} with {batch_size_actual} chunks"
            )

            # Print progress at batch start - guaranteed to be visible
            print(f"\nBATCH {batch_num}/{total_batches}: Starting with {batch_size_actual} chunks")
            print(
                f"Overall progress: {total_completed}/{num_chunks} chunks ({total_completed/num_chunks*100:.1f}%)"
            )

            # Report batch progress to callback if provided
            if progress_callback:
                progress_callback(
                    total_completed, num_chunks, f"Starting batch {batch_num}/{total_batches}"
                )

            # Create batch progress display
            if self.show_progress:
                if RICH_AVAILABLE and master_progress:
                    # Add a batch task to the master progress
                    batch_task_id = master_progress.add_task(
                        f"[green]Batch {batch_num}/{total_batches}", total=batch_size_actual
                    )
                else:
                    # Fall back to tqdm
                    # Force a newline before each batch progress bar
                    print("", file=sys.__stdout__)
                    batch_pbar = tqdm.tqdm(
                        total=batch_size_actual,
                        desc=f"Batch {batch_num}/{total_batches}",
                        unit="chunk",
                        position=1,
                        leave=False,
                        file=sys.__stdout__,  # Use sys.__stdout__ to bypass any redirection
                    )

            completed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit batch of tasks
                future_to_chunk = {
                    executor.submit(
                        _process_chunk_directly,
                        chunk=chunk,
                        diarizer_name=self.diarizer_name,
                        timeout=timeout,
                        **kwargs,
                    ): chunk
                    for chunk in batch_chunks
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        segments = future.result()
                        results[chunk.id] = segments

                        # Save results to repository
                        if segments:
                            self.repository.save(job.id, chunk.id, segments)

                        completed += 1
                        total_completed += 1

                        # Update progress display
                        if self.show_progress:
                            if RICH_AVAILABLE and master_progress:
                                master_progress.update(master_task_id, advance=1)
                                master_progress.update(batch_task_id, advance=1)
                            else:
                                if "batch_pbar" in locals():
                                    batch_pbar.update(1)
                                if "master_pbar" in locals():
                                    master_pbar.update(1)

                        # Calculate and display completion percentage
                        completion_pct = (total_completed / num_chunks) * 100
                        # Print direct progress update every 10% or at the end of each batch
                        if (
                            completed % max(1, batch_size_actual // 4) == 0
                            or completed == batch_size_actual
                        ):
                            print(
                                f"PROGRESS: {total_completed}/{num_chunks} chunks processed ({completion_pct:.1f}%) - Batch {batch_num}/{total_batches}"
                            )

                        if (
                            completion_pct % 10 < 0.1 or completed == batch_size_actual
                        ):  # Log at 10% intervals or batch completion
                            logger.info(
                                f"Progress: {total_completed}/{num_chunks} chunks processed ({completion_pct:.1f}%)"
                            )

                        # Report progress to callback if provided
                        if progress_callback:
                            progress_callback(
                                total_completed,
                                num_chunks,
                                f"Processed chunk {chunk.id}, {total_completed}/{num_chunks} complete ({completion_pct:.1f}%)",
                            )

                    except concurrent.futures.TimeoutError:
                        logger.error(f"Chunk {chunk.id} processing timed out")
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk.id}: {str(e)}")

            # Print batch completion message - guaranteed to be visible
            print(
                f"BATCH {batch_num}/{total_batches} COMPLETED: {completed}/{batch_size_actual} chunks processed"
            )
            print(
                f"Overall progress: {total_completed}/{num_chunks} chunks ({total_completed/num_chunks*100:.1f}%)"
            )

            # Close batch progress display
            if self.show_progress:
                if RICH_AVAILABLE and master_progress:
                    # In Rich, we can just remove the batch task once it's done
                    master_progress.remove_task(batch_task_id)
                else:
                    # Close tqdm progress bar
                    if "batch_pbar" in locals():
                        batch_pbar.close()

                    # Print a blank line between batches for better readability
                    print("", file=sys.__stdout__)

        # Close master progress display
        if self.show_progress:
            if RICH_AVAILABLE and master_progress:
                master_progress.stop()
                # Print a completion message
                console.print(
                    f"\n[bold green]Diarization completed: {total_completed}/{num_chunks} chunks processed[/bold green]\n"
                )
            else:
                if "master_pbar" in locals():
                    master_pbar.close()
                # Print a blank line for separation
                print("", file=sys.__stdout__)

        # Print final completion message - guaranteed to be visible
        print("\n" + "=" * 50)
        print(f"DIARIZATION COMPLETED: {total_completed}/{num_chunks} chunks processed")
        time_taken = time.time() - start_time
        print(f"Total time: {time_taken:.2f} seconds ({num_chunks / time_taken:.2f} chunks/second)")
        print("=" * 50 + "\n")

        end_time = time.time()
        time_taken = end_time - start_time
        chunks_per_second = num_chunks / time_taken if time_taken > 0 else 0

        logger.info(
            f"Completed diarization for job {job.id}, processed {total_completed}/{num_chunks} chunks"
        )
        logger.info(
            f"Total processing time: {time_taken:.2f} seconds ({chunks_per_second:.2f} chunks/second)"
        )

        # Final progress callback
        if progress_callback:
            progress_callback(
                total_completed, num_chunks, f"Completed diarization in {time_taken:.2f} seconds"
            )

        return results

    def _diarize_job_sequential(
        self,
        job: ProcessingJob,
        chunks: List[AudioChunk],
        force_reprocess: bool,
        chunk_repo=None,
        **kwargs,
    ) -> Dict[str, List[DiarizationSegment]]:
        """
        Process chunks sequentially.
        """
        start_time = time.time()
        results = {}

        # Extract progress callback
        progress_callback = kwargs.pop("progress_callback", None)

        # Get chunks that need processing
        processed_chunks = []
        for chunk in chunks:
            if not force_reprocess and self.repository.exists(job.id, chunk.id):
                # Use cached results
                results[chunk.id] = self.repository.get(job.id, chunk.id)
                logger.debug(f"Using cached results for chunk {chunk.id}")
            else:
                processed_chunks.append(chunk)

        num_chunks = len(processed_chunks)
        if num_chunks == 0:
            logger.info("No chunks to process")
            return results

        logger.info(f"Processing {num_chunks} chunks sequentially")

        # Print clear separator for progress visibility
        print("\n" + "=" * 50)
        print(f"SEQUENTIAL DIARIZATION: Starting processing of {num_chunks} chunks")
        print("=" * 50)

        # Create progress display
        if self.show_progress:
            if RICH_AVAILABLE:
                # Create a Rich console for progress display
                console = Console()

                # Print a separator line
                console.print("\n[bold blue]Starting Sequential Diarization[/bold blue]")

                # Create a Rich progress display
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=None),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    expand=True,
                )

                # Create the main task
                task_id = progress.add_task(
                    f"[cyan]Processing {num_chunks} chunks", total=num_chunks
                )

                # Start the progress display
                progress.start()
            else:
                # Fall back to tqdm if Rich is not available
                # Force a newline before the progress bar
                print("", file=sys.__stdout__)

                # Create a progress bar that writes directly to stdout
                pbar = tqdm.tqdm(
                    total=num_chunks,
                    desc="Diarization progress",
                    unit="chunk",
                    file=sys.__stdout__,  # Use sys.__stdout__ to bypass any redirection
                )

        # Process each chunk
        for i, chunk in enumerate(processed_chunks):
            chunk_id = chunk.id

            # Check for cached results
            if not force_reprocess and self.repository.exists(job.id, chunk_id):
                cached_results = self.repository.get(job.id, chunk_id)
                if cached_results:
                    logger.debug(f"Using cached diarization results for chunk {chunk_id}")
                    results[chunk_id] = cached_results

                    # Update progress
                    if self.show_progress:
                        if RICH_AVAILABLE and "progress" in locals():
                            progress.update(
                                task_id,
                                advance=1,
                                description=f"[cyan]Using cached result for chunk {i+1}/{num_chunks}",
                            )
                        elif "pbar" in locals():
                            pbar.update(1)

                    if progress_callback:
                        progress_callback(
                            i + 1, num_chunks, f"Used cached results for chunk {chunk_id}"
                        )

                    continue

            # Report progress
            completion_pct = ((i + 1) / num_chunks) * 100
            # Print direct progress update every 10% or every chunk if few chunks
            if num_chunks <= 10 or i % max(1, num_chunks // 10) == 0 or i == num_chunks - 1:
                print(f"PROGRESS: {i+1}/{num_chunks} chunks processed ({completion_pct:.1f}%)")

            logger.debug(f"Processing chunk {i+1}/{num_chunks}: {chunk_id} ({completion_pct:.1f}%)")
            if progress_callback:
                progress_callback(i, num_chunks, f"Processing chunk {chunk_id}")

            # Process the chunk
            try:
                # Update progress description
                if self.show_progress and RICH_AVAILABLE and "progress" in locals():
                    progress.update(
                        task_id,
                        description=f"[cyan]Processing chunk {i+1}/{num_chunks}: {chunk_id}",
                    )

                segments = self.diarize_chunk(chunk, job, **kwargs)

                if segments:
                    results[chunk_id] = segments
                    logger.info(
                        f"Completed diarization for chunk {chunk_id}, found {len(segments)} segments"
                    )
                else:
                    logger.warning(f"No segments found for chunk {chunk_id}")

                # Update progress display
                if self.show_progress:
                    if RICH_AVAILABLE and "progress" in locals():
                        progress.update(task_id, advance=1)
                    elif "pbar" in locals():
                        pbar.update(1)

                # Report progress to callback
                if progress_callback:
                    progress_callback(
                        i + 1,
                        num_chunks,
                        f"Processed chunk {chunk_id}, found {len(segments) if segments else 0} segments",
                    )

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                # Still update progress even on error
                if self.show_progress:
                    if RICH_AVAILABLE and "progress" in locals():
                        progress.update(
                            task_id,
                            advance=1,
                            description=f"[red]Error processing chunk {i+1}/{num_chunks}",
                        )
                    elif "pbar" in locals():
                        pbar.update(1)

        # Print final completion message - guaranteed to be visible
        print("\n" + "=" * 50)
        print(
            f"SEQUENTIAL DIARIZATION COMPLETED: {len(results)}/{len(processed_chunks)} chunks processed"
        )
        time_taken = time.time() - start_time
        print(f"Total time: {time_taken:.2f} seconds ({num_chunks / time_taken:.2f} chunks/second)")
        print("=" * 50 + "\n")

        end_time = time.time()
        time_taken = end_time - start_time
        chunks_per_second = num_chunks / time_taken if time_taken > 0 else 0

        logger.info(
            f"Completed diarization for job {job.id}, processed {len(results)}/{len(processed_chunks)} chunks"
        )
        logger.info(
            f"Total processing time: {time_taken:.2f} seconds ({chunks_per_second:.2f} chunks/second)"
        )

        # Final progress callback
        if progress_callback:
            progress_callback(
                len(results), num_chunks, f"Completed diarization in {time_taken:.2f} seconds"
            )

        return results

    def close(self):
        """Close the service and release resources."""
        pass

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()

    def diarize_all(self, audio_path: Union[str, Path], **kwargs) -> List[DiarizationSegment]:
        """
        Diarize an entire audio file.

        Args:
            audio_path: Path to the audio file
            **kwargs: Additional arguments for the diarizer

        Returns:
            List of diarization segments
        """
        # Set timeout
        timeout = kwargs.get("timeout", self.default_timeout)
        kwargs["timeout"] = timeout

        return self.diarizer.diarize(audio_path, **kwargs)

    def process_job(
        self,
        job: ProcessingJob,
        chunks: List[AudioChunk],
        show_progress: bool = False,
        chunk_progress_callback=None,
        **kwargs,
    ) -> Dict[str, List[DiarizationSegment]]:
        """
        Process all chunks for a job.

        Args:
            job: The processing job
            chunks: List of audio chunks to process
            show_progress: Whether to show progress information
            chunk_progress_callback: Callback for per-chunk progress updates
            **kwargs: Additional arguments for the diarizer

        Returns:
            Dictionary mapping chunk IDs to lists of diarization segments
        """
        logger.info(f"Processing {len(chunks)} chunks for job {job.id}")

        # Initialize results dictionary
        results: Dict[str, List[DiarizationSegment]] = {}

        # Determine the processing mode
        if self.max_workers == 1 or len(chunks) == 1:
            # Sequential processing
            for i, chunk in enumerate(chunks):
                if show_progress:
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk.id}")

                # Skip already processed chunks
                if chunk.status in [ChunkStatus.DIARIZED, ChunkStatus.COMPLETED]:
                    logger.info(f"Chunk {chunk.id} already processed, loading results")
                    segments = self.repository.get(job.id, chunk.id)
                    results[chunk.id] = segments

                    if chunk_progress_callback:
                        chunk_progress_callback(chunk.id, 1.0, "Already processed")
                    continue

                # Process the chunk with progress updates
                if chunk_progress_callback:
                    # Create a wrapper that updates progress
                    def progress_wrapper(progress, message):
                        chunk_progress_callback(chunk.id, progress, message)

                    # Process with progress tracking
                    segments = self.diarize_chunk(
                        chunk, job_id=job.id, progress_callback=progress_wrapper, **kwargs
                    )
                else:
                    # Process without progress tracking
                    segments = self.diarize_chunk(chunk, job_id=job.id, **kwargs)

                results[chunk.id] = segments
        else:
            # Parallel processing
            logger.info(f"Using parallel processing with {self.max_workers} workers")

            # Create a list of tasks to process
            tasks = []
            for chunk in chunks:
                # Skip already processed chunks
                if chunk.status in [ChunkStatus.DIARIZED, ChunkStatus.COMPLETED]:
                    logger.info(f"Chunk {chunk.id} already processed, loading results")
                    segments = self.repository.get(job.id, chunk.id)
                    results[chunk.id] = segments

                    if chunk_progress_callback:
                        chunk_progress_callback(chunk.id, 1.0, "Already processed")
                    continue

                # Add the chunk to the task list
                tasks.append(chunk)

            # Process tasks in parallel
            if tasks:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    # Submit all tasks
                    future_to_chunk = {
                        executor.submit(
                            _process_chunk_directly,
                            chunk=chunk,
                            diarizer_name=self.diarizer_name,
                            **kwargs,
                        ): chunk
                        for chunk in tasks
                    }

                    # Process results as they complete
                    for i, future in enumerate(concurrent.futures.as_completed(future_to_chunk)):
                        chunk = future_to_chunk[future]
                        try:
                            segments = future.result()
                            results[chunk.id] = segments

                            if show_progress:
                                logger.info(f"Completed {i+1}/{len(tasks)} chunks")

                            if chunk_progress_callback:
                                chunk_progress_callback(
                                    chunk.id, 1.0, f"Completed {i+1}/{len(tasks)}"
                                )
                        except Exception as e:
                            logger.error(f"Error processing chunk {chunk.id}: {str(e)}")
                            if chunk_progress_callback:
                                chunk_progress_callback(chunk.id, 0.5, f"Error: {str(e)}")
                            raise

        return results

    def has_all_chunk_data(self, job_id: str) -> bool:
        """
        Check if all chunks for a job have been diarized.

        Args:
            job_id: ID of the job to check

        Returns:
            True if all chunks have been diarized, False otherwise
        """
        # Get all chunks for this job
        from pyhearingai.infrastructure.repositories.json_repositories import JsonChunkRepository

        chunk_repo = JsonChunkRepository()
        chunks = chunk_repo.get_by_job_id(job_id)

        if not chunks:
            return False

        # Check if all chunks have diarization data
        for chunk in chunks:
            if not self.repository.exists(job_id, chunk.id):
                return False

        return True
