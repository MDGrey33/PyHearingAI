"""
Progress visualization utilities for displaying job progress.

This module provides utilities for displaying real-time progress in the terminal,
including progress bars, ETA calculation, and status updates.
"""

import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

from pyhearingai.core.idempotent import ProcessingJob, AudioChunk, ProcessingStatus


class ProgressTracker:
    """
    Tracks and visualizes progress for audio processing jobs.

    This class provides real-time feedback during processing, including:
    - Progress bars for overall job completion
    - Per-chunk progress tracking
    - Estimated time remaining
    - Current processing status
    """

    def __init__(
        self,
        job: ProcessingJob,
        chunks: List[AudioChunk],
        show_chunks: bool = False,
        width: int = 50,
        output_stream=sys.stdout,
    ):
        """
        Initialize the progress tracker.

        Args:
            job: The processing job being tracked
            chunks: List of audio chunks for the job
            show_chunks: Whether to show detailed per-chunk progress
            width: Width of the progress bar in characters
            output_stream: Stream to write progress updates to
        """
        self.job = job
        self.chunks = chunks
        self.show_chunks = show_chunks
        self.width = width
        self.output = output_stream

        # Progress tracking
        self.job_progress = 0.0
        self.chunk_progress: Dict[str, float] = {chunk.id: 0.0 for chunk in chunks}
        self.status_messages: Dict[str, str] = {}

        # Timing
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.eta: Optional[datetime] = None

        # Display state
        self.lines_written = 0

    def update_job_progress(self, progress: float, message: Optional[str] = None) -> None:
        """
        Update the overall job progress.

        Args:
            progress: Progress value between 0.0 and 1.0
            message: Optional status message
        """
        self.job_progress = min(max(progress, 0.0), 1.0)

        if message:
            self.status_messages["job"] = message

        self._update_display()

    def update_chunk_progress(
        self, chunk_id: str, progress: float, message: Optional[str] = None
    ) -> None:
        """
        Update progress for a specific chunk.

        Args:
            chunk_id: ID of the chunk to update
            progress: Progress value between 0.0 and 1.0
            message: Optional status message
        """
        if chunk_id in self.chunk_progress:
            self.chunk_progress[chunk_id] = min(max(progress, 0.0), 1.0)

            if message:
                self.status_messages[chunk_id] = message

            # Update overall progress based on chunks
            self.job_progress = sum(self.chunk_progress.values()) / len(self.chunk_progress)

            self._update_display()

    def _update_display(self) -> None:
        """Update the terminal display with current progress."""
        current_time = time.time()

        # Only update at most 5 times per second to avoid flickering
        if current_time - self.last_update_time < 0.2 and self.job_progress < 1.0:
            return

        self.last_update_time = current_time

        # Calculate ETA
        elapsed = current_time - self.start_time
        if self.job_progress > 0.01:  # Avoid division by zero or unreliable initial estimates
            total_estimated = elapsed / self.job_progress
            remaining = total_estimated - elapsed
            self.eta = datetime.now() + timedelta(seconds=remaining)

        # Clear previous lines
        if self.lines_written > 0:
            self._clear_lines(self.lines_written)
            self.lines_written = 0

        # Print job progress
        job_status = self.job.status.name if self.job else "UNKNOWN"
        self._print_progress_bar(
            self.job_progress,
            prefix=f"Job {self.job.id[:8]}... ({job_status}):",
            suffix=self.status_messages.get("job", "Processing..."),
        )
        self.lines_written += 1

        # Print ETA if available
        if self.eta:
            eta_str = self.eta.strftime("%H:%M:%S")
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            self.output.write(f"Elapsed: {elapsed_str} | ETA: {eta_str}\n")
            self.lines_written += 1

        # Print per-chunk progress if enabled
        if self.show_chunks:
            for i, chunk in enumerate(self.chunks):
                progress = self.chunk_progress.get(chunk.id, 0.0)
                status_msg = self.status_messages.get(chunk.id, f"Chunk {i+1}/{len(self.chunks)}")
                chunk_status = chunk.status.name if chunk else "UNKNOWN"

                self._print_progress_bar(
                    progress,
                    prefix=f"Chunk {i+1}: {chunk_status}:",
                    suffix=status_msg,
                    width=self.width - 10,  # Slightly narrower for chunks
                )
                self.lines_written += 1

        self.output.flush()

    def _print_progress_bar(
        self, progress: float, prefix: str = "", suffix: str = "", width: Optional[int] = None
    ) -> None:
        """
        Print a progress bar to the output stream.

        Args:
            progress: Progress value between 0.0 and 1.0
            prefix: Text to display before the progress bar
            suffix: Text to display after the progress bar
            width: Width of the progress bar in characters
        """
        if width is None:
            width = self.width

        filled_length = int(width * progress)
        bar = "█" * filled_length + "░" * (width - filled_length)
        percent = f"{progress * 100:.1f}%"

        self.output.write(f"\r{prefix} |{bar}| {percent} {suffix}\n")

    def _clear_lines(self, num_lines: int) -> None:
        """
        Clear a specified number of lines in the terminal.

        Args:
            num_lines: Number of lines to clear
        """
        # Move up num_lines and clear each line
        for _ in range(num_lines):
            self.output.write("\033[F")  # Move cursor up one line
            self.output.write("\033[K")  # Clear line

    def complete(self, message: str = "Processing complete") -> None:
        """
        Mark the job as complete and display final progress.

        Args:
            message: Final status message to display
        """
        self.update_job_progress(1.0, message)
        self.output.write("\n")  # Add an extra line after completion

    def error(self, message: str = "Error occurred during processing") -> None:
        """
        Display an error message.

        Args:
            message: Error message to display
        """
        self.status_messages["job"] = f"ERROR: {message}"
        self._update_display()
        self.output.write("\n")  # Add an extra line after error


def create_progress_callback(tracker: ProgressTracker):
    """
    Create a progress callback function for use with processing functions.

    Args:
        tracker: The progress tracker to update

    Returns:
        A callback function that can be passed to processing functions
    """

    def progress_callback(chunk_id_or_progress, progress_or_message=None, message=None):
        """
        Handle different callback signatures:
        1. progress_callback(chunk_id, progress, message) - for chunk-specific updates
        2. progress_callback(progress, message) - for job-level updates
        """
        if progress_or_message is None:
            # Called as progress_callback(progress, None)
            progress = chunk_id_or_progress
            tracker.update_job_progress(progress, None)
        elif message is None:
            # Called as progress_callback(progress, message)
            progress = chunk_id_or_progress
            message = progress_or_message
            tracker.update_job_progress(progress, message)
        else:
            # Called as progress_callback(chunk_id, progress, message)
            chunk_id = chunk_id_or_progress
            progress = progress_or_message
            tracker.update_chunk_progress(chunk_id, progress, message)

    return progress_callback
