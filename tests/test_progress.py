"""
Tests for the progress visualization module.

This module contains tests for the ProgressTracker class and related functionality
for displaying real-time progress information during processing.
"""

import io
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Import the correct classes based on what's used in the tests
from pyhearingai.application.progress import ProgressTracker, create_progress_callback
from pyhearingai.core.idempotent import AudioChunk, ProcessingJob, ProcessingStatus


@pytest.fixture
def mock_job():
    """Create a mock processing job."""
    return ProcessingJob(
        id="test-job-12345678",
        original_audio_path="/path/to/audio.wav",
        status=ProcessingStatus.IN_PROGRESS,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


@pytest.fixture
def mock_chunks():
    """Create mock audio chunks."""
    return [
        AudioChunk(
            id="test-job-12345678_chunk_0",
            job_id="test-job-12345678",
            chunk_path="/path/to/chunks/chunk_0.wav",
            start_time=0.0,
            end_time=10.0,
            chunk_index=0,
            status=ProcessingStatus.PENDING,
        ),
        AudioChunk(
            id="test-job-12345678_chunk_1",
            job_id="test-job-12345678",
            chunk_path="/path/to/chunks/chunk_1.wav",
            start_time=10.0,
            end_time=20.0,
            chunk_index=1,
            status=ProcessingStatus.PENDING,
        ),
    ]


class TestProgressTracker:
    """Tests for the ProgressTracker class."""

    def test_init(self, mock_job, mock_chunks):
        """Test initialization of the ProgressTracker."""
        # Use StringIO to capture output
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=True, width=40, output_stream=output
        )

        assert tracker.job == mock_job
        assert tracker.chunks == mock_chunks
        assert tracker.show_chunks == True
        assert tracker.width == 40
        assert tracker.output == output

        # Check initial state
        assert tracker.job_progress == 0.0
        assert len(tracker.chunk_progress) == 2
        assert all(progress == 0.0 for progress in tracker.chunk_progress.values())
        assert tracker.lines_written == 0

    def test_update_job_progress(self, mock_job, mock_chunks):
        """Test updating the overall job progress."""
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=False, output_stream=output
        )

        # Mock the _update_display method to avoid rate limiting
        with patch.object(tracker, "_update_display") as mock_update_display:
            # Update progress
            tracker.update_job_progress(0.5, "Processing chunk 1/2")

            # Check state was updated
            assert tracker.job_progress == 0.5
            assert tracker.status_messages.get("job") == "Processing chunk 1/2"

            # Verify the _update_display method was called
            mock_update_display.assert_called_once()

    def test_update_chunk_progress(self, mock_job, mock_chunks):
        """Test updating progress for individual chunks."""
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=True, output_stream=output
        )

        # Mock the _update_display method to avoid rate limiting
        with patch.object(tracker, "_update_display") as mock_update_display:
            # Update chunk progress
            chunk_id = mock_chunks[0].id
            tracker.update_chunk_progress(chunk_id, 0.75, "Diarizing")

            # Check state was updated
            assert tracker.chunk_progress[chunk_id] == 0.75
            assert tracker.status_messages.get(chunk_id) == "Diarizing"

            # Job progress should be the average of chunk progress
            expected_job_progress = 0.75 / len(mock_chunks)
            assert tracker.job_progress == expected_job_progress

            # Verify the _update_display method was called
            mock_update_display.assert_called_once()

    def test_complete(self, mock_job, mock_chunks):
        """Test the complete method."""
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=False, output_stream=output
        )

        # Mock the _update_display method to avoid rate limiting
        with patch.object(tracker, "_update_display") as mock_update_display:
            # Mark as complete
            tracker.complete("All done!")

            # Job progress should be 1.0
            assert tracker.job_progress == 1.0

            # Verify the _update_display method was called
            mock_update_display.assert_called_once()

    def test_error(self, mock_job, mock_chunks):
        """Test the error method."""
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=False, output_stream=output
        )

        # Mock the _update_display method to avoid rate limiting
        with patch.object(tracker, "_update_display") as mock_update_display:
            # Report an error
            tracker.error("Something went wrong")

            # Verify the status message was set
            assert tracker.status_messages.get("job") == "ERROR: Something went wrong"

            # Verify the _update_display method was called
            mock_update_display.assert_called_once()

    def test_progress_calculation(self, mock_job, mock_chunks):
        """Test calculation of overall progress from chunk progress."""
        output = io.StringIO()

        # Add a second chunk to the mock_chunks list
        mock_chunks.append(
            AudioChunk(
                id="test-job-12345678_chunk_1",
                job_id="test-job-12345678",
                chunk_path="/path/to/chunks/chunk_1.wav",
                start_time=10.0,
                end_time=20.0,
                chunk_index=1,
                status=ProcessingStatus.PENDING,
            )
        )

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=True, output_stream=output
        )

        # Mock the _update_display method to avoid rate limiting
        with patch.object(tracker, "_update_display") as mock_update_display:
            # Update both chunks
            tracker.update_chunk_progress(mock_chunks[0].id, 0.5, "Processing")
            tracker.update_chunk_progress(mock_chunks[1].id, 0.75, "Processing")

            # Job progress should be the average
            expected_job_progress = (0.5 + 0.75) / 2
            assert tracker.job_progress == expected_job_progress

            # Verify the _update_display method was called twice
            assert mock_update_display.call_count == 2

    def test_progress_callback(self, mock_job, mock_chunks):
        """Test the progress callback creation function."""
        tracker = ProgressTracker(job=mock_job, chunks=mock_chunks, show_chunks=False)

        # Create a progress callback
        callback = create_progress_callback(tracker)

        # Spy on the update_job_progress method
        with patch.object(tracker, "update_job_progress") as mock_update:
            # Call the callback
            callback(0.42, "Working hard")

            # Verify the method was called with correct arguments
            mock_update.assert_called_once_with(0.42, "Working hard")


if __name__ == "__main__":
    pytest.main()
