"""
Unit tests for the ProgressTracker class.

This module tests the functionality of the ProgressTracker class, including
progress bar rendering, ETA calculations, and status message handling.
"""

import io
import time
from unittest.mock import MagicMock, patch

import pytest

from pyhearingai.application.progress import ProgressTracker
from pyhearingai.core.idempotent import AudioChunk, ProcessingJob, ProcessingStatus


class TestProgressTracker:
    """Tests for the ProgressTracker class."""

    @pytest.fixture
    def mock_job(self):
        """Create a mock processing job for testing."""
        job = MagicMock(spec=ProcessingJob)
        job.id = "test-job-12345678"
        job.status = ProcessingStatus.PROCESSING
        return job

    @pytest.fixture
    def mock_chunks(self):
        """Create mock audio chunks for testing."""
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

    def test_init(self, mock_job, mock_chunks):
        """Test initialization of the ProgressTracker."""
        # Use StringIO to capture output
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=True, width=40, output_stream=output
        )

        assert tracker.job == mock_job
        assert tracker.chunks == mock_chunks
        assert tracker.show_chunks is True
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
            assert pytest.approx(tracker.job_progress, 0.01) == expected_job_progress

            # Verify the _update_display method was called
            mock_update_display.assert_called_once()

    def test_progress_bar_rendering(self, mock_job, mock_chunks):
        """Test rendering of progress bars."""
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=False, width=20, output_stream=output
        )

        # Set progress to 50%
        tracker.update_job_progress(0.5, "Processing...")

        # Get the output and check that a progress bar was rendered
        output_text = output.getvalue()
        assert "█" in output_text  # Progress bar should contain filled blocks
        assert "░" in output_text  # Progress bar should contain empty blocks
        assert "50.0%" in output_text  # Progress percentage should be displayed
        assert "Processing..." in output_text  # Status message should be displayed

    def test_eta_calculation(self, mock_job, mock_chunks):
        """Test calculation of estimated time to completion."""
        output = io.StringIO()

        with patch("time.time") as mock_time:
            # Mock time.time to return controlled values
            # First call (start time) = 1000
            # Second call (update time) = 1060 (60 seconds later)
            mock_time.side_effect = [1000, 1060]

            tracker = ProgressTracker(
                job=mock_job, chunks=mock_chunks, show_chunks=False, output_stream=output
            )

            # Set progress to 25% after 60 seconds
            # This should result in an ETA of 180 seconds from now (240 seconds total)
            tracker.update_job_progress(0.25, "Processing...")

            # Check that ETA was calculated
            assert tracker.eta is not None

            # Get the output and check for elapsed time and ETA
            output_text = output.getvalue()
            assert "Elapsed:" in output_text
            assert "ETA:" in output_text

    def test_complete(self, mock_job, mock_chunks):
        """Test marking a job as complete."""
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=False, output_stream=output
        )

        # Mark as complete
        tracker.complete("All done!")

        # Check state was updated
        assert tracker.job_progress == 1.0
        assert tracker.status_messages.get("job") == "All done!"

        # Get the output and check for completed state
        output_text = output.getvalue()
        assert "100.0%" in output_text
        assert "All done!" in output_text

    def test_error(self, mock_job, mock_chunks):
        """Test marking a job as having an error."""
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=False, output_stream=output
        )

        # Report an error
        tracker.error("Something went wrong")

        # Check state was updated
        assert tracker.status_messages.get("job") == "ERROR: Something went wrong"

        # Get the output and check for error message
        output_text = output.getvalue()
        assert "ERROR: Something went wrong" in output_text

    def test_multiple_chunk_progress(self, mock_job, mock_chunks):
        """Test updating progress for multiple chunks."""
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=True, output_stream=output
        )

        # Update progress for both chunks
        tracker.update_chunk_progress(mock_chunks[0].id, 0.5, "Processing chunk 1")
        tracker.update_chunk_progress(mock_chunks[1].id, 0.75, "Processing chunk 2")

        # Job progress should be the average of chunk progress
        expected_job_progress = (0.5 + 0.75) / 2
        assert pytest.approx(tracker.job_progress, 0.01) == expected_job_progress

        # Get the output and check for both chunk progress bars
        output_text = output.getvalue()
        assert "Chunk 1" in output_text
        assert "Chunk 2" in output_text
        assert "Processing chunk 1" in output_text
        assert "Processing chunk 2" in output_text

    def test_clear_lines(self, mock_job, mock_chunks):
        """Test clearing of previously written lines."""
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=True, output_stream=output
        )

        # Set some initial progress
        tracker.update_job_progress(0.25, "Initial progress")

        # Get the output and count the number of lines
        initial_output = output.getvalue()
        initial_line_count = initial_output.count("\n")

        # Reset the output buffer to simulate a fresh terminal
        output.truncate(0)
        output.seek(0)

        # Update progress again, which should clear previous lines
        with patch.object(tracker, "_clear_lines") as mock_clear_lines:
            tracker.update_job_progress(0.5, "Updated progress")
            mock_clear_lines.assert_called_once_with(tracker.lines_written)

    def test_rate_limiting(self, mock_job, mock_chunks):
        """Test rate limiting of display updates."""
        output = io.StringIO()

        with patch("time.time") as mock_time:
            # Mock time.time to simulate rapid updates
            # Initial time = 1000
            # First update = 1000.1 (0.1 seconds later)
            # Second update = 1000.15 (0.05 seconds later)
            # Third update = 1000.3 (0.15 seconds later)
            mock_time.side_effect = [1000, 1000, 1000.1, 1000.15, 1000.3]

            tracker = ProgressTracker(
                job=mock_job, chunks=mock_chunks, show_chunks=False, output_stream=output
            )

            # Mock _clear_lines to track calls
            with patch.object(tracker, "_clear_lines") as mock_clear_lines:
                # First update - should display (more than 0.2s since creation)
                tracker.update_job_progress(0.25, "Update 1")
                assert mock_clear_lines.call_count == 0  # No previous lines to clear

                # Second update - should not display (less than 0.2s since last update)
                tracker.update_job_progress(0.5, "Update 2")
                assert mock_clear_lines.call_count == 0  # Still no clear calls

                # Third update - should display (more than 0.2s since last displayed update)
                tracker.update_job_progress(0.75, "Update 3")
                assert mock_clear_lines.call_count == 1  # Should clear previous lines

    def test_progress_bounds(self, mock_job, mock_chunks):
        """Test that progress values are kept within bounds [0.0, 1.0]."""
        output = io.StringIO()

        tracker = ProgressTracker(
            job=mock_job, chunks=mock_chunks, show_chunks=False, output_stream=output
        )

        # Test with out-of-bounds values
        tracker.update_job_progress(-0.5, "Negative progress")
        assert tracker.job_progress == 0.0  # Should be clamped to 0.0

        tracker.update_job_progress(1.5, "Excessive progress")
        assert tracker.job_progress == 1.0  # Should be clamped to 1.0

        # Test chunk progress bounds
        chunk_id = mock_chunks[0].id
        tracker.update_chunk_progress(chunk_id, -0.2, "Negative chunk progress")
        assert tracker.chunk_progress[chunk_id] == 0.0  # Should be clamped to 0.0

        tracker.update_chunk_progress(chunk_id, 2.0, "Excessive chunk progress")
        assert tracker.chunk_progress[chunk_id] == 1.0  # Should be clamped to 1.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
