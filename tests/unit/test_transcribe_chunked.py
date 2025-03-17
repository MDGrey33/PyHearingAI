"""
Unit tests for the transcribe_chunked function.

This module tests the transcribe_chunked function, which is a wrapper around the
idempotent processing workflow with customized chunk size and overlap.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pyhearingai import transcribe_chunked
from pyhearingai.core.models import TranscriptionResult, Segment
from tests.utils.test_helpers import TestFixtures


class TestTranscribeChunked:
    """Tests for the transcribe_chunked function."""

    @pytest.fixture
    def test_audio_path(self):
        """Create a test audio file for testing."""
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "test_audio.wav")

        # Create a 10-second test audio file
        TestFixtures.create_test_audio(audio_path, duration=10.0)

        yield audio_path

        # Clean up
        os.remove(audio_path)
        os.rmdir(temp_dir)

    @patch("pyhearingai.application.transcribe.WorkflowOrchestrator")
    def test_transcribe_chunked_creates_job_with_custom_settings(
        self, mock_orchestrator_class, test_audio_path
    ):
        """Test that transcribe_chunked creates a job with the specified chunk settings."""
        # Set up mock
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        mock_job = MagicMock()
        mock_orchestrator.create_or_resume_job.return_value = mock_job

        # Create a valid TranscriptionResult with segments
        mock_result = TranscriptionResult(segments=[Segment(text="Test", start=0.0, end=1.0)])
        mock_orchestrator.process_job.return_value = mock_result

        # Call the function with specific chunk settings
        chunk_size = 60.0
        overlap = 15.0
        result = transcribe_chunked(
            test_audio_path, chunk_size_seconds=chunk_size, overlap_seconds=overlap
        )

        # Verify orchestrator was created with the right settings
        mock_orchestrator_class.assert_called_once()
        _, kwargs = mock_orchestrator_class.call_args
        assert kwargs["chunk_size"] == chunk_size
        assert kwargs["show_chunks"] is True

        # Verify job was created with the right settings
        mock_orchestrator.create_or_resume_job.assert_called_once()
        _, kwargs = mock_orchestrator.create_or_resume_job.call_args
        # Note: chunk_size is now only passed to the orchestrator constructor, not to create_or_resume_job
        # This avoids redundantly passing the same parameter twice
        assert kwargs["overlap_duration"] == overlap

        # Verify job was processed
        mock_orchestrator.process_job.assert_called_once_with(
            job=mock_job,
            progress_tracker=mock_orchestrator.process_job.call_args[1]["progress_tracker"],
        )

        # Verify result
        assert result == mock_result

    @patch("pyhearingai.application.transcribe.WorkflowOrchestrator")
    def test_transcribe_chunked_saves_output(self, mock_orchestrator_class, test_audio_path):
        """Test that transcribe_chunked saves output when output_path is provided."""
        # Set up mock
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        mock_job = MagicMock()
        mock_orchestrator.create_or_resume_job.return_value = mock_job

        mock_result = MagicMock(spec=TranscriptionResult)
        mock_orchestrator.process_job.return_value = mock_result

        # Create a temporary output file
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_output:
            # Call the function with an output path
            transcribe_chunked(test_audio_path, output_path=temp_output.name, format="txt")

            # Verify result was saved
            mock_result.save.assert_called_once_with(Path(temp_output.name), format="txt")
