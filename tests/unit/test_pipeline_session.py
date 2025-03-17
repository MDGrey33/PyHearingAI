"""
Tests for the pipeline_session context manager.

This module contains tests for the pipeline_session context manager, which
allows reusing resources across multiple transcription jobs.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyhearingai import pipeline_session
from pyhearingai.application.session import Session
from pyhearingai.core.models import Segment, TranscriptionResult


class TestPipelineSession:
    """Unit tests for the pipeline_session context manager."""

    @pytest.fixture
    def test_audio_paths(self):
        """Create temporary audio files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temp file paths
            audio_path1 = os.path.join(temp_dir, "test_audio1.wav")
            audio_path2 = os.path.join(temp_dir, "test_audio2.wav")

            # In a real test, we would create valid audio files here
            # For this test, we'll just create empty files
            with open(audio_path1, "w") as f:
                f.write("")
            with open(audio_path2, "w") as f:
                f.write("")

            yield audio_path1, audio_path2

    @patch("pyhearingai.application.session.WorkflowOrchestrator")
    def test_session_resource_sharing(self, mock_orchestrator_class, test_audio_paths):
        """Test that the session reuses resources across multiple transcriptions."""
        # Set up mock
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        mock_job1 = MagicMock()
        mock_job2 = MagicMock()
        mock_orchestrator.create_or_resume_job.side_effect = [mock_job1, mock_job2]

        mock_chunks = [MagicMock(), MagicMock()]
        mock_orchestrator.chunk_repository.get_by_job_id.return_value = mock_chunks

        mock_result1 = TranscriptionResult(segments=[Segment(text="Test 1", start=0.0, end=1.0)])
        mock_result2 = TranscriptionResult(segments=[Segment(text="Test 2", start=0.0, end=1.0)])
        mock_orchestrator.process_job.side_effect = [mock_result1, mock_result2]

        # Use the session to transcribe two files
        audio_path1, audio_path2 = test_audio_paths

        with pipeline_session(verbose=True) as session:
            result1 = session.transcribe(audio_path1)
            result2 = session.transcribe(audio_path2)

            # Verify results
            assert result1 is mock_result1
            assert result2 is mock_result2

        # Verify orchestrator was created only once
        mock_orchestrator_class.assert_called_once()

        # Verify create_or_resume_job was called twice with different audio paths
        assert mock_orchestrator.create_or_resume_job.call_count == 2

        # Convert string paths to Path objects for comparison
        path1 = Path(audio_path1)
        path2 = Path(audio_path2)

        # Get the actual paths from the call arguments
        actual_path1 = mock_orchestrator.create_or_resume_job.call_args_list[0][1]["audio_path"]
        actual_path2 = mock_orchestrator.create_or_resume_job.call_args_list[1][1]["audio_path"]

        # Compare the paths
        assert actual_path1 == path1
        assert actual_path2 == path2

        # Verify process_job was called twice
        assert mock_orchestrator.process_job.call_count == 2

        # Verify resources were cleaned up
        mock_orchestrator.diarization_service.close.assert_called_once()
        mock_orchestrator.transcription_service.close.assert_called_once()
        mock_orchestrator.reconciliation_service.close.assert_called_once()

    @patch("pyhearingai.application.session.WorkflowOrchestrator")
    def test_session_cleanup_on_exception(self, mock_orchestrator_class, test_audio_paths):
        """Test that resources are cleaned up even if an exception occurs."""
        # Set up mock
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        # Make process_job raise an exception
        mock_orchestrator.process_job.side_effect = RuntimeError("Test error")

        # Use the session
        audio_path1, _ = test_audio_paths

        with pytest.raises(RuntimeError):
            with pipeline_session() as session:
                session.transcribe(audio_path1)

        # Verify resources were cleaned up despite the exception
        mock_orchestrator.diarization_service.close.assert_called_once()
        mock_orchestrator.transcription_service.close.assert_called_once()
        mock_orchestrator.reconciliation_service.close.assert_called_once()

    @patch("pyhearingai.application.session.WorkflowOrchestrator")
    def test_session_custom_parameters(self, mock_orchestrator_class):
        """Test that session respects custom parameters."""
        # Create session with custom parameters
        custom_transcriber = {"model": "custom_model"}
        custom_diarizer = "custom_diarizer"
        custom_chunk_size = 30.0
        custom_overlap = 5.0

        with pipeline_session(
            transcriber=custom_transcriber,
            diarizer=custom_diarizer,
            chunk_size_seconds=custom_chunk_size,
            overlap_seconds=custom_overlap,
            max_workers=4,
            show_chunks=True,
            verbose=True,
        ) as session:
            # Verify session attributes
            assert session.transcriber == custom_transcriber
            assert session.diarizer == custom_diarizer
            assert session.chunk_size_seconds == custom_chunk_size
            assert session.overlap_seconds == custom_overlap

        # Verify orchestrator was created with custom parameters
        _, kwargs = mock_orchestrator_class.call_args
        assert kwargs["transcriber_name"] == custom_transcriber
        assert kwargs["diarizer_name"] == custom_diarizer
        assert kwargs["chunk_size"] == custom_chunk_size
        assert kwargs["max_workers"] == 4
        assert kwargs["show_chunks"] is True
        assert kwargs["verbose"] is True
