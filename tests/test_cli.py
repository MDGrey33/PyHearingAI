"""
Test suite for the PyHearingAI CLI.

This module contains tests for the CLI functionality including:
- Basic transcription
- Job management (resumption, listing)
- Parameter handling
- Progress visualization
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse
import uuid

from pyhearingai.cli import main, list_jobs, find_job_by_audio_path
from pyhearingai.core.idempotent import ProcessingJob, ProcessingStatus, AudioChunk
from pyhearingai.infrastructure.repositories.json_repositories import JsonJobRepository


@pytest.fixture
def mock_transcribe():
    """Mock the transcribe function to avoid actual processing."""
    with patch("pyhearingai.cli.transcribe") as mock:
        # Configure the mock to return a fake result
        mock_result = MagicMock()
        mock_result.save = MagicMock()
        mock.return_value = mock_result
        yield mock


@pytest.fixture
def mock_job_repo():
    """Mock the job repository."""
    with patch("pyhearingai.cli.JsonJobRepository") as mock_class:
        mock_repo = MagicMock()
        mock_class.return_value = mock_repo

        # Setup list_all to return empty list by default
        mock_repo.list_all.return_value = []

        yield mock_repo


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing."""
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "test_audio.wav")

    # Create an empty file
    with open(audio_path, "wb") as f:
        f.write(
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )

    yield audio_path

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_progress_tracker():
    """Mock the ProgressTracker class."""
    with patch("pyhearingai.application.progress.ProgressTracker") as mock_class:
        mock_tracker = MagicMock()
        mock_class.return_value = mock_tracker
        yield mock_tracker


class TestCLIBasic:
    """Test basic CLI functionality."""

    def test_cli_help(self, capsys):
        """Test that the help command works."""
        with pytest.raises(SystemExit) as excinfo:
            with patch("sys.argv", ["pyhearingai", "--help"]):
                main()

        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        assert "PyHearingAI - Transcribe audio with speaker diarization" in captured.out
        assert "--resume" in captured.out
        assert "--show-chunks" in captured.out

    def test_cli_version(self, capsys):
        """Test that the version command works."""
        with pytest.raises(SystemExit) as excinfo:
            with patch("sys.argv", ["pyhearingai", "--version"]):
                main()

        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        assert "pyhearingai" in captured.out

    def test_cli_missing_file(self, capsys):
        """Test behavior with a missing audio file."""
        with patch("sys.argv", ["pyhearingai", "nonexistent_file.wav"]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Error: Audio file not found" in captured.err


class TestCLITranscription:
    """Test CLI transcription functionality."""

    def test_basic_transcription(self, temp_audio_file, mock_transcribe):
        """Test basic transcription functionality."""
        with patch("sys.argv", ["pyhearingai", temp_audio_file]):
            result = main()

        assert result == 0

        # Verify the transcribe function was called with correct arguments
        mock_transcribe.assert_called_once()
        args, kwargs = mock_transcribe.call_args

        # Check for audio_path in kwargs
        assert "audio_path" in kwargs
        assert str(kwargs["audio_path"]) == str(temp_audio_file)
        assert kwargs["use_idempotent_processing"] == True
        assert kwargs["show_chunks"] == False

        # Verify result was saved
        mock_transcribe.return_value.save.assert_called_once()

    def test_transcription_with_options(self, temp_audio_file, mock_transcribe):
        """Test transcription with various CLI options."""
        with patch(
            "sys.argv",
            [
                "pyhearingai",
                temp_audio_file,
                "--show-chunks",
                "--max-workers",
                "4",
                "--chunk-size",
                "5.0",
                "--format",
                "json",
            ],
        ):
            result = main()

        assert result == 0

        # Verify transcribe was called with correct options
        mock_transcribe.assert_called_once()
        args, kwargs = mock_transcribe.call_args

        # Check for audio_path in kwargs
        assert "audio_path" in kwargs
        assert str(kwargs["audio_path"]) == str(temp_audio_file)
        assert kwargs["use_idempotent_processing"] == True
        assert kwargs["show_chunks"] == True
        assert kwargs["max_workers"] == 4
        assert kwargs["chunk_size"] == 5.0

        # Verify result was saved with json format
        mock_transcribe.return_value.save.assert_called_once()
        save_args, save_kwargs = mock_transcribe.return_value.save.call_args
        assert Path(save_args[0]).suffix == ".json"

    def test_legacy_mode(self, temp_audio_file, mock_transcribe):
        """Test the legacy mode (non-idempotent processing)."""
        with patch("sys.argv", ["pyhearingai", temp_audio_file, "--use-legacy"]):
            result = main()

        assert result == 0

        # Verify transcribe was called with use_idempotent_processing=False
        mock_transcribe.assert_called_once()
        args, kwargs = mock_transcribe.call_args

        # Check for audio_path in kwargs
        assert "audio_path" in kwargs
        assert str(kwargs["audio_path"]) == str(temp_audio_file)
        assert kwargs["use_idempotent_processing"] == False


class TestCLIJobManagement:
    """Test CLI job management functionality."""

    def test_list_jobs_empty(self, capsys, mock_job_repo):
        """Test listing jobs when none exist."""
        mock_job_repo.list_all.return_value = []

        with patch("sys.argv", ["pyhearingai", "--list-jobs"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "No jobs found" in captured.out

    def test_list_jobs(self, capsys, mock_job_repo):
        """Test listing jobs with some existing jobs."""
        # Create mock jobs
        job1 = ProcessingJob(
            id=str(uuid.uuid4()),
            original_audio_path="/path/to/audio1.wav",
            status=ProcessingStatus.COMPLETED,
            created_at="2023-01-01 10:00:00",
        )
        job2 = ProcessingJob(
            id=str(uuid.uuid4()),
            original_audio_path="/path/to/audio2.wav",
            status=ProcessingStatus.FAILED,
            created_at="2023-01-02 11:00:00",
        )

        mock_job_repo.list_all.return_value = [job1, job2]

        with patch("sys.argv", ["pyhearingai", "--list-jobs"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Found 2 jobs" in captured.out
        assert job1.id in captured.out
        assert job2.id in captured.out
        assert "COMPLETED" in captured.out
        assert "FAILED" in captured.out

    @pytest.mark.parametrize(
        "status,should_resume",
        [
            (ProcessingStatus.PENDING, True),
            (ProcessingStatus.IN_PROGRESS, True),
            (ProcessingStatus.FAILED, True),
            (ProcessingStatus.COMPLETED, False),
        ],
    )
    def test_resume_job(
        self, temp_audio_file, mock_job_repo, mock_transcribe, status, should_resume
    ):
        """Test resuming a job with various statuses."""
        job_id = str(uuid.uuid4())

        # Create a mock job
        job = ProcessingJob(
            id=job_id,
            original_audio_path=temp_audio_file,
            status=status,
            created_at="2023-01-01 10:00:00",
        )

        # Configure the mock repository
        mock_job_repo.get.return_value = job

        with patch("sys.argv", ["pyhearingai", "--resume", job_id]):
            result = main()

        if should_resume:
            assert result == 0
            # Verify transcribe was called with the job_id
            mock_transcribe.assert_called_once()
            args, kwargs = mock_transcribe.call_args

            # Check that audio_path and job_id are in kwargs
            assert "audio_path" in kwargs
            assert str(kwargs["audio_path"]) == str(temp_audio_file)
            assert "job_id" in kwargs
            assert kwargs["job_id"] == job_id
        else:
            # For completed jobs, should exit without calling transcribe
            assert result == 0
            mock_transcribe.assert_not_called()

    def test_resume_nonexistent_job(self, capsys, mock_job_repo):
        """Test attempting to resume a job that doesn't exist."""
        job_id = str(uuid.uuid4())

        # Configure the mock repository to return None
        mock_job_repo.get.return_value = None

        with patch("sys.argv", ["pyhearingai", "--resume", job_id]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Error: Job not found" in captured.err

    def test_find_job_by_audio_path(self, temp_audio_file, mock_job_repo):
        """Test finding a job by its audio path."""
        job_id = str(uuid.uuid4())

        # Create a mock job with the same path
        job = ProcessingJob(
            id=job_id,
            original_audio_path=temp_audio_file,
            status=ProcessingStatus.IN_PROGRESS,
            created_at="2023-01-01 10:00:00",
        )

        # Configure the mock to return our job
        mock_job_repo.list_all.return_value = [job]

        # Test the find_job_by_audio_path function
        with patch("pyhearingai.cli.JsonJobRepository", return_value=mock_job_repo):
            result = find_job_by_audio_path(temp_audio_file)

        assert result is not None
        assert result.id == job_id

        # Test with a path that doesn't match any job
        with patch("pyhearingai.cli.JsonJobRepository", return_value=mock_job_repo):
            result = find_job_by_audio_path("/nonexistent/path.wav")

        assert result is None


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_transcribe_error(self, temp_audio_file, mock_transcribe, capsys):
        """Test handling of errors during transcription."""
        # Configure the mock to raise an exception
        mock_transcribe.side_effect = Exception("Test error")

        with patch("sys.argv", ["pyhearingai", temp_audio_file]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Error: Test error" in captured.err

    def test_verbose_error(self, temp_audio_file, mock_transcribe, capsys):
        """Test verbose error handling with traceback."""
        # Configure the mock to raise an exception
        mock_transcribe.side_effect = Exception("Test error")

        with patch("sys.argv", ["pyhearingai", temp_audio_file, "--verbose"]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Error: Test error" in captured.err
        assert "Traceback" in captured.err  # Should include traceback in verbose mode


if __name__ == "__main__":
    pytest.main()
