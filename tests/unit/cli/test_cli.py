"""
Test suite for the PyHearingAI CLI.

This module contains tests for the CLI functionality including:
- Basic transcription
- Job management (resumption, listing)
- Parameter handling
- Progress visualization
"""

import argparse
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyhearingai.cli import find_job_by_audio_path, list_jobs, main
from pyhearingai.core.idempotent import AudioChunk, ProcessingJob, ProcessingStatus
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
