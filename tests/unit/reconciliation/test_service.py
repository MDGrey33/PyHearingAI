"""
Tests for the ReconciliationService.

This module tests the ReconciliationService class, which is responsible for
reconciling diarization and transcription results.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pyhearingai.core.idempotent import AudioChunk, ProcessingJob
from pyhearingai.core.models import DiarizationSegment, Segment
from pyhearingai.diarization.repositories.diarization_repository import DiarizationRepository
from pyhearingai.reconciliation.repositories.reconciliation_repository import (
    ReconciliationRepository,
)
from pyhearingai.reconciliation.service import ReconciliationService
from pyhearingai.transcription.repositories.transcription_repository import TranscriptionRepository


@pytest.fixture
def mock_repositories():
    """Create mock repositories for testing."""
    mock_reconciliation_repo = Mock(spec=ReconciliationRepository)
    mock_diarization_repo = Mock(spec=DiarizationRepository)
    mock_transcription_repo = Mock(spec=TranscriptionRepository)

    # Configure mock behavior
    mock_reconciliation_repo.reconciled_result_exists.return_value = False

    return {
        "reconciliation": mock_reconciliation_repo,
        "diarization": mock_diarization_repo,
        "transcription": mock_transcription_repo,
    }


@pytest.fixture
def test_job():
    """Create a test processing job."""
    return ProcessingJob(
        id="test_job_123",
        original_audio_path=Path("/path/to/audio.wav"),
        output_path=Path("/path/to/output"),
        chunk_size=10.0,
    )


@pytest.fixture
def test_chunks():
    """Create test audio chunks."""
    return [
        AudioChunk(
            id="chunk_1",
            job_id="test_job_123",
            chunk_path=Path("/path/to/chunks/chunk_1.wav"),
            start_time=0.0,
            end_time=10.0,
            chunk_index=0,
        ),
        AudioChunk(
            id="chunk_2",
            job_id="test_job_123",
            chunk_path=Path("/path/to/chunks/chunk_2.wav"),
            start_time=10.0,
            end_time=20.0,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def test_diarization_segments():
    """Create test diarization segments."""
    return [
        DiarizationSegment(start=0.0, end=5.0, speaker_id="SPEAKER_01"),
        DiarizationSegment(start=5.5, end=9.0, speaker_id="SPEAKER_02"),
        DiarizationSegment(start=10.0, end=15.0, speaker_id="SPEAKER_01"),
        DiarizationSegment(start=15.5, end=19.0, speaker_id="SPEAKER_02"),
    ]


@pytest.fixture
def test_transcription_segments():
    """Create test transcription segments."""
    return [
        Segment(text="Hello, how are you today?", start=0.0, end=5.0),
        Segment(text="I'm doing well, thank you.", start=5.5, end=9.0),
        Segment(text="That's great to hear.", start=10.0, end=15.0),
        Segment(text="Yes, the weather is nice.", start=15.5, end=19.0),
    ]


@pytest.fixture
def test_reconciled_segments():
    """Create test reconciled segments."""
    return [
        Segment(text="Hello, how are you today?", start=0.0, end=5.0, speaker_id="Speaker 1"),
        Segment(text="I'm doing well, thank you.", start=5.5, end=9.0, speaker_id="Speaker 2"),
        Segment(text="That's great to hear.", start=10.0, end=15.0, speaker_id="Speaker 1"),
        Segment(text="Yes, the weather is nice.", start=15.5, end=19.0, speaker_id="Speaker 2"),
    ]
