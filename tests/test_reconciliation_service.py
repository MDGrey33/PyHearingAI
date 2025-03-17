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


def test_reconciliation_service_process(
    mock_repositories,
    test_job,
    test_chunks,
    test_diarization_segments,
    test_transcription_segments,
    test_reconciled_segments,
):
    """
    Test the reconciliation process.

    Given: A job with diarization and transcription results
    When: Calling reconcile
    Then: The service should return reconciled segments
    """
    # Arrange
    service = ReconciliationService(
        repository=mock_repositories["reconciliation"],
        diarization_repository=mock_repositories["diarization"],
        transcription_repository=mock_repositories["transcription"],
    )

    # Configure mocks
    mock_repositories["diarization"].get.return_value = test_diarization_segments
    mock_repositories[
        "transcription"
    ].get_chunk_transcription.return_value = test_transcription_segments
    mock_repositories[
        "reconciliation"
    ].get_reconciled_result.return_value = test_reconciled_segments

    # Mock the chunk repository
    with patch.object(service.chunk_repository, "get_by_job_id") as mock_get_chunks:
        mock_get_chunks.return_value = test_chunks

        # Mock the adapter
        with patch.object(service.adapter, "reconcile") as mock_reconcile:
            mock_reconcile.return_value = test_reconciled_segments

            # Act
            result = service.reconcile(test_job)

            # Assert
            assert len(result) == 4
            assert result[0].speaker_id == "Speaker 1"
            assert result[1].speaker_id == "Speaker 2"
            assert result[2].speaker_id == "Speaker 1"
            assert result[3].speaker_id == "Speaker 2"

            # Verify text is preserved
            assert result[0].text == "Hello, how are you today?"

            # Verify repository was called to save the result
            mock_repositories["reconciliation"].save_reconciled_result.assert_called_once()


def test_reconciliation_service_error_handling(mock_repositories, test_job, test_chunks):
    """
    Test error handling in the reconciliation service.

    Given: A job with missing diarization or transcription results
    When: Calling reconcile
    Then: The service should return an empty list with appropriate warnings
    """
    # Arrange
    service = ReconciliationService(
        repository=mock_repositories["reconciliation"],
        diarization_repository=mock_repositories["diarization"],
        transcription_repository=mock_repositories["transcription"],
    )

    # Configure mocks for the first test case - No diarization segments
    mock_repositories["diarization"].get.return_value = []
    mock_repositories["transcription"].get_chunk_transcription.return_value = []

    # Mock the chunk repository
    with patch.object(service.chunk_repository, "get_by_job_id") as mock_get_chunks:
        mock_get_chunks.return_value = test_chunks

        # Act
        result = service.reconcile(test_job)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0  # Empty list returned

        # Now test with diarization segments but no transcription segments
        mock_repositories["diarization"].get.return_value = [
            DiarizationSegment(start=0, end=1, speaker_id="S1")
        ]
        mock_repositories["transcription"].get_chunk_transcription.return_value = None

        # Act
        result = service.reconcile(test_job)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0  # Empty list returned
