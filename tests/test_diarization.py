"""
Tests for the DiarizationService class.

This module tests the functionality of the diarization service,
including initialization, chunk processing, job processing, error handling,
and more.
"""

import os
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
from pytest import fixture

from pyhearingai.core.idempotent import AudioChunk, ChunkStatus, ProcessingJob, ProcessingStatus
from pyhearingai.core.models import DiarizationSegment, Segment
from pyhearingai.diarization.repositories.diarization_repository import DiarizationRepository
from pyhearingai.diarization.service import DiarizationService
from pyhearingai.infrastructure.repositories.json_repositories import (
    JsonChunkRepository,
    JsonJobRepository,
)
from tests.conftest import create_processing_job_func


def create_test_job(
    job_id, audio_path, chunks=None, status=None, parallel=False, force_reprocess=False
):
    """
    Create a ProcessingJob instance with the given parameters.
    Handles differences between constructor signatures in different versions.
    """
    job = create_processing_job_func(audio_path=str(audio_path), job_id=job_id, status=status)

    # Add custom attributes
    job.parallel = parallel
    job.force_reprocess = force_reprocess

    # Set chunks if provided
    if chunks is not None:
        job.chunks = chunks

    return job


class TestDiarizationService(unittest.TestCase):
    """
    Test cases for the DiarizationService class.

    This class tests various aspects of the DiarizationService functionality:
    - Initialization
    - Single chunk processing
    - Job processing (sequential and parallel)
    - Error handling
    - Caching behavior
    """

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create a test audio file
        self.test_audio_path = self.test_dir / "test_audio.wav"
        with open(self.test_audio_path, "wb") as f:
            f.write(b"dummy audio data")

        # Mock repository
        self.mock_repository = MagicMock(spec=DiarizationRepository)

        # Create a mock diarizer
        self.mock_diarizer = MagicMock()

        # Create the service with the patch
        with patch(
            "pyhearingai.diarization.service.DiarizationService.diarizer", new_callable=PropertyMock
        ) as self.mock_diarizer_prop:
            # Set the mock diarizer to be returned when the property is accessed
            self.mock_diarizer_prop.return_value = self.mock_diarizer

            # Create the service
            self.service = DiarizationService(
                diarizer_name="mock_diarizer", repository=self.mock_repository, max_workers=2
            )

        # Create a test chunk
        self.test_chunk = AudioChunk(
            id=str(uuid.uuid4()),
            job_id=str(uuid.uuid4()),
            chunk_path=str(self.test_audio_path),
            start_time=0,
            end_time=10,
            chunk_index=0,
            status=ChunkStatus.PENDING,  # Use a valid status
        )

        # Create test segments
        self.test_segments = [
            DiarizationSegment(
                speaker_id="SPEAKER_01", start=1.0, end=2.5  # Use speaker_id instead of speaker
            ),
            DiarizationSegment(
                speaker_id="SPEAKER_02", start=3.0, end=4.5  # Use speaker_id instead of speaker
            ),
        ]

        # Create a test job
        self.test_job = create_test_job(
            job_id=self.test_chunk.job_id,
            audio_path=self.test_audio_path,
            chunks=[self.test_chunk],
            parallel=False,
            force_reprocess=False,
        )

        # Create a test parallel job
        self.test_parallel_job = create_test_job(
            job_id=str(uuid.uuid4()),
            audio_path=self.test_audio_path,
            chunks=[self.test_chunk],
            parallel=True,
        )

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
        self.service.close()

    def test_initialization(self):
        """Test that the service initializes correctly."""
        self.assertEqual(self.service.diarizer_name, "mock_diarizer")
        self.assertEqual(self.service.repository, self.mock_repository)
        self.assertEqual(self.service.max_workers, 2)

    def test_diarize_chunk(self):
        """Test diarizing a single chunk."""
        # Set up the mock diarizer to return test segments
        self.mock_diarizer.diarize.return_value = self.test_segments

        # Set repository to return no cached results
        self.mock_repository.exists.return_value = False

        # Call the method with the diarizer property patched
        with patch(
            "pyhearingai.diarization.service.DiarizationService.diarizer", new_callable=PropertyMock
        ) as mock_diarizer_prop:
            mock_diarizer_prop.return_value = self.mock_diarizer
            result = self.service.diarize_chunk(self.test_chunk)

        # Verify results
        self.assertEqual(result, self.test_segments)
        self.mock_diarizer.diarize.assert_called_once_with(str(self.test_audio_path), timeout=7200)
        self.mock_repository.save.assert_called_once_with(
            self.test_chunk.job_id, self.test_chunk.id, self.test_segments
        )

    def test_diarize_chunk_with_cache(self):
        """Test diarizing a chunk with cached results."""
        # Set repository to return cached results
        self.mock_repository.exists.return_value = True
        self.mock_repository.get.return_value = self.test_segments

        # Call the method with the diarizer property patched
        with patch(
            "pyhearingai.diarization.service.DiarizationService.diarizer", new_callable=PropertyMock
        ) as mock_diarizer_prop:
            mock_diarizer_prop.return_value = self.mock_diarizer
            result = self.service.diarize_chunk(self.test_chunk)

        # Verify results
        self.assertEqual(result, self.test_segments)
        self.mock_diarizer.diarize.assert_not_called()
        self.mock_repository.save.assert_not_called()

    def test_diarize_chunk_error(self):
        """Test error handling when diarizing a chunk."""
        # Set up the mock diarizer to raise an exception
        self.mock_diarizer.diarize.side_effect = Exception("Test error")

        # Set repository to return no cached results
        self.mock_repository.exists.return_value = False

        # Call the method with the diarizer property patched
        with patch(
            "pyhearingai.diarization.service.DiarizationService.diarizer", new_callable=PropertyMock
        ) as mock_diarizer_prop:
            mock_diarizer_prop.return_value = self.mock_diarizer
            result = self.service.diarize_chunk(self.test_chunk)

        # Verify results
        self.assertEqual(result, [])
        self.mock_diarizer.diarize.assert_called_once_with(str(self.test_audio_path), timeout=7200)
        self.mock_repository.save.assert_not_called()

    def test_diarize_job_sequential(self):
        """Test diarizing a job sequentially."""
        # Set repository to return no cached results
        self.mock_repository.exists.return_value = False

        # Patch the diarize_chunk method to return test segments
        with patch.object(
            self.service, "diarize_chunk", return_value=self.test_segments
        ) as mock_diarize_chunk:
            # Call the method
            result = self.service.diarize_job(self.test_job)

        # Verify results
        self.assertEqual(result, {self.test_chunk.id: self.test_segments})
        # Verify diarize_chunk was called
        mock_diarize_chunk.assert_called_once_with(self.test_chunk, self.test_job)

    def test_diarize_job_parallel(self):
        """Test diarizing a job in parallel."""
        # Set repository to return no cached results
        self.mock_repository.exists.return_value = False

        # Instead of patching _process_chunk_directly, we'll patch the _diarize_job_parallel method
        expected_result = {self.test_chunk.id: self.test_segments}

        with patch.object(
            self.service, "_diarize_job_parallel", return_value=expected_result
        ) as mock_parallel:
            # Call the method
            result = self.service.diarize_job(self.test_parallel_job)

            # Verify the method was called
            mock_parallel.assert_called_once()

            # Verify results
            self.assertEqual(result, expected_result)

    def test_get_chunk_object_with_audio_chunk(self):
        """Test getting a chunk object from an existing AudioChunk."""
        result = self.service._get_chunk_object(self.test_chunk, self.test_job.id)
        self.assertEqual(result, self.test_chunk)

    def test_get_chunk_object_with_string(self):
        """Test getting a chunk object from a string ID."""
        chunk_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())

        # Mock the chunk repository
        mock_chunk_repo = MagicMock()
        mock_chunk_repo.get.return_value = self.test_chunk

        # Call the method
        result = self.service._get_chunk_object(chunk_id, job_id, mock_chunk_repo)

        # Verify results
        self.assertEqual(result, self.test_chunk)
        mock_chunk_repo.get.assert_called_once_with(chunk_id)

    def test_get_chunk_object_with_string_no_repo(self):
        """Test getting a chunk object from a string ID without a repository."""
        chunk_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())

        # Call the method
        result = self.service._get_chunk_object(chunk_id, job_id)

        # Verify we get a stub object
        self.assertEqual(result.id, chunk_id)
        self.assertEqual(result.job_id, job_id)

    def test_adjust_segment_times(self):
        """Test adjusting segment times based on chunk start time."""
        # Create a chunk with a non-zero start time
        chunk = AudioChunk(
            id=str(uuid.uuid4()),
            job_id=str(uuid.uuid4()),
            chunk_path=str(self.test_audio_path),
            start_time=10.0,  # Start at 10 seconds
            end_time=20.0,
            chunk_index=0,
            status=ChunkStatus.PENDING,  # Use a valid status
        )

        # Create some segments
        segments = [
            DiarizationSegment(
                speaker_id="SPEAKER_01",  # Use speaker_id instead of speaker
                start=1.0,  # 1 second into the chunk
                end=2.5,  # 2.5 seconds into the chunk
            ),
            DiarizationSegment(
                speaker_id="SPEAKER_02",  # Use speaker_id instead of speaker
                start=3.0,  # 3 seconds into the chunk
                end=4.5,  # 4.5 seconds into the chunk
            ),
        ]

        # Adjust segment times
        adjusted_segments = self.service._adjust_segment_times(segments, chunk)

        # Verify results - should add the chunk start time
        self.assertEqual(len(adjusted_segments), 2)
        self.assertEqual(adjusted_segments[0].start, 11.0)  # 10 + 1
        self.assertEqual(adjusted_segments[0].end, 12.5)  # 10 + 2.5
        self.assertEqual(adjusted_segments[1].start, 13.0)  # 10 + 3
        self.assertEqual(adjusted_segments[1].end, 14.5)  # 10 + 4.5


if __name__ == "__main__":
    unittest.main()
