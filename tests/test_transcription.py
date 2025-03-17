"""
Tests for the TranscriptionService.

This module contains unit tests for the TranscriptionService and its components.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyhearingai.core.idempotent import AudioChunk, ChunkStatus, ProcessingJob, ProcessingStatus
from pyhearingai.core.models import Segment
from pyhearingai.transcription.service import TranscriptionService


class TestTranscriptionService(unittest.TestCase):
    """Tests for the TranscriptionService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temp directory for repositories
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir))

        # Create sample audio file
        self.audio_file_path = Path(self.temp_dir) / "test_audio.wav"
        with open(self.audio_file_path, "wb") as f:
            f.write(b"\x00" * 100)  # Small dummy file

        # Create test job
        self.job = ProcessingJob(
            original_audio_path=self.audio_file_path, id="test-job", status=ProcessingStatus.PENDING
        )

        # Create mock chunks
        self.chunk1 = AudioChunk(
            id="test-chunk-1",
            job_id="test-job",
            chunk_index=0,
            start_time=0.0,
            end_time=10.0,
            chunk_path=str(Path(self.temp_dir) / "chunk1.wav"),
            status=ChunkStatus.PENDING,
        )

        # Create empty audio files
        with open(self.chunk1.chunk_path, "wb") as f:
            f.write(b"\x00" * 100)  # Small dummy file

    @patch("pyhearingai.transcription.adapters.whisper.WhisperAdapter")
    def test_service_initialization(self, mock_adapter_class):
        """Test that the service initializes correctly."""
        # Arrange
        # Act
        service = TranscriptionService(
            transcriber_name="test_transcriber", repository=None, max_workers=2
        )

        # Assert
        self.assertEqual(service.transcriber_name, "test_transcriber")
        self.assertEqual(service.max_workers, 2)
        self.assertIsNotNone(service.repository)

    @patch(
        "pyhearingai.transcription.repositories.transcription_repository.TranscriptionRepository"
    )
    def test_transcribe_chunk(self, mock_repo):
        """Test transcribing a single chunk."""
        # Arrange
        mock_adapter = MagicMock()
        mock_segments = [Segment(text="Test transcript", start=0.0, end=5.0)]
        mock_adapter.transcribe_chunk.return_value = mock_segments

        mock_repo_instance = MagicMock()
        mock_repo.return_value = mock_repo_instance
        mock_repo_instance.chunk_exists.return_value = False

        service = TranscriptionService(repository=mock_repo_instance)
        service.adapter = mock_adapter  # Directly set the mock adapter

        # Act
        result = service.transcribe_chunk(self.chunk1)

        # Assert
        self.assertEqual(result, mock_segments)
        mock_adapter.transcribe_chunk.assert_called_once_with(self.chunk1)
        mock_repo_instance.save_chunk_transcription.assert_called_once()

    @patch(
        "pyhearingai.transcription.repositories.transcription_repository.TranscriptionRepository"
    )
    def test_cached_transcription(self, mock_repo):
        """Test that cached transcriptions are used when available."""
        # Arrange
        mock_adapter = MagicMock()

        mock_repo_instance = MagicMock()
        mock_repo.return_value = mock_repo_instance
        mock_repo_instance.chunk_exists.return_value = True
        mock_repo_instance.get_chunk_transcription.return_value = [
            Segment(text="Cached transcript", start=0.0, end=5.0)
        ]

        service = TranscriptionService(repository=mock_repo_instance)
        service.adapter = mock_adapter  # Directly set the mock adapter

        # Act
        result = service.transcribe_chunk(self.chunk1)

        # Assert
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Cached transcript")
        mock_adapter.transcribe_chunk.assert_not_called()

    @patch("pyhearingai.transcription.service._process_chunk_directly")
    @patch("pyhearingai.infrastructure.repositories.json_repositories.JsonChunkRepository")
    @patch(
        "pyhearingai.transcription.repositories.transcription_repository.TranscriptionRepository"
    )
    def test_transcribe_job_parallel(self, mock_repo, mock_chunk_repo, mock_process_chunk):
        """Test transcribing a job in parallel mode."""
        # Arrange
        mock_repo_instance = MagicMock()
        mock_repo.return_value = mock_repo_instance
        mock_repo_instance.chunk_exists.return_value = False

        mock_chunk_repo_instance = MagicMock()
        mock_chunk_repo.return_value = mock_chunk_repo_instance
        mock_chunk_repo_instance.get_by_job_id.return_value = [self.chunk1]

        mock_process_chunk.return_value = [Segment(text="Parallel transcript", start=0.0, end=5.0)]

        service = TranscriptionService(repository=mock_repo_instance, max_workers=2)

        # Act
        result = service.transcribe_job(self.job, parallel=True)

        # Assert
        self.assertTrue(result["success"])
        self.assertEqual(result["chunks_processed"], 1)
        self.assertEqual(result["chunks_failed"], 0)
        mock_process_chunk.assert_called_once()


if __name__ == "__main__":
    unittest.main()
