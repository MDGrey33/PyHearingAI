"""
Tests for the ReconciliationService.

This module contains unit tests for the ReconciliationService and its components.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyhearingai.core.idempotent import ProcessingJob, ProcessingStatus
from pyhearingai.core.models import DiarizationSegment, Segment
from pyhearingai.reconciliation.service import ReconciliationService
from tests.conftest import create_processing_job_func


class TestReconciliationService(unittest.TestCase):
    """Tests for the ReconciliationService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temp directory for repositories
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir))

        # Create test job
        self.audio_file_path = Path(self.temp_dir) / "test_audio.wav"
        with open(self.audio_file_path, "wb") as f:
            f.write(b"\x00" * 100)  # Small dummy file

        # Use the helper function instead of direct instantiation
        self.job = create_processing_job_func(
            audio_path=self.audio_file_path,
            job_id="test-reconciliation-job",
            status=ProcessingStatus.PENDING,
        )

        # Create test segments
        self.diarization_segments = {
            "chunk1": [
                DiarizationSegment(speaker_id="SPEAKER1", start=0.0, end=2.5),
                DiarizationSegment(speaker_id="SPEAKER2", start=2.7, end=5.0),
            ],
            "chunk2": [
                DiarizationSegment(speaker_id="SPEAKER1", start=5.2, end=7.5),
                DiarizationSegment(speaker_id="SPEAKER2", start=7.7, end=10.0),
            ],
        }

        self.transcription_segments = {
            "chunk1": [
                Segment(text="Hello, this is speaker one.", start=0.0, end=2.5),
                Segment(text="And this is speaker two.", start=2.7, end=5.0),
            ],
            "chunk2": [
                Segment(text="Speaker one continues talking.", start=5.2, end=7.5),
                Segment(text="Speaker two concludes.", start=7.7, end=10.0),
            ],
        }

        self.segment_transcriptions = {
            "chunk1_segment_0_SPEAKER1": "Hello, this is speaker one.",
            "chunk1_segment_1_SPEAKER2": "And this is speaker two.",
            "chunk2_segment_0_SPEAKER1": "Speaker one continues talking.",
            "chunk2_segment_1_SPEAKER2": "Speaker two concludes.",
        }

        # Create reconciled segments
        self.reconciled_segments = [
            Segment(text="Hello, this is speaker one.", start=0.0, end=2.5, speaker_id="SPEAKER1"),
            Segment(text="And this is speaker two.", start=2.7, end=5.0, speaker_id="SPEAKER2"),
            Segment(
                text="Speaker one continues talking.", start=5.2, end=7.5, speaker_id="SPEAKER1"
            ),
            Segment(text="Speaker two concludes.", start=7.7, end=10.0, speaker_id="SPEAKER2"),
        ]

    def test_service_initialization(self):
        """Test that the service initializes correctly."""
        # Patch the adapter class at the point it's used
        with patch(
            "pyhearingai.reconciliation.service.GPT4ReconciliationAdapter"
        ) as mock_adapter_class:
            # Arrange
            mock_adapter = MagicMock()
            mock_adapter_class.return_value = mock_adapter

            # Act
            service = ReconciliationService(
                reconciliation_repository=None  # Let it create a default repository
            )

            # Assert
            # Verify the adapter was initialized correctly
            mock_adapter_class.assert_called_once()

    @patch("pyhearingai.infrastructure.repositories.json_repositories.JsonChunkRepository")
    @patch("pyhearingai.diarization.repositories.diarization_repository.DiarizationRepository")
    @patch(
        "pyhearingai.transcription.repositories.transcription_repository.TranscriptionRepository"
    )
    def test_reconcile(self, mock_tr_repo, mock_di_repo, mock_chunk_repo):
        """Test reconciling diarization and transcription results."""
        # Arrange
        mock_chunk_repo_instance = MagicMock()
        mock_chunk_repo_instance.get_by_job_id.return_value = [
            MagicMock(id="chunk1"),
            MagicMock(id="chunk2"),
        ]
        mock_chunk_repo.return_value = mock_chunk_repo_instance

        mock_di_repo_instance = MagicMock()
        mock_di_repo_instance.get.side_effect = (
            lambda job_id, chunk_id: self.diarization_segments.get(chunk_id, [])
        )
        mock_di_repo.return_value = mock_di_repo_instance

        mock_tr_repo_instance = MagicMock()
        mock_tr_repo_instance.get_chunk_transcription.side_effect = (
            lambda job_id, chunk_id: self.transcription_segments.get(chunk_id, [])
        )
        mock_tr_repo_instance.get_segment_transcription.side_effect = (
            lambda job_id, segment_id: self.segment_transcriptions.get(segment_id, "")
        )
        mock_tr_repo.return_value = mock_tr_repo_instance

        # Create a mock adapter
        mock_adapter = MagicMock()
        mock_adapter.reconcile.return_value = self.reconciled_segments

        # Create a mock repository
        mock_repo = MagicMock()
        mock_repo.has_reconciled_result.return_value = False

        # Create service with mocked dependencies
        service = ReconciliationService(
            reconciliation_repository=mock_repo,
            diarization_repository=mock_di_repo_instance,
            transcription_repository=mock_tr_repo_instance,
        )
        service.chunk_repository = mock_chunk_repo_instance
        service.adapter = mock_adapter  # Replace the adapter with our mock

        # Act
        result = service.reconcile(self.job)

        # Assert
        self.assertEqual(result, self.reconciled_segments)
        mock_adapter.reconcile.assert_called_once()

        # Use unittest.mock.ANY to match any metadata dictionary
        from unittest.mock import ANY

        mock_repo.save_reconciled_result.assert_called_once_with(
            self.job.id, self.reconciled_segments, ANY
        )

    @patch(
        "pyhearingai.reconciliation.repositories.reconciliation_repository.ReconciliationRepository"
    )
    def test_format_output(self, mock_repo):
        """Test formatting reconciled results."""
        # Arrange
        mock_repo_instance = MagicMock()
        mock_repo_instance.formatted_output_exists.return_value = False
        mock_repo.return_value = mock_repo_instance

        service = ReconciliationService(reconciliation_repository=mock_repo_instance)

        # Act - Test different format outputs
        with patch("pyhearingai.application.outputs.to_text") as mock_to_text:
            mock_to_text.return_value = "TEXT OUTPUT"
            text_result = service.format_output(self.job, self.reconciled_segments, "txt")
            self.assertEqual(text_result, "TEXT OUTPUT")

        with patch("pyhearingai.application.outputs.to_json") as mock_to_json:
            mock_to_json.return_value = '{"segments": []}'
            json_result = service.format_output(self.job, self.reconciled_segments, "json")
            self.assertEqual(json_result, '{"segments": []}')

        with patch("pyhearingai.application.outputs.to_srt") as mock_to_srt:
            mock_to_srt.return_value = "SRT OUTPUT"
            srt_result = service.format_output(self.job, self.reconciled_segments, "srt")
            self.assertEqual(srt_result, "SRT OUTPUT")

        # Assert - Should save all formatted outputs
        self.assertEqual(mock_repo_instance.save_formatted_output.call_count, 3)

    @patch(
        "pyhearingai.reconciliation.repositories.reconciliation_repository.ReconciliationRepository"
    )
    def test_save_output_file(self, mock_repo):
        """Test saving output to a file."""
        # Arrange
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_reconciled_result.return_value = self.reconciled_segments
        mock_repo.return_value = mock_repo_instance

        service = ReconciliationService(reconciliation_repository=mock_repo_instance)

        # Mock format_output method
        service.format_output = MagicMock(return_value="FORMATTED OUTPUT")

        # Act
        output_path = Path(self.temp_dir) / "output.txt"
        result_path = service.save_output_file(self.job, output_path)

        # Assert
        self.assertEqual(result_path, output_path)
        self.assertTrue(output_path.exists())
        with open(output_path, "r") as f:
            content = f.read()
            self.assertEqual(content, "FORMATTED OUTPUT")

        # Test format inference from extension
        service.format_output.reset_mock()
        json_path = Path(self.temp_dir) / "output.json"
        service.save_output_file(self.job, json_path)
        service.format_output.assert_called_once_with(self.job, self.reconciled_segments, "json")


if __name__ == "__main__":
    unittest.main()
