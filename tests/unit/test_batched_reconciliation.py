#!/usr/bin/env python3
"""
Unit tests for the batched reconciliation feature.

Tests the functionality of ReconciliationService.reconcile_batched method
without requiring full CLI execution.
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from pyhearingai.core.idempotent import AudioChunk, ProcessingJob
from pyhearingai.core.models import DiarizationSegment, Segment
from pyhearingai.reconciliation.service import ReconciliationService, BatchConfig


class TestBatchedReconciliation(unittest.TestCase):
    """Test cases for batched reconciliation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test job
        self.job = ProcessingJob(
            original_audio_path="test_audio.wav",
            id="test-job-id",
            chunk_duration=10.0,
            start_time=0.0,
            end_time=310.0  # Just over the 5-minute threshold
        )
        
        # Create test chunks
        self.chunks = []
        for i in range(31):  # 31 chunks of 10 seconds each
            chunk = AudioChunk(
                job_id=self.job.id,
                chunk_path=Path(f"/tmp/chunk_{i:04d}.wav"),
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                chunk_index=i
            )
            # Set id after creation to avoid lint error
            chunk.id = f"chunk-{i}"
            self.chunks.append(chunk)
            
        self.job.chunks = [chunk.id for chunk in self.chunks]
        self.job.total_chunks = len(self.chunks)

    def test_reconcile_batched(self):
        """Test batched reconciliation workflow."""
        # Create mock repositories
        mock_reconciliation_repo = MagicMock()
        mock_diarization_repo = MagicMock()
        mock_transcription_repo = MagicMock()
        mock_chunk_repo = MagicMock()
        
        # Configure mocks
        mock_reconciliation_repo.has_reconciled_result.return_value = False
        mock_chunk_repo.get_by_job_id.return_value = self.chunks
        
        # Set up diarization and transcription data
        for chunk in self.chunks:
            # Mock diarization results
            diarization_segments = [
                DiarizationSegment(
                    speaker_id=f"SPEAKER_{i}",
                    start=chunk.start_time + i,
                    end=chunk.start_time + i + 0.5,
                    score=0.95
                ) for i in range(3)
            ]
            mock_diarization_repo.get.return_value = diarization_segments
            
            # Mock transcription results
            transcription_segments = []
            for i in range(3):
                segment = Segment(
                    start=chunk.start_time + i,
                    end=chunk.start_time + i + 0.5,
                    text=f"Sample text {i}"
                )
                transcription_segments.append(segment)
                # Set a fixed segment ID key for mocking purposes
                seg_key = f"segment_{chunk.id}_{i}"
                mock_transcription_repo.get_segment_transcription.return_value = f"Transcription for {seg_key}"
            
            mock_transcription_repo.get_transcription_segments.return_value = transcription_segments
        
        # Create the service with our mocks
        service = ReconciliationService(
            reconciliation_repository=mock_reconciliation_repo,
            diarization_repository=mock_diarization_repo,
            transcription_repository=mock_transcription_repo,
        )
        service.chunk_repository = mock_chunk_repo
        
        # Configure batch config
        batch_config = BatchConfig(
            batch_size_seconds=180,
            batch_overlap_seconds=10,
            max_tokens_per_batch=4000
        )
        
        # Mock the adapter's reconcile method
        with patch.object(service.adapter, 'reconcile') as mock_reconcile:
            # Configure the mock to return a list of segments
            mock_reconcile.return_value = [
                Segment(
                    start=0.0,
                    end=5.0,
                    text="Reconciled text",
                    speaker_id="SPEAKER_00"
                )
            ]
            
            # Call method under test
            result = service.reconcile_batched(
                job=self.job,
                config=batch_config,
            )
            
            # Verify the call flow
            self.assertEqual(mock_chunk_repo.get_by_job_id.call_count, 1)
            self.assertEqual(mock_reconcile.call_count, 2)
            self.assertEqual(mock_reconciliation_repo.save_reconciled_result.call_count, 3)  # Interim + final + metadata
            
            # Check result
            self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main() 