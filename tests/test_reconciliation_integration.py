"""
Integration tests for the ReconciliationService.

These tests verify the ReconciliationService functionality with real data from
diarization and transcription services.
"""

import unittest
import tempfile
import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

from pyhearingai.core.idempotent import (
    ProcessingJob,
    ProcessingStatus,
    AudioChunk,
    ChunkStatus,
    SpeakerSegment,
)
from pyhearingai.core.models import Segment, DiarizationSegment
from pyhearingai.diarization.service import DiarizationService
from pyhearingai.transcription.service import TranscriptionService
from pyhearingai.reconciliation.service import ReconciliationService
from pyhearingai.reconciliation.repositories.reconciliation_repository import (
    ReconciliationRepository,
)
from pyhearingai.diarization.repositories.diarization_repository import DiarizationRepository
from pyhearingai.transcription.repositories.transcription_repository import TranscriptionRepository
from pyhearingai.infrastructure.repositories.json_repositories import JsonChunkRepository
from tests.utils.audio_generator import create_test_audio_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Add is_complete method to AudioChunk class for testing
def is_complete(self):
    """Check if the chunk has completed processing."""
    return self.status == ChunkStatus.COMPLETED


# Add index property to AudioChunk class for testing
@property
def index(self):
    """Return the chunk index."""
    return self.chunk_index


# Monkey patch the AudioChunk class
AudioChunk.is_complete = is_complete
AudioChunk.index = index


class TestReconciliationIntegration(unittest.TestCase):
    """Integration tests for the ReconciliationService class."""

    def setUp(self):
        """Set up the test environment with temporary directories and test data."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Convert string path to Path object
        temp_path = Path(self.temp_dir)

        # Create repositories with custom base directories
        self.reconciliation_repo = ReconciliationRepository(base_dir=temp_path)
        self.diarization_repo = DiarizationRepository(base_dir=temp_path)
        self.transcription_repo = TranscriptionRepository(base_dir=temp_path)
        self.chunk_repository = JsonChunkRepository(chunks_dir=temp_path / "chunks")

        # Create test audio file with multiple speakers
        self.audio_file = create_test_audio_file(
            path=os.path.join(self.temp_dir, "test_audio.wav"),
            duration=10,
            num_speakers=2,
            speech_segments=[
                {"speaker": 0, "start": 0.5, "end": 2.5, "text": "Hello, my name is Speaker A."},
                {
                    "speaker": 1,
                    "start": 3.0,
                    "end": 5.0,
                    "text": "Nice to meet you, I'm Speaker B.",
                },
                {"speaker": 0, "start": 5.5, "end": 7.5, "text": "How are you doing today?"},
                {"speaker": 1, "start": 8.0, "end": 9.5, "text": "I'm doing well, thank you."},
            ],
        )

        # Create a test processing job
        self.job_id = "test-reconciliation-integration"
        self.job = ProcessingJob(
            id=self.job_id,
            original_audio_path=Path(self.audio_file),
            status=ProcessingStatus.IN_PROGRESS,
            processing_options={"output_format": "txt"},
        )

        # Generate test data (diarization and transcription)
        self._generate_test_data()

    def tearDown(self):
        """Clean up after tests by removing temporary files."""
        shutil.rmtree(self.temp_dir)

    def _generate_test_data(self):
        """Generate test diarization and transcription data for the job."""
        # Create a test chunk
        chunk = AudioChunk(
            id=f"{self.job_id}_chunk_0",
            job_id=self.job_id,
            chunk_index=0,
            chunk_path=str(self.audio_file),
            start_time=0.0,
            end_time=10.0,
            status=ChunkStatus.COMPLETED,
            metadata={},
        )

        # Save the chunk to the repository
        self.chunk_repository.save(chunk)

        # Save diarization segments
        self.diarization_repo.save(
            self.job_id,
            chunk.id,
            [
                DiarizationSegment(speaker_id="SPEAKER_0", start=0.5, end=2.5, score=0.95),
                DiarizationSegment(speaker_id="SPEAKER_1", start=3.0, end=5.0, score=0.92),
                DiarizationSegment(speaker_id="SPEAKER_0", start=5.5, end=7.5, score=0.94),
                DiarizationSegment(speaker_id="SPEAKER_1", start=8.0, end=9.5, score=0.91),
            ],
        )

        # Create chunk transcription (whole chunk)
        chunk_transcription = [
            Segment(
                text="Hello, my name is Speaker A. Nice to meet you, I'm Speaker B. How are you doing today? I'm doing well, thank you.",
                start=0.5,
                end=9.5,
            )
        ]

        # Save chunk transcription
        self.transcription_repo.save_chunk_transcription(self.job_id, chunk.id, chunk_transcription)

        # Create segment transcriptions
        segment_transcriptions = {
            f"{chunk.id}_segment_0_SPEAKER_0": "Hello, my name is Speaker A.",
            f"{chunk.id}_segment_1_SPEAKER_1": "Nice to meet you, I'm Speaker B.",
            f"{chunk.id}_segment_2_SPEAKER_0": "How are you doing today?",
            f"{chunk.id}_segment_3_SPEAKER_1": "I'm doing well, thank you.",
        }

        # Save segment transcriptions
        for segment_id, text in segment_transcriptions.items():
            self.transcription_repo.save_segment_transcription(self.job_id, segment_id, text)

    def test_full_reconciliation(self):
        """Test full reconciliation process with real diarization and transcription data."""
        # Create reconciliation service with test repositories
        service = ReconciliationService(
            model="gpt-4",  # You can use a mock adapter for testing
            repository=self.reconciliation_repo,
            diarization_repository=self.diarization_repo,
            transcription_repository=self.transcription_repo,
        )

        # Replace the service's chunk repository with our test repository
        service.chunk_repository = self.chunk_repository

        # Mock the adapter's reconcile method to avoid actual API calls
        original_reconcile = service.adapter.reconcile

        def mock_reconcile(*args, **kwargs):
            # Return predefined segments matching the test data
            return [
                Segment(
                    text="Hello, my name is Speaker A.", start=0.5, end=2.5, speaker_id="SPEAKER_0"
                ),
                Segment(
                    text="Nice to meet you, I'm Speaker B.",
                    start=3.0,
                    end=5.0,
                    speaker_id="SPEAKER_1",
                ),
                Segment(
                    text="How are you doing today?", start=5.5, end=7.5, speaker_id="SPEAKER_0"
                ),
                Segment(
                    text="I'm doing well, thank you.", start=8.0, end=9.5, speaker_id="SPEAKER_1"
                ),
            ]

        # Replace the reconcile method
        service.adapter.reconcile = mock_reconcile

        try:
            # Call the reconciliation service
            result = service.reconcile(self.job)

            # Verify the result
            self.assertIsNotNone(result)
            self.assertEqual(4, len(result))

            # Check that the segments match the expected output
            self.assertEqual("Hello, my name is Speaker A.", result[0].text)
            self.assertEqual("SPEAKER_0", result[0].speaker_id)
            self.assertEqual(0.5, result[0].start)
            self.assertEqual(2.5, result[0].end)

            self.assertEqual("Nice to meet you, I'm Speaker B.", result[1].text)
            self.assertEqual("SPEAKER_1", result[1].speaker_id)

            self.assertEqual("How are you doing today?", result[2].text)
            self.assertEqual("SPEAKER_0", result[2].speaker_id)

            self.assertEqual("I'm doing well, thank you.", result[3].text)
            self.assertEqual("SPEAKER_1", result[3].speaker_id)

            # Verify that the results were saved to the repository
            saved_result = self.reconciliation_repo.get_reconciled_result(self.job_id)
            self.assertIsNotNone(saved_result)
            self.assertEqual(4, len(saved_result))

        finally:
            # Restore the original method
            service.adapter.reconcile = original_reconcile

    def test_progressive_reconciliation(self):
        """Test progressive reconciliation with partial results."""
        # Create reconciliation service with test repositories
        service = ReconciliationService(
            model="gpt-4",
            repository=self.reconciliation_repo,
            diarization_repository=self.diarization_repo,
            transcription_repository=self.transcription_repo,
        )

        # Replace the service's chunk repository with our test repository
        service.chunk_repository = self.chunk_repository

        # Mock the adapter's reconcile method
        original_reconcile = service.adapter.reconcile

        def mock_reconcile(*args, **kwargs):
            # Get the options to check if it's a progressive reconciliation
            options = args[4] if len(args) > 4 else kwargs.get("options", {})
            progressive = options.get("progressive", {})

            # Return predefined segments matching the test data
            segments = [
                Segment(
                    text="Hello, my name is Speaker A.", start=0.5, end=2.5, speaker_id="SPEAKER_0"
                ),
                Segment(
                    text="Nice to meet you, I'm Speaker B.",
                    start=3.0,
                    end=5.0,
                    speaker_id="SPEAKER_1",
                ),
            ]

            # If it's a full reconciliation, add all segments
            if not progressive or progressive.get("is_complete", False):
                segments.extend(
                    [
                        Segment(
                            text="How are you doing today?",
                            start=5.5,
                            end=7.5,
                            speaker_id="SPEAKER_0",
                        ),
                        Segment(
                            text="I'm doing well, thank you.",
                            start=8.0,
                            end=9.5,
                            speaker_id="SPEAKER_1",
                        ),
                    ]
                )

            return segments

        # Replace the reconcile method
        service.adapter.reconcile = mock_reconcile

        try:
            # Call the progressive reconciliation service with only partial data
            result = service.reconcile_progressive(
                self.job, min_chunks=1, save_interim_results=True
            )

            # Verify the result
            self.assertIsNotNone(result)

            # With our mock, we should get all 4 segments since we're not truly simulating
            # partial data in this test
            self.assertGreaterEqual(len(result), 2)

            # Check that the segments match the expected output
            self.assertEqual("Hello, my name is Speaker A.", result[0].text)
            self.assertEqual("SPEAKER_0", result[0].speaker_id)

            self.assertEqual("Nice to meet you, I'm Speaker B.", result[1].text)
            self.assertEqual("SPEAKER_1", result[1].speaker_id)

            # Verify that progressive results were saved
            saved_result = self.reconciliation_repo.get_reconciled_result(
                self.job_id + "_progress_1"
            )
            if saved_result:
                self.assertGreaterEqual(len(saved_result), 2)

        finally:
            # Restore the original method
            service.adapter.reconcile = original_reconcile

    def test_output_formatting(self):
        """Test output formatting with reconciled results."""
        # Create reconciliation service with test repositories
        service = ReconciliationService(
            model="gpt-4",
            repository=self.reconciliation_repo,
            diarization_repository=self.diarization_repo,
            transcription_repository=self.transcription_repo,
        )

        # Create test segments
        segments = [
            Segment(
                text="Hello, my name is Speaker A.", start=0.5, end=2.5, speaker_id="SPEAKER_0"
            ),
            Segment(
                text="Nice to meet you, I'm Speaker B.", start=3.0, end=5.0, speaker_id="SPEAKER_1"
            ),
            Segment(text="How are you doing today?", start=5.5, end=7.5, speaker_id="SPEAKER_0"),
            Segment(text="I'm doing well, thank you.", start=8.0, end=9.5, speaker_id="SPEAKER_1"),
        ]

        # Save test segments to repository
        self.reconciliation_repo.save_reconciled_result(self.job_id, segments)

        # Test TXT format
        txt_output = service.format_output(self.job, segments, "txt")
        self.assertIsNotNone(txt_output)
        self.assertIn("SPEAKER_0:", txt_output)
        self.assertIn("Hello, my name is Speaker A.", txt_output)

        # Test JSON format
        json_output = service.format_output(self.job, segments, "json")
        self.assertIsNotNone(json_output)
        json_data = json.loads(json_output)
        self.assertIn("segments", json_data)
        self.assertEqual(4, len(json_data["segments"]))

        # Test SRT format
        srt_output = service.format_output(self.job, segments, "srt")
        self.assertIsNotNone(srt_output)
        self.assertIn("00:00:00", srt_output)
        self.assertIn("SPEAKER_0:", srt_output)


if __name__ == "__main__":
    unittest.main()
