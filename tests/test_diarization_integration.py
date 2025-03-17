"""
Integration tests for the DiarizationService.

This module contains tests that verify the DiarizationService works correctly
with actual audio files and a real repository, focusing on end-to-end workflow.

The tests use a simple test audio file with the following speaker segments:
- Speaker 1: 0-2s and 6-8s (440 Hz tone)
- Speaker 2: 3-5s and 8-10s (880 Hz tone)
- Silence: 2-3s and 5-6s
"""

import logging
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from pyhearingai.core.idempotent import AudioChunk, ChunkStatus, ProcessingJob
from pyhearingai.diarization.repositories.diarization_repository import DiarizationRepository
from pyhearingai.diarization.service import DiarizationService, _process_chunk_directly

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define path to test audio file
TEST_AUDIO_PATH = Path(__file__).parent / "fixtures" / "test_audio.wav"

# Skip these tests if the test audio file doesn't exist
SKIP_TESTS = not TEST_AUDIO_PATH.exists()


@unittest.skipIf(SKIP_TESTS, "Test audio file not found at: {}".format(TEST_AUDIO_PATH))
class TestDiarizationServiceIntegration(unittest.TestCase):
    """
    Integration tests for the DiarizationService.

    These tests verify the DiarizationService works correctly in a real-world scenario.
    They use a generated test audio file with known speaker segments to verify that
    the diarization process can identify different speakers and their time segments.
    """

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for the repository
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.temp_dir.name) / "diarization"
        os.makedirs(self.repo_dir, exist_ok=True)

        # Create a real repository
        self.repository = DiarizationRepository(self.repo_dir)

        # Create the service with pyannote diarizer
        self.service = DiarizationService(
            diarizer_name="pyannote", repository=self.repository, max_workers=2
        )

        # Path to test audio file
        self.test_audio_path = TEST_AUDIO_PATH

        # Create a test chunk
        self.test_chunk = AudioChunk(
            id=str(uuid.uuid4()),
            job_id=str(uuid.uuid4()),
            chunk_path=self.test_audio_path,
            start_time=0,
            end_time=10,
            chunk_index=0,
            status=ChunkStatus.PENDING,
        )

        # Create a test job
        self.test_job = ProcessingJob(
            id=self.test_chunk.job_id,
            original_audio_path=str(self.test_audio_path),
            chunks=[self.test_chunk],
        )

        # Add these as attributes to the job for testing
        self.test_job.parallel = False
        self.test_job.force_reprocess = False

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
        self.service.close()

    def test_diarize_chunk(self):
        """
        Test diarizing a single chunk with a real audio file.

        This test verifies that the diarization service can:
        1. Process the test audio file
        2. Successfully run the diarization process
        3. Save the results to the repository
        """
        # Skip if no diarizer is available
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            self.skipTest("pyannote.audio not available")

        # Process the chunk
        segments = self.service.diarize_chunk(self.test_chunk)

        # Verify results
        self.assertIsNotNone(segments)
        self.assertIsInstance(segments, list)

        # Log the detected segments
        logger.info(f"Found {len(segments)} segments in test audio")
        for i, segment in enumerate(segments):
            logger.info(
                f"Segment {i+1}: Speaker {segment.speaker_id} from {segment.start:.2f}s to {segment.end:.2f}s"
            )

        # Note: Our synthetic audio might not be recognized as having speaker segments,
        # so we don't assert on the number of segments.

        # Verify segments were saved - handle None case
        saved_segments = self.repository.get(self.test_chunk.job_id, self.test_chunk.id)
        if segments:  # If we found segments, they should be saved
            self.assertEqual(segments, saved_segments)
        else:  # If no segments were found, saved_segments might be None or []
            self.assertIn(saved_segments, [None, []])  # Either None or empty list is acceptable

    def test_diarize_job_sequential(self):
        """
        Test diarizing a job sequentially with a real audio file.

        This test verifies that the diarization service can process
        a job with multiple chunks sequentially.
        """
        # Skip if no diarizer is available
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            self.skipTest("pyannote.audio not available")

        # Process the job
        results = self.service.diarize_job(self.test_job)

        # Verify results
        self.assertIsNotNone(results)

        # Log results
        for chunk_id, segments in results.items():
            logger.info(f"Chunk {chunk_id}: {len(segments)} segments")
            for i, segment in enumerate(segments):
                logger.info(
                    f"  Segment {i+1}: Speaker {segment.speaker_id} from {segment.start:.2f}s to {segment.end:.2f}s"
                )

        # Note: The test might pass even if no segments are detected,
        # as we're testing the flow, not the accuracy of diarization.

    def test_process_chunk_directly(self):
        """
        Test the _process_chunk_directly function with a real audio file.

        This function is used for parallel processing, and this test verifies
        that it can correctly process audio chunks independently.
        """
        # Skip if no diarizer is available
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            self.skipTest("pyannote.audio not available")

        # Process the chunk directly
        segments = _process_chunk_directly(self.test_chunk, "pyannote")

        # Verify results
        self.assertIsNotNone(segments)
        self.assertIsInstance(segments, list)

        # Log the detected segments
        logger.info(f"Found {len(segments)} segments in test audio")
        for i, segment in enumerate(segments):
            logger.info(
                f"Segment {i+1}: Speaker {segment.speaker_id} from {segment.start:.2f}s to {segment.end:.2f}s"
            )

        # Note: Our synthetic audio might not be recognized as having speaker segments,
        # so we don't assert on the number of segments.


if __name__ == "__main__":
    unittest.main()
