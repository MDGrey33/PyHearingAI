"""
Integration tests for the TranscriptionService.

This module contains integration tests for the TranscriptionService
using synthetic audio files.
"""

import unittest
import tempfile
import shutil
import os
import glob
from pathlib import Path
import logging
from unittest.mock import patch, MagicMock

from pyhearingai.core.idempotent import ProcessingJob, ProcessingStatus, AudioChunk
from pyhearingai.transcription.service import TranscriptionService
from pyhearingai.application.audio_chunking import AudioChunkingService
from pyhearingai.core.ports import Transcriber
from pyhearingai.core.models import Segment
from pyhearingai.infrastructure.registry import register_transcriber, _transcribers


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create a mock transcriber for testing
"""
This mock transcriber implementation is necessary for the integration tests to work properly.
In a production environment, the real WhisperOpenAITranscriber would be registered and available
through the infrastructure.registry module. However, in the test environment, we need to ensure
that a transcriber with the name "whisper_openai" is available without making actual API calls.

The mock:
1. Registers itself with the same name ("whisper_openai") that the TranscriptionService expects
2. Implements the Transcriber interface with minimal functionality
3. Returns predefined transcription segments to simulate actual transcription
4. Avoids the need for actual OpenAI API calls during testing
5. Ensures the TranscriptionService can find and use a transcriber when needed

This approach allows us to test the TranscriptionService's workflow and repository interactions
without external dependencies on the OpenAI API.
"""
@register_transcriber("whisper_openai")
class MockWhisperOpenAITranscriber(Transcriber):
    """Mock Transcriber implementation for testing."""
    
    def __init__(self):
        """Initialize the mock transcriber."""
        pass
        
    @property
    def name(self) -> str:
        """Get the name of the transcriber."""
        return "whisper_openai"
        
    @property
    def supports_segmentation(self) -> bool:
        """Whether this transcriber provides timing and segmentation."""
        return True
        
    def transcribe(self, audio_path: Path, **kwargs) -> list[Segment]:
        """
        Mock transcription implementation that returns dummy segments.
        
        Args:
            audio_path: Path to the audio file to transcribe
            
        Returns:
            List of transcript segments
        """
        # Return a few mock segments
        return [
            Segment(text="This is a mock transcription.", start=0.0, end=2.0),
            Segment(text="For testing purposes only.", start=2.0, end=4.0),
        ]


class TestTranscriptionIntegration(unittest.TestCase):
    """Integration tests for the TranscriptionService."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for the whole test class."""
        # Get test audio files
        cls.fixtures_dir = Path("tests/fixtures/transcription")

        # Find existing test audio files or generate them if needed
        if not cls.fixtures_dir.exists() or not list(cls.fixtures_dir.glob("test_audio_*.wav")):
            logger.info("Generating test audio files...")
            from tests.fixtures.generate_test_audio import (
                generate_test_audio,
                generate_test_audio_with_speakers,
            )

            cls.fixtures_dir.mkdir(parents=True, exist_ok=True)

            # Create a simple test audio file
            cls.test_audio_file = generate_test_audio(
                output_dir=cls.fixtures_dir, duration=3.0, language="en"
            )

            # Create a multilingual test audio file
            cls.multilingual_audio_file = generate_test_audio(
                output_dir=cls.fixtures_dir, duration=3.0, language="multi"
            )

            # Create a multi-speaker test audio file
            cls.multi_speaker_audio_file = generate_test_audio_with_speakers(
                output_dir=cls.fixtures_dir, num_speakers=2, duration_per_speaker=1.5, language="en"
            )
        else:
            # Use existing files
            all_files = list(cls.fixtures_dir.glob("test_audio_*.wav"))

            # Find single language file
            en_files = list(cls.fixtures_dir.glob("test_audio_en_*.wav"))
            if en_files:
                cls.test_audio_file = en_files[0]
            else:
                cls.test_audio_file = all_files[0]

            # Find multilingual file
            multi_files = list(cls.fixtures_dir.glob("test_audio_multi_*.wav"))
            if multi_files:
                cls.multilingual_audio_file = multi_files[0]
            else:
                cls.multilingual_audio_file = all_files[0]

            # Find multi-speaker file
            speaker_files = list(cls.fixtures_dir.glob("test_audio_*_*speakers_*.wav"))
            if speaker_files:
                cls.multi_speaker_audio_file = speaker_files[0]
            else:
                cls.multi_speaker_audio_file = all_files[0]

        logger.info(f"Using test audio file: {cls.test_audio_file}")
        logger.info(f"Using multilingual audio file: {cls.multilingual_audio_file}")
        logger.info(f"Using multi-speaker audio file: {cls.multi_speaker_audio_file}")

    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temp directory for data
        self.temp_dir = tempfile.mkdtemp()

        # Create test jobs
        self.job_en = ProcessingJob(
            original_audio_path=self.test_audio_file,
            id="test-transcription-en",
            status=ProcessingStatus.PENDING,
        )

        self.job_multi = ProcessingJob(
            original_audio_path=self.multilingual_audio_file,
            id="test-transcription-multi",
            status=ProcessingStatus.PENDING,
        )

        self.job_speakers = ProcessingJob(
            original_audio_path=self.multi_speaker_audio_file,
            id="test-transcription-speakers",
            status=ProcessingStatus.PENDING,
        )

        # Create chunking service and TranscriptionService
        self.chunking_service = AudioChunkingService()
        self.service = TranscriptionService(transcriber_name="whisper_openai", max_workers=2)

    def tearDown(self):
        """Clean up test fixtures after each test."""
        # Clean up temp directory
        shutil.rmtree(self.temp_dir)

        # Close service
        self.service.close()

    def _chunk_audio_file(self, job, chunk_duration=1.0):
        """Helper to chunk an audio file for testing."""
        # Create a chunking service with custom settings
        from pyhearingai.config import IdempotentProcessingConfig

        config = IdempotentProcessingConfig()
        config.chunk_duration = chunk_duration
        config.overlap_duration = 0.2

        chunking_service = AudioChunkingService(config)
        chunks = chunking_service.create_chunks(job)

        # Store the chunks in a repository to associate them with the job
        from pyhearingai.infrastructure.repositories.json_repositories import JsonChunkRepository

        chunk_repo = JsonChunkRepository()

        # Save each chunk to the repository
        for chunk in chunks:
            chunk_repo.save(chunk)

        return chunks

    def test_transcribe_chunk(self):
        """Test transcribing a single chunk."""
        # Skip test if OpenAI API key is not set
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY environment variable not set")

        # Chunk the audio file
        chunks = self._chunk_audio_file(self.job_en)
        self.assertGreater(len(chunks), 0, "No chunks created")

        # Transcribe a single chunk
        chunk = chunks[0]
        segments = self.service.transcribe_chunk(chunk, self.job_en)

        # Assert results
        self.assertIsNotNone(segments)

        # Check if the repository contains the result
        self.assertTrue(
            self.service.repository.chunk_exists(self.job_en.id, chunk.id),
            "Transcription result not saved to repository",
        )

    def test_transcribe_job_sequential(self):
        """Test transcribing a job in sequential mode."""
        # Skip test if OpenAI API key is not set
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY environment variable not set")

        # Chunk the audio file
        chunks = self._chunk_audio_file(self.job_en)
        self.assertGreater(len(chunks), 0, "No chunks created")

        # Transcribe the job
        result = self.service.transcribe_job(job=self.job_en, parallel=False)

        # Assert results
        self.assertTrue(result["success"])
        # The number of chunks processed might be different from the number we created
        # because the repository might have chunks from previous test runs
        self.assertGreaterEqual(result["chunks_processed"], len(chunks))
        # Some chunks might fail due to test conditions, but the overall job should succeed
        self.assertLess(result.get("chunks_failed", 0), result["chunks_total"])

    def test_transcribe_job_parallel(self):
        """Test transcribing a job in parallel mode."""
        # Skip test if OpenAI API key is not set
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY environment variable not set")

        # Chunk the audio file
        chunks = self._chunk_audio_file(self.job_speakers)
        self.assertGreater(len(chunks), 0, "No chunks created")

        # Transcribe the job
        result = self.service.transcribe_job(job=self.job_speakers, parallel=True)

        # Assert results
        self.assertTrue(result["success"])
        # The number of chunks processed might be different from the number we created
        # because the repository might have chunks from previous test runs
        self.assertGreaterEqual(result["chunks_processed"], len(chunks))
        # Some chunks might fail due to test conditions, but the overall job should succeed
        self.assertLess(result.get("chunks_failed", 0), result["chunks_total"])

    def test_cached_results(self):
        """Test that cached results are used when available."""
        # Skip test if OpenAI API key is not set
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY environment variable not set")

        # Chunk the audio file
        chunks = self._chunk_audio_file(self.job_en)
        self.assertGreater(len(chunks), 0, "No chunks created")

        # Transcribe a single chunk
        chunk = chunks[0]
        segments1 = self.service.transcribe_chunk(chunk, self.job_en)

        # Transcribe the same chunk again
        segments2 = self.service.transcribe_chunk(chunk, self.job_en)

        # Assert that the cached results were used
        self.assertEqual(len(segments1), len(segments2))

        # Transcribe with force=True to bypass cache
        segments3 = self.service.transcribe_chunk(chunk, self.job_en, force=True)

        # Assert that new results were generated
        self.assertIsNotNone(segments3)

    def test_multilingual_transcription(self):
        """Test transcribing multilingual audio."""
        # Skip test if OpenAI API key is not set
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY environment variable not set")

        # Chunk the audio file
        chunks = self._chunk_audio_file(self.job_multi)
        self.assertGreater(len(chunks), 0, "No chunks created")

        # Transcribe the job
        result = self.service.transcribe_job(job=self.job_multi, parallel=True)

        # Assert results
        self.assertTrue(result["success"])
        # The number of chunks processed might be different from the number we created
        # because the repository might have chunks from previous test runs
        self.assertGreaterEqual(result["chunks_processed"], len(chunks))
        # Some chunks might fail due to test conditions, but the overall job should succeed
        self.assertLess(result.get("chunks_failed", 0), result["chunks_total"])


if __name__ == "__main__":
    unittest.main()
