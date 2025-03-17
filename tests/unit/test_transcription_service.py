#!/usr/bin/env python3
"""
Test script for the TranscriptionService.

This script demonstrates how to use the TranscriptionService to transcribe
an audio file with parallel processing.
"""

import argparse
import logging
import sys
from pathlib import Path

from pyhearingai.core.idempotent import ProcessingJob, ProcessingStatus
from pyhearingai.application.audio_chunking import AudioChunkingService
from pyhearingai.transcription.service import TranscriptionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    """Run the test script."""
    parser = argparse.ArgumentParser(description="Test the TranscriptionService")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create a processing job
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 1

    logger.info(f"Testing transcription service with audio file: {audio_path}")

    # Create a job
    job = ProcessingJob(
        original_audio_path=audio_path,
        id=f"test-transcription-{audio_path.stem}",
        status=ProcessingStatus.PENDING,
    )

    # Chunk the audio file
    logger.info(f"Chunking audio file: {audio_path}")
    chunking_service = AudioChunkingService()
    chunks = chunking_service.create_chunks(job)

    logger.info(f"Created {len(chunks)} chunks")

    # Initialize the transcription service
    service = TranscriptionService(transcriber_name="whisper_openai", max_workers=args.workers)

    # Transcribe the job
    logger.info(f"Transcribing job with parallel={args.parallel}, workers={args.workers}")
    result = service.transcribe_job(job=job, parallel=args.parallel, chunk_batch_size=5)

    # Print results
    logger.info(f"Transcription completed: {result['success']}")
    logger.info(f"Chunks processed: {result['chunks_processed']}/{result['chunks_total']}")
    logger.info(f"Duration: {result['duration']:.2f}s")

    # Clean up
    service.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
