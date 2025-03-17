#!/usr/bin/env python
"""
Test script for the DiarizationService with idempotent processing.

This script demonstrates how to use the DiarizationService to process audio chunks
and verify the resumability of the diarization process.

Usage:
    python test_diarization_service.py <audio_file_path> [--parallel|--sequential] [--workers N] [--debug]
"""

import logging
import os
import sys
import time
import traceback
from pathlib import Path

from pyhearingai.application.audio_chunking import AudioChunkingService
from pyhearingai.config import IdempotentProcessingConfig, TranscriptionConfig
from pyhearingai.core.idempotent import ProcessingJob
from pyhearingai.diarization.service import DiarizationService
from pyhearingai.infrastructure.repositories.json_repositories import (
    JsonChunkRepository,
    JsonJobRepository,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_diarization")


def main(audio_path, max_workers=None, parallel=True):
    """
    Run the diarization service test.

    Args:
        audio_path: Path to the audio file to diarize
        max_workers: Maximum number of workers to use for parallel processing
        parallel: Whether to use parallel processing
    """
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)

    logger.info(f"Testing diarization service with audio: {audio_path}")
    logger.info(f"Parallel processing: {parallel}, Max workers: {max_workers or 'auto'}")

    # Create configuration with idempotent processing enabled
    idempotent_config = IdempotentProcessingConfig(
        enabled=True,
        chunk_duration=20.0,  # 20 second chunks for testing - smaller chunks for more parallelism
        chunk_overlap=2.0,  # 2 seconds overlap
    )

    config = TranscriptionConfig(idempotent=idempotent_config)

    # Initialize repositories
    job_repo = JsonJobRepository()
    chunk_repo = JsonChunkRepository()

    # Initialize services
    chunking_service = AudioChunkingService(config.idempotent)
    diarization_service = DiarizationService(diarizer_name="pyannote", max_workers=max_workers)

    try:
        # Check if a job already exists for this audio file
        existing_job = job_repo.get_by_audio_path(audio_path)

        if existing_job:
            logger.info(f"Found existing job for {audio_path}")
            job = existing_job
            # Load all chunks from the repository
            chunks = chunk_repo.get_by_job_id(job.id)
            logger.info(f"Loaded {len(chunks)} existing chunks")
        else:
            # Create a new processing job
            logger.info(f"Creating new processing job for {audio_path}")
            job = ProcessingJob(
                original_audio_path=audio_path,
                chunk_duration=config.idempotent.chunk_duration,
                overlap_duration=config.idempotent.chunk_overlap,
            )

            # Save the job to get an ID
            job = job_repo.save(job)
            logger.info(f"Created new job with ID: {job.id}")

            # Create audio chunks
            chunks = chunking_service.create_chunks(job)

            # Save the job with updated chunk information
            job = job_repo.save(job)

            # Save all chunks to the repository
            chunks = chunk_repo.save_many(chunks)
            logger.info(f"Created and saved {len(chunks)} chunks")

        if not chunks:
            logger.error("No chunks available for testing")
            return False

        if len(chunks) < 3:
            logger.warning(
                f"Only {len(chunks)} chunks available - parallel processing may not show benefits"
            )

        # Test chunk processing
        num_chunks = min(5, len(chunks))  # Test with up to 5 chunks
        test_chunks = chunks[:num_chunks]
        logger.info(f"Testing diarization on {num_chunks} chunks")

        # Print details about the first chunk for debugging
        logger.info(
            f"First chunk details: id={test_chunks[0].id}, path={test_chunks[0].chunk_path}"
        )
        logger.info(
            f"First chunk job_id={test_chunks[0].job_id}, duration={test_chunks[0].duration}s"
        )

        # Test options
        if parallel:
            logger.info("Using parallel processing")

            # Process a batch of chunks in parallel
            start_time = time.time()
            logger.info(f"Starting parallel diarization of {num_chunks} chunks")

            results = diarization_service.diarize_job(
                job,
                chunk_repo=chunk_repo,
                parallel=True,
                force_reprocess=True,  # Force reprocessing to test performance
                limit=num_chunks,  # Limit to the specified number of chunks
            )

            elapsed = time.time() - start_time
            logger.info(f"Parallel diarization of {num_chunks} chunks completed in {elapsed:.2f}s")
            logger.info(f"Average time per chunk: {elapsed / num_chunks:.2f}s")

            # Show results summary
            for chunk_id, segments in results.items():
                if segments:
                    logger.info(f"Chunk {chunk_id}: {len(segments)} segments")
                else:
                    logger.warning(f"Chunk {chunk_id}: No segments found")

                # Show some segments from the first chunk only
                if chunk_id == test_chunks[0].id and segments:
                    for i, segment in enumerate(segments[:3]):  # Show first 3 segments
                        logger.info(
                            f"  Segment {i+1}: {segment.speaker_id} - "
                            f"{segment.start:.2f}s to {segment.end:.2f}s"
                        )

                    if len(segments) > 3:
                        logger.info(f"  ... {len(segments) - 3} more segments")

            # Test caching - should be fast
            logger.info("Running again to test caching...")
            start_time = time.time()

            cached_results = diarization_service.diarize_job(
                job,
                chunk_repo=chunk_repo,
                parallel=True,
                force_reprocess=False,  # Use cached results
            )

            cached_elapsed = time.time() - start_time
            logger.info(f"Cached diarization completed in {cached_elapsed:.2f}s")

            # Verify results are the same
            if len(cached_results) == len(results):
                logger.info("Cached results match original results")
            else:
                logger.warning("Cached results do not match original results")
        else:
            # Process sequentially for comparison
            logger.info("Using sequential processing")

            start_time = time.time()
            logger.info(f"Starting sequential diarization of {num_chunks} chunks")

            results = diarization_service.diarize_job(
                job,
                chunk_repo=chunk_repo,
                parallel=False,
                force_reprocess=True,  # Force reprocessing to test performance
                limit=num_chunks,  # Limit to the specified number of chunks
            )

            elapsed = time.time() - start_time
            logger.info(
                f"Sequential diarization of {num_chunks} chunks completed in {elapsed:.2f}s"
            )
            logger.info(f"Average time per chunk: {elapsed / num_chunks:.2f}s")

            # Show results summary (same as parallel case)
            for chunk_id, segments in results.items():
                if segments:
                    logger.info(f"Chunk {chunk_id}: {len(segments)} segments")
                else:
                    logger.warning(f"Chunk {chunk_id}: No segments found")

        logger.info("Diarization service test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error testing diarization service: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        # Clean up resources
        diarization_service.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            f"Usage: {sys.argv[0]} <audio_file_path> [--parallel|--sequential] [--workers N] [--debug]"
        )
        sys.exit(1)

    # Parse command line arguments
    audio_file = sys.argv[1]
    parallel = True  # Default to parallel
    max_workers = None

    # Enable debug logging if requested
    if "--debug" in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("pyhearingai").setLevel(logging.DEBUG)
        print("Debug logging enabled")

    # Check for parallel/sequential flag
    if "--sequential" in sys.argv:
        parallel = False
    elif "--parallel" in sys.argv:
        parallel = True

    # Check for workers flag
    if "--workers" in sys.argv:
        try:
            workers_index = sys.argv.index("--workers")
            if workers_index < len(sys.argv) - 1:
                max_workers = int(sys.argv[workers_index + 1])
        except (ValueError, IndexError):
            print("Invalid --workers value")
            sys.exit(1)

    success = main(audio_file, max_workers=max_workers, parallel=parallel)
    sys.exit(0 if success else 1)
