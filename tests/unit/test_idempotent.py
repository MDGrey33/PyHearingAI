#!/usr/bin/env python
"""
Test script for idempotent processing in PyHearingAI.

This script demonstrates how to use the core idempotent processing components
that were implemented in Sprint 1.

Usage:
    python test_idempotent.py <audio_file_path>
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np

from pyhearingai.config import TranscriptionConfig, IdempotentProcessingConfig
from pyhearingai.core.idempotent import ProcessingJob
from pyhearingai.application.audio_chunking import AudioChunkingService
from pyhearingai.infrastructure.repositories.json_repositories import (
    JsonJobRepository,
    JsonChunkRepository,
    JsonSegmentRepository,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_idempotent")


def main(audio_path):
    """Run the idempotent processing test."""
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)

    logger.info(f"Testing idempotent processing with audio: {audio_path}")

    # Create configuration with idempotent processing enabled
    idempotent_config = IdempotentProcessingConfig(
        enabled=True,
        chunk_duration=60.0,  # 1 minute chunks for testing
        chunk_overlap=5.0,  # 5 seconds overlap
    )

    config = TranscriptionConfig(idempotent=idempotent_config)

    # Initialize repositories
    job_repo = JsonJobRepository()
    chunk_repo = JsonChunkRepository()
    segment_repo = JsonSegmentRepository()

    # Initialize audio chunking service
    chunking_service = AudioChunkingService(config.idempotent)

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

    # Print job and chunk information
    logger.info(f"Job ID: {job.id}")
    logger.info(f"Total chunks: {job.total_chunks}")
    logger.info(f"Job status: {job.status.name}")

    # Print chunk details
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1}/{len(chunks)}: {chunk.id}")
        logger.info(
            f"  Time range: {chunk.start_time:.2f}s - {chunk.end_time:.2f}s (duration: {chunk.duration:.2f}s)"
        )
        logger.info(f"  Status: {chunk.status.name}")
        logger.info(f"  Audio file: {chunk.chunk_path}")

    logger.info("Idempotent processing test completed successfully!")

    # Cleanup: The chunks and metadata files are saved to disk and can be examined
    # Return the job ID for reference
    return job.id


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_file_path>")
        sys.exit(1)

    main(sys.argv[1])
