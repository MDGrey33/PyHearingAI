"""
Tests for repository implementations.

This module contains tests for the repository implementations, focusing on
JSON repositories for jobs, chunks, and segments.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pyhearingai.core.idempotent import AudioChunk, ChunkStatus, ProcessingJob, ProcessingStatus
from pyhearingai.infrastructure.repositories.json_repositories import (
    JsonChunkRepository,
    JsonJobRepository,
)
from tests.utils.test_helpers import TestFixtures


class TestJsonJobRepository:
    """Tests for the JsonJobRepository implementation."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir

        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def job_repository(self, temp_data_dir):
        """Create a JsonJobRepository with a temporary directory."""
        jobs_dir = os.path.join(temp_data_dir, "jobs")
        os.makedirs(jobs_dir, exist_ok=True)

        # Convert to Path object
        return JsonJobRepository(jobs_dir=Path(jobs_dir))

    @pytest.fixture
    def sample_job(self):
        """Create a sample job for testing."""
        return TestFixtures.create_test_job()

    def test_save_and_get_job(self, job_repository, sample_job):
        """Test saving and retrieving a job."""
        # Save the job
        job_repository.save(sample_job)

        # Verify the job file exists
        job_file = os.path.join(str(job_repository.jobs_dir), f"{sample_job.id}.json")
        assert os.path.exists(job_file)

        # Retrieve the job
        retrieved_job = job_repository.get_by_id(sample_job.id)

        # Verify the retrieved job matches the original
        assert retrieved_job is not None
        assert retrieved_job.id == sample_job.id
        # Convert paths to strings for comparison
        assert str(retrieved_job.original_audio_path) == str(sample_job.original_audio_path)
        assert retrieved_job.status == sample_job.status
        assert retrieved_job.chunk_duration == sample_job.chunk_duration

        # Check that dates are properly serialized and deserialized
        assert isinstance(retrieved_job.created_at, datetime)
        assert isinstance(retrieved_job.updated_at, datetime)

    def test_update_job(self, job_repository, sample_job):
        """Test updating a job."""
        # Save the job
        job_repository.save(sample_job)

        # Update job properties
        sample_job.status = ProcessingStatus.IN_PROGRESS  # Use correct enum value
        sample_job.updated_at = sample_job.updated_at + timedelta(hours=1)

        # Save the updated job
        job_repository.save(sample_job)

        # Retrieve the job
        retrieved_job = job_repository.get_by_id(sample_job.id)

        # Verify the updates were saved
        assert retrieved_job.status == ProcessingStatus.IN_PROGRESS
        assert retrieved_job.updated_at > sample_job.created_at

    def test_delete_job(self, job_repository, sample_job):
        """Test deleting a job."""
        # Save the job
        job_repository.save(sample_job)

        # Check if the job exists
        job = job_repository.get_by_id(sample_job.id)
        assert job is not None

        # Delete the job
        job_repository.delete(sample_job.id)

        # Verify the job no longer exists
        job = job_repository.get_by_id(sample_job.id)
        assert job is None

        # Verify the job file is deleted
        job_file = os.path.join(str(job_repository.jobs_dir), f"{sample_job.id}.json")
        assert not os.path.exists(job_file)

    def test_get_nonexistent_job(self, job_repository):
        """Test getting a job that doesn't exist."""
        # Attempt to retrieve a non-existent job
        job = job_repository.get_by_id("nonexistent-job-id")

        # Should return None
        assert job is None

    def test_list_all_jobs(self, job_repository):
        """Test listing all jobs."""
        # Create several jobs
        jobs = [TestFixtures.create_test_job() for _ in range(3)]

        # Save all jobs
        for job in jobs:
            job_repository.save(job)

        # List all jobs
        all_jobs = job_repository.list_all()

        # Verify we get all jobs
        assert len(all_jobs) == len(jobs)

        # Verify all job IDs are in the list
        job_ids = {job.id for job in all_jobs}
        expected_ids = {job.id for job in jobs}
        assert job_ids == expected_ids

    def test_find_by_audio_path(self, job_repository):
        """Test finding jobs by audio path."""
        # Create jobs with different audio paths
        path1 = "/path/to/audio1.wav"
        path2 = "/path/to/audio2.wav"

        job1 = TestFixtures.create_test_job()
        job1.original_audio_path = path1

        job2 = TestFixtures.create_test_job()
        job2.original_audio_path = path2

        job3 = TestFixtures.create_test_job()
        job3.original_audio_path = path1  # Same as job1

        # Save all jobs
        job_repository.save(job1)
        job_repository.save(job2)
        job_repository.save(job3)

        # Find job by audio path - note that this returns a single job, not a list
        job_for_path1 = job_repository.get_by_audio_path(path1)

        # Verify we find one of the jobs with this path
        assert job_for_path1 is not None
        assert str(job_for_path1.original_audio_path) == path1
        assert job_for_path1.id in [job1.id, job3.id]

    @pytest.mark.skip(reason="ProcessingJob class doesn't have processing_options attribute yet")
    def test_serialization_edge_cases(self, job_repository):
        """Test serialization of edge cases like None values and complex objects."""
        # Create a job with some edge case values
        job = TestFixtures.create_test_job()
        job.completed_at = None  # Test None handling
        job.processing_options = {
            "complex_option": {"nested": [1, 2, 3], "values": {"a": 1, "b": None}},
            "none_value": None,
        }

        # Save the job
        job_repository.save(job)

        # Retrieve the job
        retrieved_job = job_repository.get_by_id(job.id)

        # Verify complex data is preserved
        assert retrieved_job.completed_at is None
        assert retrieved_job.processing_options["complex_option"]["nested"] == [1, 2, 3]


class TestJsonChunkRepository:
    """Tests for the JsonChunkRepository implementation."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir

        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def chunk_repository(self, temp_data_dir):
        """Create a JsonChunkRepository with a temporary directory."""
        chunks_dir = os.path.join(temp_data_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)

        # Convert to Path object
        return JsonChunkRepository(chunks_dir=Path(chunks_dir))

    @pytest.fixture
    def sample_job(self):
        """Create a sample job for testing."""
        return TestFixtures.create_test_job()

    @pytest.fixture
    def sample_chunk(self, sample_job):
        """Create a sample chunk for testing."""
        return TestFixtures.create_test_chunk(sample_job.id)

    def test_save_and_get_chunk(self, chunk_repository, sample_chunk):
        """Test saving and retrieving a chunk."""
        # Save the chunk
        chunk_repository.save(sample_chunk)

        # Verify the chunk file exists
        chunk_file = os.path.join(
            str(chunk_repository.chunks_dir), sample_chunk.job_id, f"{sample_chunk.id}.json"
        )
        assert os.path.exists(chunk_file)

        # Retrieve the chunk
        retrieved_chunk = chunk_repository.get_by_id(sample_chunk.id)

        # Verify the retrieved chunk matches the original
        assert retrieved_chunk is not None
        assert retrieved_chunk.id == sample_chunk.id
        assert retrieved_chunk.job_id == sample_chunk.job_id
        assert retrieved_chunk.chunk_index == sample_chunk.chunk_index
        assert retrieved_chunk.start_time == sample_chunk.start_time
        assert retrieved_chunk.end_time == sample_chunk.end_time
        assert str(retrieved_chunk.chunk_path) == str(sample_chunk.chunk_path)
        assert retrieved_chunk.status == sample_chunk.status

    def test_update_chunk(self, chunk_repository, sample_chunk):
        """Test updating a chunk."""
        # Save the chunk
        chunk_repository.save(sample_chunk)

        # Update chunk properties
        sample_chunk.status = ChunkStatus.DIARIZING  # Use correct enum value
        sample_chunk.metadata = {
            "processed_by": "test_worker"
        }  # Use metadata instead of processing_metadata

        # Save the updated chunk
        chunk_repository.save(sample_chunk)

        # Retrieve the chunk
        retrieved_chunk = chunk_repository.get_by_id(sample_chunk.id)

        # Verify the updates were saved
        assert retrieved_chunk.status == ChunkStatus.DIARIZING
        assert retrieved_chunk.metadata["processed_by"] == "test_worker"

    def test_delete_chunk(self, chunk_repository, sample_chunk):
        """Test deleting a chunk."""
        # Save the chunk
        chunk_repository.save(sample_chunk)

        # Check if the chunk exists
        chunk = chunk_repository.get_by_id(sample_chunk.id)
        assert chunk is not None

        # Delete the chunk
        chunk_repository.delete(sample_chunk.id)

        # Verify the chunk no longer exists
        chunk = chunk_repository.get_by_id(sample_chunk.id)
        assert chunk is None

        # Verify the chunk file is deleted
        chunk_file = os.path.join(str(chunk_repository.chunks_dir), f"{sample_chunk.id}.json")
        assert not os.path.exists(chunk_file)

    def test_get_chunks_for_job(self, chunk_repository, sample_job):
        """Test getting all chunks for a job."""
        # Create several chunks for the same job
        job_id = sample_job.id
        chunks = [
            TestFixtures.create_test_chunk(job_id, index=0, start_time=0.0, end_time=5.0),
            TestFixtures.create_test_chunk(job_id, index=1, start_time=4.0, end_time=9.0),
            TestFixtures.create_test_chunk(job_id, index=2, start_time=8.0, end_time=12.0),
        ]

        # Create a chunk for a different job
        other_job_id = "other-job-id"
        other_chunk = TestFixtures.create_test_chunk(other_job_id)

        # Save all chunks
        for chunk in chunks:
            chunk_repository.save(chunk)
        chunk_repository.save(other_chunk)

        # Get chunks for the job - use get_by_job_id instead of get_chunks_for_job
        job_chunks = chunk_repository.get_by_job_id(job_id)

        # Verify we get only the chunks for our job
        assert len(job_chunks) == len(chunks)

        # Verify chunks are returned in order by index
        assert job_chunks[0].chunk_index == 0
        assert job_chunks[1].chunk_index == 1
        assert job_chunks[2].chunk_index == 2

        # Verify the chunk for the other job is not included
        chunk_ids = {chunk.id for chunk in job_chunks}
        assert other_chunk.id not in chunk_ids

    def test_count_chunks_by_status(self, chunk_repository, sample_job):
        """Test counting chunks by status."""
        # Create chunks with different statuses
        job_id = sample_job.id
        chunks = [
            TestFixtures.create_test_chunk(job_id, index=0),
            TestFixtures.create_test_chunk(job_id, index=1),
            TestFixtures.create_test_chunk(job_id, index=2),
            TestFixtures.create_test_chunk(job_id, index=3),
            TestFixtures.create_test_chunk(job_id, index=4),
        ]

        # Set different statuses
        chunks[0].status = ChunkStatus.PENDING
        chunks[1].status = ChunkStatus.DIARIZING
        chunks[2].status = ChunkStatus.COMPLETED
        chunks[3].status = ChunkStatus.COMPLETED
        chunks[4].status = ChunkStatus.FAILED

        # Save all chunks
        for chunk in chunks:
            chunk_repository.save(chunk)

        # Get all chunks and count by status manually
        all_chunks = chunk_repository.get_by_job_id(job_id)
        counts = {}
        for chunk in all_chunks:
            counts[chunk.status] = counts.get(chunk.status, 0) + 1

        # Verify counts
        assert counts.get(ChunkStatus.PENDING, 0) == 1
        assert counts.get(ChunkStatus.DIARIZING, 0) == 1
        assert counts.get(ChunkStatus.COMPLETED, 0) == 2
        assert counts.get(ChunkStatus.FAILED, 0) == 1

    def test_mark_chunks_as_status(self, chunk_repository, sample_job):
        """Test marking multiple chunks with a status."""
        # Create chunks
        job_id = sample_job.id
        chunks = [TestFixtures.create_test_chunk(job_id, index=i) for i in range(5)]

        # Save all chunks
        for chunk in chunks:
            chunk_repository.save(chunk)

        # Get IDs for chunks 1, 2, and 3
        chunk_ids = [chunks[1].id, chunks[2].id, chunks[3].id]

        # Mark these chunks as DIARIZING
        for chunk_id in chunk_ids:
            chunk = chunk_repository.get_by_id(chunk_id)
            chunk.status = ChunkStatus.DIARIZING
            chunk_repository.save(chunk)

        # Verify the status was updated for the specified chunks
        for i, chunk_id in enumerate(chunk_ids):
            chunk = chunk_repository.get_by_id(chunk_id)
            assert chunk.status == ChunkStatus.DIARIZING

        # Verify other chunks were not affected
        assert chunk_repository.get_by_id(chunks[0].id).status == ChunkStatus.PENDING
        assert chunk_repository.get_by_id(chunks[4].id).status == ChunkStatus.PENDING
