"""
Tests for the idempotent processing domain entities and repositories.

This module tests the core domain entities and repository implementations
that enable idempotent processing of audio files.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from pyhearingai.core.idempotent import (
    AudioChunk,
    ChunkStatus,
    ProcessingJob,
    ProcessingStatus,
    SpeakerSegment,
)
from pyhearingai.infrastructure.repositories.json_repositories import (
    JsonChunkRepository,
    JsonJobRepository,
    JsonSegmentRepository,
)


class TestProcessingStatus:
    """Tests for the ProcessingStatus enum."""

    def test_status_progression(self):
        """Test the progression of status values."""
        # Test that statuses are defined in a logical order
        assert ProcessingStatus.PENDING.value < ProcessingStatus.IN_PROGRESS.value
        assert ProcessingStatus.IN_PROGRESS.value < ProcessingStatus.DIARIZING.value
        assert ProcessingStatus.DIARIZING.value < ProcessingStatus.DIARIZED.value
        assert ProcessingStatus.DIARIZED.value < ProcessingStatus.TRANSCRIBING.value
        assert ProcessingStatus.TRANSCRIBING.value < ProcessingStatus.TRANSCRIBED.value
        assert ProcessingStatus.TRANSCRIBED.value < ProcessingStatus.RECONCILING.value
        assert ProcessingStatus.RECONCILING.value < ProcessingStatus.COMPLETED.value


class TestChunkStatus:
    """Tests for the ChunkStatus enum."""

    def test_status_progression(self):
        """Test the progression of status values."""
        # Test that statuses are defined in a logical order
        assert ChunkStatus.PENDING.value < ChunkStatus.DIARIZING.value
        assert ChunkStatus.DIARIZING.value < ChunkStatus.DIARIZED.value
        assert ChunkStatus.DIARIZED.value < ChunkStatus.TRANSCRIBING.value
        assert ChunkStatus.TRANSCRIBING.value < ChunkStatus.TRANSCRIBED.value
        assert ChunkStatus.TRANSCRIBED.value < ChunkStatus.COMPLETED.value


class TestAudioChunk:
    """Tests for the AudioChunk entity."""

    def test_create_audio_chunk(self):
        """Test creating an AudioChunk entity."""
        # Arrange
        job_id = "test-job-123"
        chunk_path = Path("/tmp/test-chunk.wav")

        # Act
        chunk = AudioChunk(
            job_id=job_id,
            chunk_path=chunk_path,
            start_time=0.0,
            end_time=5.0,
            chunk_index=1,
        )

        # Assert
        assert chunk.id is not None
        assert chunk.job_id == job_id
        assert chunk.chunk_path == chunk_path
        assert chunk.start_time == 0.0
        assert chunk.end_time == 5.0
        assert chunk.chunk_index == 1
        assert chunk.status == ChunkStatus.PENDING
        assert chunk.diarization_segments == []
        assert chunk.transcription_segments == []
        assert chunk.created_at is not None
        assert chunk.updated_at is not None
        assert chunk.metadata == {}

    def test_audio_chunk_properties(self):
        """Test the properties of the AudioChunk entity."""
        # Arrange
        chunk = AudioChunk(
            job_id="test-job-123",
            chunk_path=Path("/tmp/test-chunk.wav"),
            start_time=10.0,
            end_time=15.0,
            chunk_index=2,
        )

        # Act & Assert
        assert chunk.duration == 5.0
        assert not chunk.is_processed
        assert not chunk.is_diarized
        assert not chunk.is_transcribed

        # Change status and check properties
        chunk.status = ChunkStatus.DIARIZED
        assert not chunk.is_processed
        assert chunk.is_diarized
        assert not chunk.is_transcribed

        chunk.status = ChunkStatus.TRANSCRIBED
        assert not chunk.is_processed
        assert chunk.is_diarized
        assert chunk.is_transcribed

        chunk.status = ChunkStatus.COMPLETED
        assert chunk.is_processed
        assert chunk.is_diarized
        assert chunk.is_transcribed


class TestSpeakerSegment:
    """Tests for the SpeakerSegment entity."""

    def test_create_speaker_segment(self):
        """Test creating a SpeakerSegment entity."""
        # Arrange
        job_id = "test-job-123"
        chunk_id = "test-chunk-456"
        speaker_id = "speaker1"

        # Act
        segment = SpeakerSegment(
            job_id=job_id,
            chunk_id=chunk_id,
            speaker_id=speaker_id,
            start_time=0.0,
            end_time=5.0,
        )

        # Assert
        assert segment.id is not None
        assert segment.job_id == job_id
        assert segment.chunk_id == chunk_id
        assert segment.speaker_id == speaker_id
        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.confidence == 0.0
        assert segment.metadata == {}

    def test_speaker_segment_properties(self):
        """Test the properties of the SpeakerSegment entity."""
        # Arrange
        segment = SpeakerSegment(
            job_id="test-job-123",
            chunk_id="test-chunk-456",
            speaker_id="speaker1",
            start_time=10.0,
            end_time=15.0,
            confidence=0.8,
        )

        # Act & Assert
        assert segment.duration == 5.0


class TestProcessingJob:
    """Tests for the ProcessingJob entity."""

    def test_create_processing_job(self):
        """Test creating a ProcessingJob entity."""
        # Arrange
        audio_path = Path("/tmp/test-audio.mp3")
        output_path = Path("/tmp/test-output.txt")

        # Act
        job = ProcessingJob(
            original_audio_path=audio_path,
            output_path=output_path,
        )

        # Assert
        assert job.id is not None
        assert job.original_audio_path == audio_path
        assert job.output_path == output_path
        assert job.status == ProcessingStatus.PENDING
        assert job.chunks == []
        assert job.total_chunks == 0
        assert job.current_chunk_index == 0
        assert job.chunk_duration == 300.0  # Default 5 minutes
        assert job.overlap_duration == 5.0  # Default 5 seconds
        assert job.processing_options == {}
        assert job.created_at is not None
        assert job.updated_at is not None
        assert job.completed_at is None
        assert job.errors == []

    def test_processing_job_properties(self):
        """Test the properties of the ProcessingJob entity."""
        # Arrange
        job = ProcessingJob(
            original_audio_path=Path("/tmp/test-audio.mp3"),
        )

        # Act & Assert
        assert not job.is_completed
        assert not job.is_failed
        assert job.progress_percentage == 0.0

        # Change status and check properties
        job.status = ProcessingStatus.IN_PROGRESS
        assert not job.is_completed
        assert not job.is_failed

        job.status = ProcessingStatus.COMPLETED
        assert job.is_completed
        assert not job.is_failed

        job.status = ProcessingStatus.FAILED
        assert not job.is_completed
        assert job.is_failed


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def job_repository(temp_data_dir):
    """Create a job repository with a temporary directory."""
    jobs_dir = temp_data_dir / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    return JsonJobRepository(jobs_dir=jobs_dir)


@pytest.fixture
def chunk_repository(temp_data_dir):
    """Create a chunk repository with a temporary directory."""
    chunks_dir = temp_data_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    return JsonChunkRepository(chunks_dir=chunks_dir)


@pytest.fixture
def segment_repository(temp_data_dir):
    """Create a segment repository with a temporary directory."""
    chunks_dir = temp_data_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    return JsonSegmentRepository(chunks_dir=chunks_dir)


@pytest.fixture
def sample_job():
    """Create a sample job for testing."""
    return ProcessingJob(
        original_audio_path=Path("/tmp/test-audio.mp3"),
        output_path=Path("/tmp/test-output.txt"),
    )


@pytest.fixture
def sample_chunk(sample_job):
    """Create a sample chunk for testing."""
    return AudioChunk(
        job_id=sample_job.id,
        chunk_path=Path("/tmp/chunks/chunk1.wav"),
        start_time=0.0,
        end_time=5.0,
        chunk_index=1,
    )


@pytest.fixture
def sample_segment(sample_job, sample_chunk):
    """Create a sample segment for testing."""
    return SpeakerSegment(
        job_id=sample_job.id,
        chunk_id=sample_chunk.id,
        speaker_id="speaker1",
        start_time=0.0,
        end_time=2.5,
        confidence=0.9,
    )


class TestJsonJobRepository:
    """Tests for the JsonJobRepository."""

    def test_save_and_get_job(self, job_repository, sample_job):
        """Test saving and retrieving a job."""
        # Arrange
        job_id = sample_job.id

        # Act
        saved_job = job_repository.save(sample_job)
        retrieved_job = job_repository.get_by_id(job_id)

        # Assert
        assert saved_job.id == job_id
        assert retrieved_job is not None
        assert retrieved_job.id == job_id
        assert retrieved_job.original_audio_path == sample_job.original_audio_path
        assert retrieved_job.status == sample_job.status

    def test_get_by_audio_path(self, job_repository, sample_job):
        """Test retrieving a job by audio path."""
        # Arrange
        job_repository.save(sample_job)

        # Act
        retrieved_job = job_repository.get_by_audio_path(sample_job.original_audio_path)

        # Assert
        assert retrieved_job is not None
        assert retrieved_job.id == sample_job.id

    def test_list_all(self, job_repository, sample_job):
        """Test listing all jobs."""
        # Arrange
        job_repository.save(sample_job)

        # Create another job
        job2 = ProcessingJob(
            original_audio_path=Path("/tmp/test-audio2.mp3"),
        )
        job_repository.save(job2)

        # Act
        jobs = job_repository.list_all()

        # Assert
        assert len(jobs) == 2
        job_ids = [job.id for job in jobs]
        assert sample_job.id in job_ids
        assert job2.id in job_ids

    def test_delete_job(self, job_repository, sample_job):
        """Test deleting a job."""
        # Arrange
        job_repository.save(sample_job)

        # Act
        result = job_repository.delete(sample_job.id)

        # Assert
        assert result is True
        assert job_repository.get_by_id(sample_job.id) is None


class TestJsonChunkRepository:
    """Tests for the JsonChunkRepository."""

    def test_save_and_get_chunk(self, chunk_repository, sample_chunk):
        """Test saving and retrieving a chunk."""
        # Arrange
        chunk_id = sample_chunk.id

        # Act
        saved_chunk = chunk_repository.save(sample_chunk)
        retrieved_chunk = chunk_repository.get_by_id(chunk_id)

        # Assert
        assert saved_chunk.id == chunk_id
        assert retrieved_chunk is not None
        assert retrieved_chunk.id == chunk_id
        assert retrieved_chunk.job_id == sample_chunk.job_id
        assert retrieved_chunk.chunk_path == sample_chunk.chunk_path
        assert retrieved_chunk.status == sample_chunk.status

    def test_save_many_chunks(self, chunk_repository, sample_job):
        """Test saving multiple chunks."""
        # Arrange
        chunks = [
            AudioChunk(
                job_id=sample_job.id,
                chunk_path=Path(f"/tmp/chunks/chunk{i}.wav"),
                start_time=float(i * 5),
                end_time=float((i + 1) * 5),
                chunk_index=i,
            )
            for i in range(3)
        ]

        # Act
        saved_chunks = chunk_repository.save_many(chunks)

        # Assert
        assert len(saved_chunks) == 3
        for i, chunk in enumerate(saved_chunks):
            assert chunk.chunk_index == i

    def test_get_by_job_id(self, chunk_repository, sample_job):
        """Test retrieving chunks by job ID."""
        # Arrange
        chunks = [
            AudioChunk(
                job_id=sample_job.id,
                chunk_path=Path(f"/tmp/chunks/chunk{i}.wav"),
                start_time=float(i * 5),
                end_time=float((i + 1) * 5),
                chunk_index=i,
            )
            for i in range(3)
        ]
        chunk_repository.save_many(chunks)

        # Act
        retrieved_chunks = chunk_repository.get_by_job_id(sample_job.id)

        # Assert
        assert len(retrieved_chunks) == 3
        # Check that chunks are sorted by index
        for i, chunk in enumerate(retrieved_chunks):
            assert chunk.chunk_index == i

    def test_get_by_index(self, chunk_repository, sample_job):
        """Test retrieving a chunk by its index."""
        # Arrange
        chunks = [
            AudioChunk(
                job_id=sample_job.id,
                chunk_path=Path(f"/tmp/chunks/chunk{i}.wav"),
                start_time=float(i * 5),
                end_time=float((i + 1) * 5),
                chunk_index=i,
            )
            for i in range(3)
        ]
        chunk_repository.save_many(chunks)

        # Act
        chunk1 = chunk_repository.get_by_index(sample_job.id, 1)

        # Assert
        assert chunk1 is not None
        assert chunk1.chunk_index == 1
        assert chunk1.start_time == 5.0
        assert chunk1.end_time == 10.0

    def test_delete_chunk(self, chunk_repository, sample_chunk):
        """Test deleting a chunk."""
        # Arrange
        chunk_repository.save(sample_chunk)

        # Act
        result = chunk_repository.delete(sample_chunk.id)

        # Assert
        assert result is True
        assert chunk_repository.get_by_id(sample_chunk.id) is None

    def test_delete_by_job_id(self, chunk_repository, sample_job):
        """Test deleting all chunks for a job."""
        # Arrange
        chunks = [
            AudioChunk(
                job_id=sample_job.id,
                chunk_path=Path(f"/tmp/chunks/chunk{i}.wav"),
                start_time=float(i * 5),
                end_time=float((i + 1) * 5),
                chunk_index=i,
            )
            for i in range(3)
        ]
        chunk_repository.save_many(chunks)

        # Act
        count = chunk_repository.delete_by_job_id(sample_job.id)

        # Assert
        assert count == 3
        assert len(chunk_repository.get_by_job_id(sample_job.id)) == 0


class TestJsonSegmentRepository:
    """Tests for the JsonSegmentRepository."""

    def test_save_and_get_segment(self, segment_repository, sample_segment):
        """Test saving and retrieving a segment."""
        # Arrange
        segment_id = sample_segment.id

        # Act
        saved_segment = segment_repository.save(sample_segment)
        retrieved_segment = segment_repository.get_by_id(segment_id)

        # Assert
        assert saved_segment.id == segment_id
        assert retrieved_segment is not None
        assert retrieved_segment.id == segment_id
        assert retrieved_segment.job_id == sample_segment.job_id
        assert retrieved_segment.chunk_id == sample_segment.chunk_id
        assert retrieved_segment.speaker_id == sample_segment.speaker_id
        assert retrieved_segment.start_time == sample_segment.start_time
        assert retrieved_segment.end_time == sample_segment.end_time

    def test_save_many_segments(self, segment_repository, sample_job, sample_chunk):
        """Test saving multiple segments."""
        # Arrange
        segments = [
            SpeakerSegment(
                job_id=sample_job.id,
                chunk_id=sample_chunk.id,
                speaker_id=f"speaker{i}",
                start_time=float(i),
                end_time=float(i + 1),
                confidence=0.8,
            )
            for i in range(3)
        ]

        # Act
        saved_segments = segment_repository.save_many(segments)

        # Assert
        assert len(saved_segments) == 3
        for i, segment in enumerate(saved_segments):
            assert segment.speaker_id == f"speaker{i}"

    def test_get_by_job_id(self, segment_repository, sample_job, sample_chunk):
        """Test retrieving segments by job ID."""
        # Arrange
        segments = [
            SpeakerSegment(
                job_id=sample_job.id,
                chunk_id=sample_chunk.id,
                speaker_id=f"speaker{i}",
                start_time=float(i),
                end_time=float(i + 1),
                confidence=0.8,
            )
            for i in range(3)
        ]
        segment_repository.save_many(segments)

        # Act
        retrieved_segments = segment_repository.get_by_job_id(sample_job.id)

        # Assert
        assert len(retrieved_segments) == 3
        # Check that segments are sorted by start time
        for i, segment in enumerate(retrieved_segments):
            assert segment.start_time == float(i)

    def test_get_by_chunk_id(self, segment_repository, sample_job, sample_chunk):
        """Test retrieving segments by chunk ID."""
        # Arrange
        segments = [
            SpeakerSegment(
                job_id=sample_job.id,
                chunk_id=sample_chunk.id,
                speaker_id=f"speaker{i}",
                start_time=float(i),
                end_time=float(i + 1),
                confidence=0.8,
            )
            for i in range(3)
        ]
        segment_repository.save_many(segments)

        # Act
        retrieved_segments = segment_repository.get_by_chunk_id(sample_chunk.id)

        # Assert
        assert len(retrieved_segments) == 3
        # Check that segments are sorted by start time
        for i, segment in enumerate(retrieved_segments):
            assert segment.start_time == float(i)

    def test_delete_segment(self, segment_repository, sample_segment):
        """Test deleting a segment."""
        # Arrange
        segment_repository.save(sample_segment)

        # Act
        result = segment_repository.delete(sample_segment.id)

        # Assert
        assert result is True
        assert segment_repository.get_by_id(sample_segment.id) is None

    def test_delete_by_job_id(self, segment_repository, sample_job, sample_chunk):
        """Test deleting all segments for a job."""
        # Arrange
        segments = [
            SpeakerSegment(
                job_id=sample_job.id,
                chunk_id=sample_chunk.id,
                speaker_id=f"speaker{i}",
                start_time=float(i),
                end_time=float(i + 1),
                confidence=0.8,
            )
            for i in range(3)
        ]
        segment_repository.save_many(segments)

        # Act
        count = segment_repository.delete_by_job_id(sample_job.id)

        # Assert
        assert count == 3
        assert len(segment_repository.get_by_job_id(sample_job.id)) == 0

    def test_delete_by_chunk_id(self, segment_repository, sample_job, sample_chunk):
        """Test deleting all segments for a chunk."""
        # Arrange
        segments = [
            SpeakerSegment(
                job_id=sample_job.id,
                chunk_id=sample_chunk.id,
                speaker_id=f"speaker{i}",
                start_time=float(i),
                end_time=float(i + 1),
                confidence=0.8,
            )
            for i in range(3)
        ]
        segment_repository.save_many(segments)

        # Act
        count = segment_repository.delete_by_chunk_id(sample_chunk.id)

        # Assert
        assert count == 3
        assert len(segment_repository.get_by_chunk_id(sample_chunk.id)) == 0
