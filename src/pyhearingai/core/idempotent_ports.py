"""
Idempotent processing ports (interfaces) for PyHearingAI.

This module defines the abstract interfaces for repositories that store and retrieve
entities related to idempotent audio processing.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

from pyhearingai.core.idempotent import AudioChunk, ProcessingJob, SpeakerSegment


class JobRepository(ABC):
    """Interface for storing and retrieving processing jobs."""

    @abstractmethod
    def save(self, job: ProcessingJob) -> ProcessingJob:
        """
        Save a processing job to the repository.

        Args:
            job: The processing job to save

        Returns:
            The saved processing job with any updates (e.g., ID)
        """
        pass

    @abstractmethod
    def get_by_id(self, job_id: str) -> Optional[ProcessingJob]:
        """
        Retrieve a processing job by its ID.

        Args:
            job_id: The ID of the job to retrieve

        Returns:
            The processing job or None if not found
        """
        pass

    @abstractmethod
    def get_by_audio_path(self, audio_path: Path) -> Optional[ProcessingJob]:
        """
        Retrieve a processing job by the original audio path.

        Args:
            audio_path: The path to the original audio file

        Returns:
            The processing job or None if not found
        """
        pass

    @abstractmethod
    def list_all(self) -> List[ProcessingJob]:
        """
        List all processing jobs in the repository.

        Returns:
            A list of all processing jobs
        """
        pass

    @abstractmethod
    def delete(self, job_id: str) -> bool:
        """
        Delete a processing job from the repository.

        Args:
            job_id: The ID of the job to delete

        Returns:
            True if the job was deleted, False otherwise
        """
        pass


class ChunkRepository(ABC):
    """Interface for storing and retrieving audio chunks."""

    @abstractmethod
    def save(self, chunk: AudioChunk) -> AudioChunk:
        """
        Save an audio chunk to the repository.

        Args:
            chunk: The audio chunk to save

        Returns:
            The saved audio chunk with any updates
        """
        pass

    @abstractmethod
    def save_many(self, chunks: List[AudioChunk]) -> List[AudioChunk]:
        """
        Save multiple audio chunks to the repository.

        Args:
            chunks: The audio chunks to save

        Returns:
            The saved audio chunks with any updates
        """
        pass

    @abstractmethod
    def get_by_id(self, chunk_id: str) -> Optional[AudioChunk]:
        """
        Retrieve an audio chunk by its ID.

        Args:
            chunk_id: The ID of the chunk to retrieve

        Returns:
            The audio chunk or None if not found
        """
        pass

    @abstractmethod
    def get_by_job_id(self, job_id: str) -> List[AudioChunk]:
        """
        Retrieve all audio chunks for a specific job.

        Args:
            job_id: The ID of the job to retrieve chunks for

        Returns:
            A list of audio chunks for the job
        """
        pass

    @abstractmethod
    def get_by_index(self, job_id: str, chunk_index: int) -> Optional[AudioChunk]:
        """
        Retrieve an audio chunk by its job ID and index.

        Args:
            job_id: The ID of the job
            chunk_index: The index of the chunk within the job

        Returns:
            The audio chunk or None if not found
        """
        pass

    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        """
        Delete an audio chunk from the repository.

        Args:
            chunk_id: The ID of the chunk to delete

        Returns:
            True if the chunk was deleted, False otherwise
        """
        pass

    @abstractmethod
    def delete_by_job_id(self, job_id: str) -> int:
        """
        Delete all audio chunks for a specific job.

        Args:
            job_id: The ID of the job to delete chunks for

        Returns:
            The number of chunks deleted
        """
        pass


class SegmentRepository(ABC):
    """Interface for storing and retrieving speaker segments."""

    @abstractmethod
    def save(self, segment: SpeakerSegment) -> SpeakerSegment:
        """
        Save a speaker segment to the repository.

        Args:
            segment: The speaker segment to save

        Returns:
            The saved speaker segment with any updates
        """
        pass

    @abstractmethod
    def save_many(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Save multiple speaker segments to the repository.

        Args:
            segments: The speaker segments to save

        Returns:
            The saved speaker segments with any updates
        """
        pass

    @abstractmethod
    def get_by_id(self, segment_id: str) -> Optional[SpeakerSegment]:
        """
        Retrieve a speaker segment by its ID.

        Args:
            segment_id: The ID of the segment to retrieve

        Returns:
            The speaker segment or None if not found
        """
        pass

    @abstractmethod
    def get_by_job_id(self, job_id: str) -> List[SpeakerSegment]:
        """
        Retrieve all speaker segments for a specific job.

        Args:
            job_id: The ID of the job to retrieve segments for

        Returns:
            A list of speaker segments for the job
        """
        pass

    @abstractmethod
    def get_by_chunk_id(self, chunk_id: str) -> List[SpeakerSegment]:
        """
        Retrieve all speaker segments for a specific chunk.

        Args:
            chunk_id: The ID of the chunk to retrieve segments for

        Returns:
            A list of speaker segments for the chunk
        """
        pass

    @abstractmethod
    def delete(self, segment_id: str) -> bool:
        """
        Delete a speaker segment from the repository.

        Args:
            segment_id: The ID of the segment to delete

        Returns:
            True if the segment was deleted, False otherwise
        """
        pass

    @abstractmethod
    def delete_by_job_id(self, job_id: str) -> int:
        """
        Delete all speaker segments for a specific job.

        Args:
            job_id: The ID of the job to delete segments for

        Returns:
            The number of segments deleted
        """
        pass

    @abstractmethod
    def delete_by_chunk_id(self, chunk_id: str) -> int:
        """
        Delete all speaker segments for a specific chunk.

        Args:
            chunk_id: The ID of the chunk to delete segments for

        Returns:
            The number of segments deleted
        """
        pass
