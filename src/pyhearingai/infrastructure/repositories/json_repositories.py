"""
JSON-based implementations of repository interfaces.

This module provides repository implementations that store entities as JSON files
on the local filesystem.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from pyhearingai.config import JOBS_DIR, CHUNKS_DIR
from pyhearingai.core.idempotent import (
    AudioChunk,
    ChunkStatus,
    ProcessingJob,
    ProcessingStatus,
    SpeakerSegment,
)
from pyhearingai.core.idempotent_ports import (
    ChunkRepository,
    JobRepository,
    SegmentRepository,
)


class JsonFileHandler:
    """Helper class for reading and writing JSON files with proper error handling."""

    @staticmethod
    def read_json(file_path: Path) -> Dict[str, Any]:
        """
        Read a JSON file with proper error handling.

        Args:
            file_path: Path to the JSON file

        Returns:
            The parsed JSON data as a dictionary
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            # If the file exists but is corrupted, return an empty dict
            # In a production system, we might want to log this or take recovery actions
            return {}

    @staticmethod
    def write_json(file_path: Path, data: Dict[str, Any]) -> None:
        """
        Write data to a JSON file with proper error handling.

        Args:
            file_path: Path to the JSON file
            data: Data to write to the file
        """
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=JsonFileHandler.json_serializer)

    @staticmethod
    def json_serializer(obj):
        """
        Custom JSON serializer to handle non-serializable types.

        Args:
            obj: Object to serialize

        Returns:
            A JSON serializable representation of the object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (ProcessingStatus, ChunkStatus)):
            return obj.name
        raise TypeError(f"Type {type(obj)} not serializable")


class JsonJobRepository(JobRepository):
    """
    Repository implementation for ProcessingJob using JSON files.

    Each job is stored as a separate JSON file in the jobs directory.
    """

    def __init__(self, jobs_dir: Union[str, Path] = JOBS_DIR):
        """
        Initialize the repository with the directory to store jobs.

        Args:
            jobs_dir: Directory to store job JSON files
        """
        self.jobs_dir = Path(jobs_dir) if isinstance(jobs_dir, str) else jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _get_job_path(self, job_id: str) -> Path:
        """Get the file path for a job."""
        return self.jobs_dir / f"{job_id}.json"

    def _to_dict(self, job: ProcessingJob) -> Dict[str, Any]:
        """Convert a ProcessingJob to a dictionary for serialization."""
        # Handle datetime fields - ensure they're converted to strings if they're datetime objects
        created_at = job.created_at
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()

        updated_at = job.updated_at
        if isinstance(updated_at, datetime):
            updated_at = updated_at.isoformat()

        completed_at = job.completed_at
        if isinstance(completed_at, datetime):
            completed_at = completed_at.isoformat()

        return {
            "id": job.id,
            "original_audio_path": str(job.original_audio_path),
            "status": job.status.name,  # Use the enum name instead of value
            "chunk_duration": job.chunk_duration,
            "overlap_duration": job.overlap_duration,
            "start_time": job.start_time,
            "end_time": job.end_time,
            "total_chunks": job.total_chunks,
            "processed_chunks": job.processed_chunks,
            "chunks": job.chunks,
            "results": job.results,
            "created_at": created_at,
            "updated_at": updated_at,
            "completed_at": completed_at,
            "error": job.error,
        }

    def _from_dict(self, data: Dict[str, Any]) -> ProcessingJob:
        """Convert a dictionary to a ProcessingJob."""
        # Convert status string to enum
        status_value = data.get("status", "PENDING")
        if isinstance(status_value, str):
            status = ProcessingStatus[status_value]
        else:
            status = ProcessingStatus(status_value)

        # Create the job with required fields
        job = ProcessingJob(
            original_audio_path=str(data["original_audio_path"]),
            id=data.get("id"),
            chunk_duration=data.get("chunk_duration", 0.0),
            overlap_duration=data.get("overlap_duration", 0.0),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
        )

        # Set additional fields
        job.status = status
        job.total_chunks = data.get("total_chunks", 0)
        job.processed_chunks = data.get("processed_chunks", 0)
        job.chunks = data.get("chunks", [])
        job.results = data.get("results", [])
        job.error = data.get("error")

        # Handle datetime fields
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            job.created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if updated_at and isinstance(updated_at, str):
            job.updated_at = datetime.fromisoformat(updated_at)

        completed_at = data.get("completed_at")
        if completed_at and isinstance(completed_at, str):
            job.completed_at = datetime.fromisoformat(completed_at)

        return job

    def save(self, job: ProcessingJob) -> ProcessingJob:
        """Save a job to the repository."""
        # Update the timestamp
        job.updated_at = datetime.now()

        # Convert to dict and save
        job_dict = self._to_dict(job)
        JsonFileHandler.write_json(self._get_job_path(job.id), job_dict)

        return job

    def get_by_id(self, job_id: str) -> Optional[ProcessingJob]:
        """Get a job by its ID."""
        job_path = self._get_job_path(job_id)
        try:
            data = JsonFileHandler.read_json(job_path)
            if not data:
                return None

            # Check if essential fields are present
            if "status" not in data:
                return None

            return self._from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def find_by_id(self, job_id: str) -> Optional[ProcessingJob]:
        """Alias for get_by_id for compatibility."""
        return self.get_by_id(job_id)

    def get_by_audio_path(self, audio_path: Path) -> Optional[ProcessingJob]:
        """Get a job by the original audio path."""
        # Normalize path string representation
        audio_path_str = str(audio_path)

        # Iterate through all job files
        for job_file in self.jobs_dir.glob("*.json"):
            job_dict = JsonFileHandler.read_json(job_file)
            # Compare just the path strings, not the resolved paths
            if job_dict and str(job_dict.get("original_audio_path")) == audio_path_str:
                return self._from_dict(job_dict)

        return None

    def list_all(self) -> List[ProcessingJob]:
        """List all jobs in the repository."""
        jobs = []

        for job_file in self.jobs_dir.glob("*.json"):
            job_dict = JsonFileHandler.read_json(job_file)
            if job_dict:  # Skip empty/corrupted files
                jobs.append(self._from_dict(job_dict))

        return jobs

    def delete(self, job_id: str) -> bool:
        """Delete a job from the repository."""
        job_path = self._get_job_path(job_id)
        try:
            job_path.unlink()
            return True
        except FileNotFoundError:
            return False


class JsonChunkRepository(ChunkRepository):
    """
    Repository implementation for AudioChunk using JSON files.

    Chunks are stored in a directory structure organized by job ID.
    """

    def __init__(self, chunks_dir: Union[str, Path] = CHUNKS_DIR):
        """
        Initialize the repository with the directory to store chunks.

        Args:
            chunks_dir: Base directory to store chunk JSON files
        """
        self.chunks_dir = Path(chunks_dir) if isinstance(chunks_dir, str) else chunks_dir
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def _get_job_chunks_dir(self, job_id: str) -> Path:
        """Get the directory for a job's chunks."""
        job_chunks_dir = self.chunks_dir / job_id
        job_chunks_dir.mkdir(parents=True, exist_ok=True)
        return job_chunks_dir

    def _get_chunk_path(self, chunk_id: str, job_id: str) -> Path:
        """Get the file path for a chunk."""
        return self._get_job_chunks_dir(job_id) / f"{chunk_id}.json"

    def _to_dict(self, chunk: AudioChunk) -> Dict[str, Any]:
        """Convert an AudioChunk to a dictionary for serialization."""
        return {
            "id": chunk.id,
            "job_id": chunk.job_id,
            "chunk_path": str(chunk.chunk_path),
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "chunk_index": chunk.chunk_index,
            "status": chunk.status.name,
            # For diarization and transcription segments, store IDs only
            # The actual segments are stored in the segment repository
            "diarization_segment_ids": [
                segment.speaker_id for segment in chunk.diarization_segments
            ],
            "transcription_segment_ids": [
                segment.speaker_id if segment.speaker_id else f"segment_{i}"
                for i, segment in enumerate(chunk.transcription_segments)
            ],
            "created_at": chunk.created_at.isoformat(),
            "updated_at": chunk.updated_at.isoformat(),
            "metadata": chunk.metadata,
        }

    def _from_dict(self, data: Dict[str, Any]) -> AudioChunk:
        """Convert a dictionary to an AudioChunk."""
        # Convert string path to Path object
        chunk_path = Path(data["chunk_path"])

        # Convert status string to enum
        status_value = data["status"]
        if isinstance(status_value, str):
            status = ChunkStatus[status_value]
        else:
            status = ChunkStatus(status_value)

        # Parse datetime strings
        created_at = datetime.fromisoformat(data["created_at"])
        updated_at = datetime.fromisoformat(data["updated_at"])

        # Note: We don't load the actual segments here, as they're stored in the segment repository
        # The application layer is responsible for loading and combining these

        return AudioChunk(
            id=data["id"],
            job_id=data["job_id"],
            chunk_path=chunk_path,
            start_time=data["start_time"],
            end_time=data["end_time"],
            chunk_index=data["chunk_index"],
            status=status,
            diarization_segments=[],  # Empty list, to be populated by application layer
            transcription_segments=[],  # Empty list, to be populated by application layer
            created_at=created_at,
            updated_at=updated_at,
            metadata=data["metadata"],
        )

    def save(self, chunk: AudioChunk) -> AudioChunk:
        """Save a chunk to the repository."""
        # Update the timestamp
        chunk.updated_at = datetime.now()

        # Convert to dict and save
        chunk_dict = self._to_dict(chunk)
        JsonFileHandler.write_json(self._get_chunk_path(chunk.id, chunk.job_id), chunk_dict)

        return chunk

    def save_many(self, chunks: List[AudioChunk]) -> List[AudioChunk]:
        """Save multiple chunks to the repository."""
        saved_chunks = []
        for chunk in chunks:
            saved_chunks.append(self.save(chunk))
        return saved_chunks

    def get_by_id(self, chunk_id: str) -> Optional[AudioChunk]:
        """Get a chunk by its ID."""
        # We need to find which job this chunk belongs to
        for job_dir in self.chunks_dir.glob("*"):
            if not job_dir.is_dir():
                continue

            chunk_path = job_dir / f"{chunk_id}.json"
            if chunk_path.exists():
                chunk_dict = JsonFileHandler.read_json(chunk_path)
                if chunk_dict:
                    return self._from_dict(chunk_dict)

        return None

    def get_by_job_id(self, job_id: str) -> List[AudioChunk]:
        """Get all chunks for a job."""
        chunks = []
        job_chunks_dir = self._get_job_chunks_dir(job_id)

        for chunk_file in job_chunks_dir.glob("*.json"):
            chunk_dict = JsonFileHandler.read_json(chunk_file)
            if chunk_dict:  # Skip empty/corrupted files
                chunks.append(self._from_dict(chunk_dict))

        # Sort chunks by index to ensure correct order
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    def get_by_index(self, job_id: str, chunk_index: int) -> Optional[AudioChunk]:
        """Get a chunk by its job ID and index."""
        for chunk in self.get_by_job_id(job_id):
            if chunk.chunk_index == chunk_index:
                return chunk
        return None

    def delete(self, chunk_id: str) -> bool:
        """Delete a chunk from the repository."""
        # We need to find which job this chunk belongs to
        for job_dir in self.chunks_dir.glob("*"):
            if not job_dir.is_dir():
                continue

            chunk_path = job_dir / f"{chunk_id}.json"
            try:
                chunk_path.unlink()
                return True
            except FileNotFoundError:
                continue

        return False

    def delete_by_job_id(self, job_id: str) -> int:
        """Delete all chunks for a job."""
        job_chunks_dir = self._get_job_chunks_dir(job_id)
        count = 0

        for chunk_file in job_chunks_dir.glob("*.json"):
            try:
                chunk_file.unlink()
                count += 1
            except FileNotFoundError:
                continue

        return count

    def has_chunks(self, job_id: str) -> bool:
        """
        Check if a job has any chunks.

        Args:
            job_id: ID of the job to check

        Returns:
            True if the job has chunks, False otherwise
        """
        job_chunks_dir = self._get_job_chunks_dir(job_id)
        return any(job_chunks_dir.glob("*.json"))


class JsonSegmentRepository(SegmentRepository):
    """
    Repository implementation for SpeakerSegment using JSON files.

    Segments are stored in a directory structure organized by job ID and chunk ID.
    """

    def __init__(self, chunks_dir: Path = CHUNKS_DIR):
        """
        Initialize the repository with the directory to store segments.

        Args:
            chunks_dir: Base directory where chunk and segment data is stored
        """
        self.chunks_dir = chunks_dir
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def _get_segments_dir(self, job_id: str, chunk_id: str) -> Path:
        """Get the directory for a chunk's segments."""
        segments_dir = self.chunks_dir / job_id / chunk_id / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        return segments_dir

    def _get_segment_path(self, segment_id: str, job_id: str, chunk_id: str) -> Path:
        """Get the file path for a segment."""
        return self._get_segments_dir(job_id, chunk_id) / f"{segment_id}.json"

    def _to_dict(self, segment: SpeakerSegment) -> Dict[str, Any]:
        """Convert a SpeakerSegment to a dictionary for serialization."""
        return {
            "id": segment.id,
            "job_id": segment.job_id,
            "chunk_id": segment.chunk_id,
            "speaker_id": segment.speaker_id,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "confidence": segment.confidence,
            "metadata": segment.metadata,
        }

    def _from_dict(self, data: Dict[str, Any]) -> SpeakerSegment:
        """Convert a dictionary to a SpeakerSegment."""
        return SpeakerSegment(
            id=data["id"],
            job_id=data["job_id"],
            chunk_id=data["chunk_id"],
            speaker_id=data["speaker_id"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            confidence=data["confidence"],
            metadata=data["metadata"],
        )

    def save(self, segment: SpeakerSegment) -> SpeakerSegment:
        """Save a segment to the repository."""
        # Convert to dict and save
        segment_dict = self._to_dict(segment)
        JsonFileHandler.write_json(
            self._get_segment_path(segment.id, segment.job_id, segment.chunk_id),
            segment_dict,
        )

        return segment

    def save_many(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Save multiple segments to the repository."""
        saved_segments = []
        for segment in segments:
            saved_segments.append(self.save(segment))
        return saved_segments

    def get_by_id(self, segment_id: str) -> Optional[SpeakerSegment]:
        """Get a segment by its ID."""
        # We need to search through all job and chunk directories
        for job_dir in self.chunks_dir.glob("*"):
            if not job_dir.is_dir():
                continue

            for chunk_dir in job_dir.glob("*"):
                if not chunk_dir.is_dir():
                    continue

                segments_dir = chunk_dir / "segments"
                if not segments_dir.exists() or not segments_dir.is_dir():
                    continue

                segment_path = segments_dir / f"{segment_id}.json"
                if segment_path.exists():
                    segment_dict = JsonFileHandler.read_json(segment_path)
                    if segment_dict:
                        return self._from_dict(segment_dict)

        return None

    def get_by_job_id(self, job_id: str) -> List[SpeakerSegment]:
        """Get all segments for a job."""
        segments = []
        job_dir = self.chunks_dir / job_id

        if not job_dir.exists() or not job_dir.is_dir():
            return segments

        for chunk_dir in job_dir.glob("*"):
            if not chunk_dir.is_dir():
                continue

            segments_dir = chunk_dir / "segments"
            if not segments_dir.exists() or not segments_dir.is_dir():
                continue

            for segment_file in segments_dir.glob("*.json"):
                segment_dict = JsonFileHandler.read_json(segment_file)
                if segment_dict:  # Skip empty/corrupted files
                    segments.append(self._from_dict(segment_dict))

        # Sort segments by start time to ensure correct order
        segments.sort(key=lambda s: s.start_time)
        return segments

    def get_by_chunk_id(self, chunk_id: str) -> List[SpeakerSegment]:
        """Get all segments for a chunk."""
        segments = []

        # We need to find which job this chunk belongs to
        for job_dir in self.chunks_dir.glob("*"):
            if not job_dir.is_dir():
                continue

            chunk_dir = job_dir / chunk_id
            if not chunk_dir.exists() or not chunk_dir.is_dir():
                continue

            segments_dir = chunk_dir / "segments"
            if not segments_dir.exists() or not segments_dir.is_dir():
                continue

            for segment_file in segments_dir.glob("*.json"):
                segment_dict = JsonFileHandler.read_json(segment_file)
                if segment_dict:  # Skip empty/corrupted files
                    segments.append(self._from_dict(segment_dict))

        # Sort segments by start time to ensure correct order
        segments.sort(key=lambda s: s.start_time)
        return segments

    def delete(self, segment_id: str) -> bool:
        """Delete a segment from the repository."""
        # We need to search through all job and chunk directories
        for job_dir in self.chunks_dir.glob("*"):
            if not job_dir.is_dir():
                continue

            for chunk_dir in job_dir.glob("*"):
                if not chunk_dir.is_dir():
                    continue

                segments_dir = chunk_dir / "segments"
                if not segments_dir.exists() or not segments_dir.is_dir():
                    continue

                segment_path = segments_dir / f"{segment_id}.json"
                try:
                    segment_path.unlink()
                    return True
                except FileNotFoundError:
                    continue

        return False

    def delete_by_job_id(self, job_id: str) -> int:
        """Delete all segments for a job."""
        job_dir = self.chunks_dir / job_id
        count = 0

        if not job_dir.exists() or not job_dir.is_dir():
            return count

        for chunk_dir in job_dir.glob("*"):
            if not chunk_dir.is_dir():
                continue

            segments_dir = chunk_dir / "segments"
            if not segments_dir.exists() or not segments_dir.is_dir():
                continue

            for segment_file in segments_dir.glob("*.json"):
                try:
                    segment_file.unlink()
                    count += 1
                except FileNotFoundError:
                    continue

        return count

    def delete_by_chunk_id(self, chunk_id: str) -> int:
        """Delete all segments for a chunk."""
        count = 0

        # We need to find which job this chunk belongs to
        for job_dir in self.chunks_dir.glob("*"):
            if not job_dir.is_dir():
                continue

            chunk_dir = job_dir / chunk_id
            if not chunk_dir.exists() or not chunk_dir.is_dir():
                continue

            segments_dir = chunk_dir / "segments"
            if not segments_dir.exists() or not segments_dir.is_dir():
                continue

            for segment_file in segments_dir.glob("*.json"):
                try:
                    segment_file.unlink()
                    count += 1
                except FileNotFoundError:
                    continue

        return count
