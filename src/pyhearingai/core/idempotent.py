"""
Idempotent processing domain models for PyHearingAI.

This module contains the entity classes and value objects for supporting
idempotent processing of audio files, allowing for resumable transcription
of long recordings.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pyhearingai.core.models import DiarizationSegment, Segment


class ProcessingStatus(Enum):
    """Status enum for tracking the state of a processing job or chunk."""

    PENDING = auto()
    IN_PROGRESS = auto()
    DIARIZING = auto()
    DIARIZED = auto()
    TRANSCRIBING = auto()
    TRANSCRIBED = auto()
    RECONCILING = auto()
    COMPLETED = auto()
    FAILED = auto()


class ChunkStatus(Enum):
    """Status enum for tracking the state of an audio chunk."""

    PENDING = auto()
    DIARIZING = auto()
    DIARIZED = auto()
    TRANSCRIBING = auto()
    TRANSCRIBED = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class AudioChunk:
    """
    Represents a chunk of audio that is part of a larger processing job.

    A chunk is a segment of the original audio file that can be processed
    independently, allowing for resumable processing and efficient memory usage.
    """

    job_id: str
    chunk_path: Path
    start_time: float  # start time in seconds relative to original audio
    end_time: float  # end time in seconds relative to original audio
    chunk_index: int  # sequential index in the original audio

    # Optional/default fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ChunkStatus = field(default=ChunkStatus.PENDING)
    diarization_segments: List[DiarizationSegment] = field(default_factory=list)
    transcription_segments: List[Segment] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get the duration of the chunk in seconds."""
        return self.end_time - self.start_time

    @property
    def is_processed(self) -> bool:
        """Check if the chunk has been fully processed."""
        return self.status in (ChunkStatus.COMPLETED, ChunkStatus.FAILED)

    @property
    def is_diarized(self) -> bool:
        """Check if the chunk has been diarized."""
        return self.status in (
            ChunkStatus.DIARIZED,
            ChunkStatus.TRANSCRIBING,
            ChunkStatus.TRANSCRIBED,
            ChunkStatus.COMPLETED,
        )

    @property
    def is_transcribed(self) -> bool:
        """Check if the chunk has been transcribed."""
        return self.status in (ChunkStatus.TRANSCRIBED, ChunkStatus.COMPLETED)

    @property
    def index(self) -> int:
        """Return the chunk index for compatibility with batched reconciliation."""
        return self.chunk_index


@dataclass
class SpeakerSegment:
    """
    Represents a segment of audio attributed to a specific speaker.

    Similar to DiarizationSegment but designed for the idempotent processing workflow
    with additional metadata and linking to specific audio chunks.
    """

    job_id: str
    chunk_id: str
    speaker_id: str
    start_time: float  # start time in seconds relative to original audio
    end_time: float  # end time in seconds relative to original audio

    # Optional fields with defaults
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get the duration of the segment in seconds."""
        return self.end_time - self.start_time


@dataclass
class ProcessingJob:
    """
    Represents a processing job for audio transcription.

    This class tracks the state and progress of an audio transcription job,
    including chunk processing and overall job status.
    """

    def __init__(
        self,
        original_audio_path: str,
        id: Optional[str] = None,
        chunk_duration: float = 0.0,
        overlap_duration: float = 5.0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        """
        Initialize a processing job.

        Args:
            original_audio_path: Path to the original audio file
            id: Optional job ID (will be generated if not provided)
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            start_time: Start time in seconds for processing (optional)
            end_time: End time in seconds for processing (optional)
        """
        self.id = id or str(uuid.uuid4())
        self.original_audio_path = original_audio_path
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.start_time = start_time
        self.end_time = end_time
        
        # Status tracking
        self.status = ProcessingStatus.PENDING
        self.error = None
        self.total_chunks = 0
        self.processed_chunks = 0
        self.chunks = []  # List of chunk IDs
        self.results = []  # List of chunk results
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.completed_at = None

    @property
    def is_completed(self) -> bool:
        """Check if the job has been fully processed."""
        return self.status == ProcessingStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if the job has failed."""
        return self.status == ProcessingStatus.FAILED

    @property
    def progress_percentage(self) -> float:
        """Calculate the progress percentage of the job."""
        if self.total_chunks == 0:
            return 0.0
        return (
            len([c for c in self.chunks if c == ChunkStatus.COMPLETED]) / self.total_chunks
        ) * 100
