"""
Transcription repository for storing and retrieving transcription results.

This module provides a repository implementation for managing transcription results
for both whole chunks and individual speaker segments.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from pyhearingai.core.idempotent import AudioChunk, ProcessingJob, SpeakerSegment
from pyhearingai.core.models import Segment

logger = logging.getLogger(__name__)


class TranscriptionRepository:
    """
    Repository for storing and retrieving transcription results.

    This repository handles both whole chunk transcriptions and
    individual speaker segment transcriptions, storing them as JSON files.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the transcription repository.

        Args:
            base_dir: Optional base directory for storing results. If not provided,
                     the default location in the user's home directory will be used.
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Use default location in user's home directory
            home_dir = Path.home()
            self.base_dir = home_dir / ".local" / "share" / "pyhearingai"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized TranscriptionRepository with base directory: {self.base_dir}")

    def _get_job_dir(self, job_id: str) -> Path:
        """Get the directory for a specific job"""
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _get_chunks_dir(self, job_id: str) -> Path:
        """Get the directory for chunk transcriptions"""
        chunks_dir = self._get_job_dir(job_id) / "transcriptions" / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        return chunks_dir

    def _get_segments_dir(self, job_id: str) -> Path:
        """Get the directory for segment transcriptions"""
        segments_dir = self._get_job_dir(job_id) / "transcriptions" / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        return segments_dir

    def save_chunk_transcription(
        self, job_id: str, chunk_id: str, segments: List[Segment], metadata: Optional[Dict] = None
    ) -> None:
        """
        Save transcription results for a chunk.

        Args:
            job_id: ID of the processing job
            chunk_id: ID of the audio chunk
            segments: List of transcription segments
            metadata: Optional metadata about the transcription
        """
        chunks_dir = self._get_chunks_dir(job_id)
        file_path = chunks_dir / f"{chunk_id}.json"

        data = {
            "chunk_id": chunk_id,
            "job_id": job_id,
            "segments": [
                {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "speaker_id": segment.speaker_id,
                }
                for segment in segments
            ],
            "metadata": metadata or {},
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved transcription for chunk {chunk_id} with {len(segments)} segments")

    def save_segment_transcription(
        self, job_id: str, segment_id: str, text: str, metadata: Optional[Dict] = None
    ) -> None:
        """
        Save transcription results for a speaker segment.

        Args:
            job_id: ID of the processing job
            segment_id: ID of the speaker segment
            text: Transcribed text for the segment
            metadata: Optional metadata about the transcription
        """
        segments_dir = self._get_segments_dir(job_id)
        file_path = segments_dir / f"{segment_id}.json"

        data = {
            "segment_id": segment_id,
            "job_id": job_id,
            "text": text,
            "metadata": metadata or {},
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved transcription for segment {segment_id}")

    def get_chunk_transcription(self, job_id: str, chunk_id: str) -> Optional[List[Segment]]:
        """
        Get transcription results for a chunk.

        Args:
            job_id: ID of the processing job
            chunk_id: ID of the audio chunk

        Returns:
            List of transcription segments, or None if not found
        """
        chunks_dir = self._get_chunks_dir(job_id)
        file_path = chunks_dir / f"{chunk_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            segments = [
                Segment(
                    text=segment["text"],
                    start=segment["start"],
                    end=segment["end"],
                    speaker_id=segment.get("speaker_id"),
                )
                for segment in data.get("segments", [])
            ]

            logger.debug(f"Loaded transcription for chunk {chunk_id} with {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Error loading transcription for chunk {chunk_id}: {str(e)}")
            return None

    def get_segment_transcription(self, job_id: str, segment_id: str) -> Optional[str]:
        """
        Get transcription results for a speaker segment.

        Args:
            job_id: ID of the processing job
            segment_id: ID of the speaker segment

        Returns:
            Transcribed text for the segment, or None if not found
        """
        segments_dir = self._get_segments_dir(job_id)
        file_path = segments_dir / f"{segment_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            text = data.get("text", "")

            logger.debug(f"Loaded transcription for segment {segment_id}")
            return text

        except Exception as e:
            logger.error(f"Error loading transcription for segment {segment_id}: {str(e)}")
            return None

    def chunk_exists(self, job_id: str, chunk_id: str) -> bool:
        """Check if transcription exists for a chunk"""
        chunks_dir = self._get_chunks_dir(job_id)
        file_path = chunks_dir / f"{chunk_id}.json"
        return file_path.exists()

    def segment_exists(self, job_id: str, segment_id: str) -> bool:
        """Check if transcription exists for a segment"""
        segments_dir = self._get_segments_dir(job_id)
        file_path = segments_dir / f"{segment_id}.json"
        return file_path.exists()

    def delete_chunk_transcription(self, job_id: str, chunk_id: str) -> bool:
        """Delete transcription for a chunk"""
        chunks_dir = self._get_chunks_dir(job_id)
        file_path = chunks_dir / f"{chunk_id}.json"

        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted transcription for chunk {chunk_id}")
            return True
        return False

    def delete_segment_transcription(self, job_id: str, segment_id: str) -> bool:
        """Delete transcription for a segment"""
        segments_dir = self._get_segments_dir(job_id)
        file_path = segments_dir / f"{segment_id}.json"

        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted transcription for segment {segment_id}")
            return True
        return False
