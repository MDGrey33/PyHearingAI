"""
DiarizationRepository implementation for storing and retrieving diarization results.

This module provides persistence for speaker diarization results, enabling
resumable processing of audio chunks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from pyhearingai.core.idempotent import AudioChunk, ProcessingStatus
from pyhearingai.core.models import DiarizationSegment

logger = logging.getLogger(__name__)


class DiarizationRepository:
    """
    Repository for storing and retrieving diarization results per chunk.

    This repository manages the persistence of diarization results, allowing
    resumable processing of audio files.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the diarization repository.

        Args:
            base_dir: Base directory for storing diarization data.
                     Defaults to ~/.local/share/pyhearingai/diarization
        """
        if base_dir is None:
            home_dir = Path.home()
            base_dir = home_dir / ".local" / "share" / "pyhearingai" / "diarization"

        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized DiarizationRepository at {self.base_dir}")

    def _get_job_dir(self, job_id: str) -> Path:
        """
        Get the directory for a specific job.

        Args:
            job_id: ID of the processing job

        Returns:
            Path to the job's diarization directory
        """
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _get_chunk_file_path(self, job_id: str, chunk_id: str) -> Path:
        """
        Get the file path for storing diarization results for a chunk.

        Args:
            job_id: ID of the processing job
            chunk_id: ID of the audio chunk

        Returns:
            Path to the diarization results file
        """
        return self._get_job_dir(job_id) / f"{chunk_id}.json"

    def save(self, job_id: str, chunk_id: str, segments: List[DiarizationSegment]) -> bool:
        """
        Save diarization segments for a specific chunk.

        Args:
            job_id: ID of the processing job
            chunk_id: ID of the audio chunk
            segments: List of diarization segments

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_chunk_file_path(job_id, chunk_id)

            # Convert segments to dicts for serialization
            segments_data = [
                {
                    "speaker_id": segment.speaker_id,
                    "start": segment.start,
                    "end": segment.end,
                    "score": segment.score,
                }
                for segment in segments
            ]

            # Add metadata
            data = {"job_id": job_id, "chunk_id": chunk_id, "segments": segments_data}

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(segments)} diarization segments for chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving diarization segments for chunk {chunk_id}: {e}")
            return False

    def get(self, job_id: str, chunk_id: str) -> Optional[List[DiarizationSegment]]:
        """
        Get diarization segments for a specific chunk.

        Args:
            job_id: ID of the processing job
            chunk_id: ID of the audio chunk

        Returns:
            List of diarization segments, or None if not found
        """
        file_path = self._get_chunk_file_path(job_id, chunk_id)

        if not file_path.exists():
            logger.debug(f"No diarization data found for chunk {chunk_id}")
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            segments = [
                DiarizationSegment(
                    speaker_id=segment["speaker_id"],
                    start=segment["start"],
                    end=segment["end"],
                    score=segment["score"],
                )
                for segment in data["segments"]
            ]

            logger.debug(f"Loaded {len(segments)} diarization segments for chunk {chunk_id}")
            return segments
        except Exception as e:
            logger.error(f"Error loading diarization segments for chunk {chunk_id}: {e}")
            return None

    def exists(self, job_id: str, chunk_id: str) -> bool:
        """
        Check if diarization results exist for a specific chunk.

        Args:
            job_id: ID of the processing job
            chunk_id: ID of the audio chunk

        Returns:
            True if results exist, False otherwise
        """
        file_path = self._get_chunk_file_path(job_id, chunk_id)
        return file_path.exists()

    def delete(self, job_id: str, chunk_id: str) -> bool:
        """
        Delete diarization results for a specific chunk.

        Args:
            job_id: ID of the processing job
            chunk_id: ID of the audio chunk

        Returns:
            True if successful, False otherwise
        """
        file_path = self._get_chunk_file_path(job_id, chunk_id)

        if not file_path.exists():
            logger.debug(f"No diarization data found for chunk {chunk_id} to delete")
            return True

        try:
            file_path.unlink()
            logger.debug(f"Deleted diarization results for chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting diarization results for chunk {chunk_id}: {e}")
            return False

    def get_by_job_id(self, job_id: str) -> Dict[str, List[DiarizationSegment]]:
        """
        Get all diarization results for a specific job.

        Args:
            job_id: ID of the processing job

        Returns:
            Dict mapping chunk IDs to lists of diarization segments
        """
        job_dir = self._get_job_dir(job_id)
        result = {}

        if not job_dir.exists():
            logger.debug(f"No diarization data found for job {job_id}")
            return result

        for file_path in job_dir.glob("*.json"):
            chunk_id = file_path.stem
            segments = self.get(job_id, chunk_id)
            if segments:
                result[chunk_id] = segments

        logger.debug(f"Loaded diarization results for {len(result)} chunks in job {job_id}")
        return result

    def get_all(self, job_id: str) -> Dict[str, List[DiarizationSegment]]:
        """
        Get all diarization results for a specific job.

        This is an alias for get_by_job_id for compatibility with the reconciliation service.

        Args:
            job_id: ID of the processing job

        Returns:
            Dict mapping chunk IDs to lists of diarization segments
        """
        return self.get_by_job_id(job_id)
