"""
Reconciliation repository for storing and retrieving reconciled transcription results.

This module provides a repository implementation for managing the final reconciled
transcription results, which combine diarization and transcription data.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from pyhearingai.core.models import Segment

logger = logging.getLogger(__name__)


class ReconciliationRepository:
    """
    Repository for storing and retrieving reconciled transcription results.

    This repository handles the final output of the reconciliation process,
    which combines diarization and transcription data into a coherent result.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the reconciliation repository.

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
        logger.debug(f"Initialized ReconciliationRepository with base directory: {self.base_dir}")

    def _get_job_dir(self, job_id: str) -> Path:
        """Get the directory for a specific job"""
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _get_reconciliation_dir(self, job_id: str) -> Path:
        """Get the directory for reconciled results"""
        reconciliation_dir = self._get_job_dir(job_id) / "reconciliation"
        reconciliation_dir.mkdir(parents=True, exist_ok=True)
        return reconciliation_dir

    def save_reconciled_result(
        self, job_id: str, segments: List[Segment], metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save reconciled transcription results for a job.

        Args:
            job_id: ID of the processing job
            segments: List of reconciled transcription segments
            metadata: Optional metadata about the reconciliation process

        Returns:
            Path to the saved file
        """
        reconciliation_dir = self._get_reconciliation_dir(job_id)
        file_path = reconciliation_dir / "reconciled.json"

        data = {
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

        logger.debug(f"Saved reconciled result for job {job_id} with {len(segments)} segments")
        return file_path

    def save_formatted_output(self, job_id: str, output_format: str, content: str) -> Path:
        """
        Save formatted output in the requested format.

        Args:
            job_id: ID of the processing job
            output_format: Format of the output (txt, json, srt, vtt, md)
            content: Formatted content

        Returns:
            Path to the saved file
        """
        reconciliation_dir = self._get_reconciliation_dir(job_id)
        file_path = reconciliation_dir / f"output.{output_format}"

        with open(file_path, "w") as f:
            f.write(content)

        logger.debug(f"Saved formatted output for job {job_id} in {output_format} format")
        return file_path

    def get_reconciled_result(self, job_id: str) -> Optional[List[Segment]]:
        """
        Get reconciled transcription results for a job.

        Args:
            job_id: ID of the processing job

        Returns:
            List of reconciled transcription segments, or None if not found
        """
        reconciliation_dir = self._get_reconciliation_dir(job_id)
        file_path = reconciliation_dir / "reconciled.json"

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

            logger.debug(f"Loaded reconciled result for job {job_id} with {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Error loading reconciled result for job {job_id}: {str(e)}")
            return None

    def get_formatted_output(self, job_id: str, output_format: str) -> Optional[str]:
        """
        Get formatted output for a job.

        Args:
            job_id: ID of the processing job
            output_format: Format of the output (txt, json, srt, vtt, md)

        Returns:
            Formatted content, or None if not found
        """
        reconciliation_dir = self._get_reconciliation_dir(job_id)
        file_path = reconciliation_dir / f"output.{output_format}"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                content = f.read()

            logger.debug(f"Loaded formatted output for job {job_id} in {output_format} format")
            return content

        except Exception as e:
            logger.error(f"Error loading formatted output for job {job_id}: {str(e)}")
            return None

    def reconciled_result_exists(self, job_id: str) -> bool:
        """Check if reconciled result exists for a job"""
        reconciliation_dir = self._get_reconciliation_dir(job_id)
        file_path = reconciliation_dir / "reconciled.json"
        return file_path.exists()

    def has_reconciled_result(self, job_id: str) -> bool:
        """Alias for reconciled_result_exists for compatibility with batched reconciliation"""
        return self.reconciled_result_exists(job_id)

    def formatted_output_exists(self, job_id: str, output_format: str) -> bool:
        """Check if formatted output exists for a job"""
        reconciliation_dir = self._get_reconciliation_dir(job_id)
        file_path = reconciliation_dir / f"output.{output_format}"
        return file_path.exists()

    def delete_reconciled_result(self, job_id: str) -> bool:
        """Delete reconciled result for a job"""
        reconciliation_dir = self._get_reconciliation_dir(job_id)
        file_path = reconciliation_dir / "reconciled.json"

        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted reconciled result for job {job_id}")
            return True
        return False

    def delete_formatted_output(self, job_id: str, output_format: str) -> bool:
        """Delete formatted output for a job"""
        reconciliation_dir = self._get_reconciliation_dir(job_id)
        file_path = reconciliation_dir / f"output.{output_format}"

        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted formatted output for job {job_id} in {output_format} format")
            return True
        return False

    def get_job_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a job from the reconciled result file.
        
        Args:
            job_id: ID of the processing job
            
        Returns:
            Dictionary of metadata, or None if not found or no metadata exists
        """
        reconciliation_dir = self._get_reconciliation_dir(job_id)
        file_path = reconciliation_dir / "reconciled.json"
        
        if not file_path.exists():
            logger.debug(f"No reconciled result file found for job {job_id}")
            return {}
            
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                
            metadata = data.get("metadata", {})
            logger.debug(f"Loaded metadata for job {job_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata for job {job_id}: {str(e)}")
            return {}
