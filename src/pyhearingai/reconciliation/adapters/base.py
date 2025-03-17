"""
Base adapter interface for reconciliation services.

This module defines the abstract base class that all reconciliation adapters must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pyhearingai.core.idempotent import ProcessingJob
from pyhearingai.core.models import DiarizationSegment, Segment

logger = logging.getLogger(__name__)


class BaseReconciliationAdapter(ABC):
    """
    Abstract base class for reconciliation adapters.

    All reconciliation adapters must implement this interface to ensure
    interchangeability between different reconciliation strategies.
    """

    @abstractmethod
    def reconcile(
        self,
        job: ProcessingJob,
        diarization_segments: Dict[str, List[DiarizationSegment]],
        transcription_segments: Dict[str, List[Segment]],
        segment_transcriptions: Dict[str, str],
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Segment]:
        """
        Reconcile diarization and transcription data into coherent segments.

        Args:
            job: The processing job
            diarization_segments: Dictionary mapping chunk IDs to diarization segments
            transcription_segments: Dictionary mapping chunk IDs to transcription segments
            segment_transcriptions: Dictionary mapping segment IDs to transcribed text
            options: Optional settings for the reconciliation process

        Returns:
            List of reconciled segments with speaker information
        """
        pass
