"""
Speaker assignment implementation.

This module provides a concrete implementation of the SpeakerAssigner port
to merge transcription and diarization results.
"""

import logging
from typing import Dict, List

from pyhearingai.core.models import DiarizationSegment, Segment
from pyhearingai.core.ports import SpeakerAssigner
from pyhearingai.infrastructure.registry import register_speaker_assigner

logger = logging.getLogger(__name__)


@register_speaker_assigner
class DefaultSpeakerAssigner(SpeakerAssigner):
    """
    Default implementation of speaker assignment.

    This implementation assigns speakers to transcript segments based on
    the maximum overlap between transcript segments and diarization segments.
    """

    def assign_speakers(
        self,
        transcript_segments: List[Segment],
        diarization_segments: List[DiarizationSegment],
        **kwargs,
    ) -> List[Segment]:
        """
        Assign speaker IDs to transcript segments based on diarization data.

        Args:
            transcript_segments: List of transcript segments
            diarization_segments: List of diarization segments
            **kwargs: Additional options
                - min_overlap: Minimum overlap ratio required to assign a speaker (default: 0.5)
                - speaker_prefix: Prefix to add to speaker IDs (default: "Speaker ")
                - normalize_speakers: Whether to normalize speaker IDs to consecutive numbers (default: True)

        Returns:
            List of transcript segments with speaker IDs assigned
        """
        if not transcript_segments:
            logger.warning("No transcript segments provided")
            return []

        if not diarization_segments:
            logger.warning("No diarization segments provided")
            return transcript_segments

        # Extract options
        min_overlap = kwargs.get("min_overlap", 0.5)
        speaker_prefix = kwargs.get("speaker_prefix", "Speaker ")
        normalize_speakers = kwargs.get("normalize_speakers", True)

        # Sort segments by start time to ensure correct order
        transcript_segments = sorted(transcript_segments, key=lambda s: s.start)
        diarization_segments = sorted(diarization_segments, key=lambda s: s.start)

        # Normalize speaker IDs if requested
        if normalize_speakers:
            # Create a mapping from original speaker IDs to normalized ones
            unique_speakers = sorted(set(s.speaker_id for s in diarization_segments))
            speaker_map = {spk: f"{speaker_prefix}{i}" for i, spk in enumerate(unique_speakers)}

            # Apply the mapping to diarization segments
            for segment in diarization_segments:
                segment.speaker_id = speaker_map.get(segment.speaker_id, segment.speaker_id)

        # Assign speakers to transcript segments
        result_segments = []
        for transcript_segment in transcript_segments:
            # Find the diarization segment with the most overlap
            best_speaker = None
            best_overlap = 0

            for diarization_segment in diarization_segments:
                # Calculate overlap between segments
                overlap_start = max(transcript_segment.start, diarization_segment.start)
                overlap_end = min(transcript_segment.end, diarization_segment.end)

                if overlap_start < overlap_end:
                    # There is some overlap
                    overlap_duration = overlap_end - overlap_start
                    transcript_duration = transcript_segment.end - transcript_segment.start

                    # Calculate overlap ratio relative to transcript segment
                    overlap_ratio = overlap_duration / transcript_duration

                    # Update best match if this overlap is better
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_speaker = diarization_segment.speaker_id

            # Create a new segment with the assigned speaker
            if best_overlap >= min_overlap:
                # Only assign if overlap exceeds minimum threshold
                new_segment = Segment(
                    text=transcript_segment.text,
                    start=transcript_segment.start,
                    end=transcript_segment.end,
                    speaker_id=best_speaker,
                )
            else:
                # Keep original segment without speaker ID
                new_segment = transcript_segment

            result_segments.append(new_segment)

        return result_segments
