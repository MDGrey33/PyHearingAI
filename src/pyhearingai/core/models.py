"""
Core domain models for PyHearingAI.

This module contains the entity classes representing the core domain concepts
of the transcription system.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any


@dataclass
class Segment:
    """
    A segment of audio with transcribed text and timing information.
    """
    text: str
    start: float  # start time in seconds
    end: float  # end time in seconds
    speaker_id: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Get the duration of the segment in seconds."""
        return self.end - self.start
    
    @property
    def start_timecode(self) -> str:
        """Get the start time as a timecode string (HH:MM:SS.mmm)."""
        return str(timedelta(seconds=self.start)).rstrip('0').rstrip('.')
    
    @property
    def end_timecode(self) -> str:
        """Get the end time as a timecode string (HH:MM:SS.mmm)."""
        return str(timedelta(seconds=self.end)).rstrip('0').rstrip('.')


@dataclass
class DiarizationSegment:
    """
    A segment identified by a diarization system, with speaker ID and timing.
    """
    speaker_id: str
    start: float  # start time in seconds
    end: float  # end time in seconds
    score: Optional[float] = None


@dataclass
class TranscriptionResult:
    """
    The complete result of a transcription with speaker diarization.
    """
    segments: List[Segment] = field(default_factory=list)
    audio_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_text(self) -> str:
        """Get the complete transcript as a single text string."""
        return ' '.join(segment.text for segment in self.segments)
    
    @property
    def duration(self) -> float:
        """Get the total duration of the audio in seconds."""
        if not self.segments:
            return 0.0
        return max(segment.end for segment in self.segments)
    
    def save(self, path: Union[str, Path], format: str = "txt") -> Path:
        """
        Save the transcription to a file in the specified format.
        
        Args:
            path: The path to save the file to
            format: The format to save as (txt, json, srt, vtt, md)
            
        Returns:
            The path to the saved file
        """
        from pyhearingai.application.outputs import save_transcript
        return save_transcript(self, path, format) 