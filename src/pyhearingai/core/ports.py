"""
Core ports (interfaces) for the PyHearingAI system.

This module defines the abstract interfaces that adapters must implement
to provide transcription, diarization, and other services to the application.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any

from pyhearingai.core.models import DiarizationSegment, Segment, TranscriptionResult


class AudioConverter(ABC):
    """Interface for converting audio files to a format suitable for processing."""
    
    @abstractmethod
    def convert(self, audio_path: Path, target_format: str = "wav", **kwargs) -> Path:
        """
        Convert an audio file to the specified format.
        
        Args:
            audio_path: Path to the audio file to convert
            target_format: Target format to convert to (e.g., 'wav')
            **kwargs: Additional conversion options
            
        Returns:
            Path to the converted audio file
        """
        pass


class Transcriber(ABC):
    """Interface for speech-to-text transcription services."""
    
    @abstractmethod
    def transcribe(self, audio_path: Path, **kwargs) -> List[Segment]:
        """
        Transcribe speech in an audio file to text.
        
        Args:
            audio_path: Path to the audio file to transcribe
            **kwargs: Additional transcription options
            
        Returns:
            List of segments with transcribed text and timing information
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the transcriber."""
        pass
    
    @property
    @abstractmethod
    def supports_segmentation(self) -> bool:
        """Whether this transcriber provides timing and segmentation."""
        pass


class Diarizer(ABC):
    """Interface for speaker diarization services."""
    
    @abstractmethod
    def diarize(self, audio_path: Path, **kwargs) -> List[DiarizationSegment]:
        """
        Perform speaker diarization on an audio file.
        
        Args:
            audio_path: Path to the audio file to diarize
            **kwargs: Additional diarization options
            
        Returns:
            List of segments with speaker identification and timing information
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the diarizer."""
        pass


class SpeakerAssigner(ABC):
    """Interface for merging transcription and diarization results."""
    
    @abstractmethod
    def assign_speakers(
        self, 
        transcript_segments: List[Segment], 
        diarization_segments: List[DiarizationSegment],
        **kwargs
    ) -> List[Segment]:
        """
        Assign speaker IDs to transcript segments based on diarization data.
        
        Args:
            transcript_segments: List of transcript segments
            diarization_segments: List of diarization segments
            **kwargs: Additional options
            
        Returns:
            List of transcript segments with speaker IDs assigned
        """
        pass


class OutputFormatter(ABC):
    """Interface for formatting and saving transcription results."""
    
    @abstractmethod
    def format(self, result: TranscriptionResult, **kwargs) -> str:
        """
        Format a transcription result as a string.
        
        Args:
            result: The transcription result to format
            **kwargs: Additional formatting options
            
        Returns:
            Formatted string representation
        """
        pass
    
    @abstractmethod
    def save(self, result: TranscriptionResult, path: Path, **kwargs) -> Path:
        """
        Save a transcription result to a file.
        
        Args:
            result: The transcription result to save
            path: Path to save the file to
            **kwargs: Additional saving options
            
        Returns:
            Path to the saved file
        """
        pass
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Get the name of the format."""
        pass 