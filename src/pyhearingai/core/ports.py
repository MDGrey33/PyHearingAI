"""
Core ports (interfaces) for the PyHearingAI system.

This module defines the abstract interfaces that adapters must implement
to provide transcription, diarization, and other services to the application.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyhearingai.core.models import DiarizationSegment, Segment, TranscriptionResult
from pyhearingai.core.domain.audio_quality import AudioQualitySpecification
from pyhearingai.core.domain.api_constraints import ApiProvider


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


class SizeAwareAudioConverter(AudioConverter):
    """Interface for audio converters with size constraint awareness."""
    
    @abstractmethod
    def convert_with_size_constraint(
        self, 
        audio_path: Path, 
        max_size_bytes: int,
        target_format: str = "wav",
        **kwargs
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Convert audio ensuring result is under max_size_bytes.
        
        Args:
            audio_path: Path to the audio file to convert
            max_size_bytes: Maximum size in bytes for the output file
            target_format: Target format to convert to (e.g., 'wav')
            **kwargs: Additional conversion options
        
        Returns:
            Tuple of (path_to_converted_file, metadata)
        
        Raises:
            ValueError: If conversion to target size is not possible
        """
        pass
    
    @abstractmethod
    def convert_with_quality_spec(
        self,
        audio_path: Path,
        quality_spec: AudioQualitySpecification,
        **kwargs
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Convert audio according to quality specification.
        
        Args:
            audio_path: Path to the audio file to convert
            quality_spec: Quality specification for the conversion
            **kwargs: Additional conversion options
        
        Returns:
            Tuple of (path_to_converted_file, metadata)
        """
        pass
    
    @abstractmethod
    def estimate_output_size(
        self,
        audio_path: Path,
        quality_spec: AudioQualitySpecification
    ) -> int:
        """
        Estimate the size of the output file after conversion.
        
        Args:
            audio_path: Path to the audio file to convert
            quality_spec: Quality specification for the conversion
        
        Returns:
            Estimated size in bytes
        """
        pass
    
    @abstractmethod
    def check_file_size(
        self,
        file_path: Path,
        max_size_bytes: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a file is within size constraints.
        
        Args:
            file_path: Path to the file to check
            max_size_bytes: Maximum allowed size in bytes
            
        Returns:
            Tuple of (is_within_limit, message)
        """
        pass
    
    @abstractmethod
    def check_api_compatibility(
        self,
        file_path: Path,
        api_provider: ApiProvider
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a file is compatible with a specific API provider.
        
        Args:
            file_path: Path to the file to check
            api_provider: Target API provider
            
        Returns:
            Tuple of (is_compatible, message)
        """
        pass


class AudioFormatService(ABC):
    """Interface for audio format operations and metadata retrieval."""
    
    @abstractmethod
    def get_audio_metadata(self, audio_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary of metadata
        """
        pass
    
    @abstractmethod
    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        pass
    
    @abstractmethod
    def extract_audio_segment(
        self,
        audio_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
        quality_spec: Optional[AudioQualitySpecification] = None
    ) -> Path:
        """
        Extract a segment of audio from a file.
        
        Args:
            audio_path: Path to the audio file
            output_path: Path to save the extracted segment
            start_time: Start time in seconds
            end_time: End time in seconds
            quality_spec: Quality specification for the output
            
        Returns:
            Path to the extracted segment
        """
        pass
    
    @abstractmethod
    def detect_silence(
        self,
        audio_path: Path,
        min_silence_duration: float = 0.5,
        silence_threshold: float = -40
    ) -> List[Dict[str, float]]:
        """
        Detect silence regions in an audio file.
        
        Args:
            audio_path: Path to the audio file
            min_silence_duration: Minimum silence duration in seconds
            silence_threshold: Silence threshold in dB
            
        Returns:
            List of dictionaries with 'start' and 'end' keys for each silence region
        """
        pass


class ChunkingService(ABC):
    """Interface for audio chunking services."""
    
    @abstractmethod
    def calculate_chunk_boundaries(
        self,
        audio_duration: float,
        chunk_duration: float,
        overlap_duration: float = 0.0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """
        Calculate chunk boundaries for audio.
        
        Args:
            audio_duration: Total audio duration in seconds
            chunk_duration: Target chunk duration in seconds
            overlap_duration: Overlap between chunks in seconds
            start_time: Start time offset in seconds (optional)
            end_time: End time limit in seconds (optional)
        
        Returns:
            List of chunk boundaries as (start_time, end_time) tuples
        """
        pass
    
    @abstractmethod
    def create_audio_chunks(
        self,
        audio_path: Path,
        output_dir: Path,
        chunk_boundaries: List[Tuple[float, float]],
        quality_spec: AudioQualitySpecification,
        api_provider: Optional[ApiProvider] = None,
        job_id: Optional[str] = None
    ) -> List[Path]:
        """
        Create audio chunks from chunk boundaries.
        
        Args:
            audio_path: Path to the original audio file
            output_dir: Directory to save chunks
            chunk_boundaries: List of (start_time, end_time) tuples
            quality_spec: Quality specification for chunks
            api_provider: Target API provider (for size validation)
            job_id: Optional job ID for event publishing
        
        Returns:
            List of paths to created chunks
        """
        pass
    
    @abstractmethod
    def detect_silence(
        self,
        audio_path: Path,
        min_silence_duration: float = 0.5,
        silence_threshold: float = -40
    ) -> List[Tuple[float, float]]:
        """
        Detect silence regions in audio.
        
        Args:
            audio_path: Path to the audio file
            min_silence_duration: Minimum silence duration in seconds
            silence_threshold: Silence threshold in dB
        
        Returns:
            List of (start_time, end_time) tuples for silence regions
        """
        pass


class Transcriber(ABC):
    """Interface for speech-to-text transcribers."""

    @abstractmethod
    def transcribe(self, audio_path: Path, **kwargs) -> List[Segment]:
        """
        Transcribe an audio file into text segments with timing information.

        Args:
            audio_path: Path to the audio file to transcribe
            **kwargs: Additional transcription options

        Returns:
            List of Segment objects containing the transcribed text and timing information
        """
        pass

    @abstractmethod
    def close(self):
        """Release any resources used by the transcriber."""
        pass


class Diarizer(ABC):
    """Interface for speaker diarization systems."""

    @abstractmethod
    def diarize(self, audio_path: Path, **kwargs) -> List[DiarizationSegment]:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file to diarize
            **kwargs: Additional diarization options

        Returns:
            List of DiarizationSegment objects containing speaker IDs and timing information
        """
        pass

    @abstractmethod
    def close(self):
        """Release any resources used by the diarizer."""
        pass


class SpeakerAssigner(ABC):
    """Interface for assigning speakers to transcript segments."""

    @abstractmethod
    def assign_speakers(
        self,
        transcript_segments: List[Segment],
        diarization_segments: List[DiarizationSegment],
        **kwargs
    ) -> List[Segment]:
        """
        Assign speakers to transcript segments based on diarization results.

        Args:
            transcript_segments: List of transcript segments
            diarization_segments: List of diarization segments
            **kwargs: Additional options

        Returns:
            List of transcript segments with speaker IDs assigned
        """
        pass

    @abstractmethod
    def close(self):
        """Release any resources used by the speaker assigner."""
        pass


class OutputFormatter(ABC):
    """Interface for formatting transcription results into different output formats."""

    @abstractmethod
    def format(self, result: TranscriptionResult) -> str:
        """
        Format a transcription result into a string.

        Args:
            result: The transcription result to format

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
