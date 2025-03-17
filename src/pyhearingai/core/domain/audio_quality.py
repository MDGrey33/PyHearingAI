"""
Audio quality specifications and related domain objects.

This module defines value objects and domain services related to audio quality
and conversion specifications, providing speech-optimized defaults and helper methods.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    M4A = "m4a"


class AudioCodec(Enum):
    """Supported audio codecs."""
    PCM_S16LE = "pcm_s16le"  # Standard 16-bit PCM (WAV)
    PCM_S24LE = "pcm_s24le"  # 24-bit PCM (WAV)
    PCM_F32LE = "pcm_f32le"  # 32-bit float PCM (WAV)
    ADPCM_MS = "adpcm_ms"    # MS ADPCM, compressed (WAV)
    MP3 = "libmp3lame"       # MP3 compression
    OPUS = "libopus"         # Opus compression (OGG)
    AAC = "aac"              # AAC compression (M4A)
    FLAC = "flac"            # FLAC compression


@dataclass(frozen=True)
class AudioQualitySpecification:
    """
    Value object representing audio quality specifications.
    
    This immutable object encapsulates all parameters related to audio quality,
    format, and encoding, with speech-optimized defaults and helper methods.
    """
    
    sample_rate: int = 16000  # 16kHz is optimal for speech recognition
    channels: int = 1         # Mono for speech recognition
    bit_depth: int = 16       # 16-bit is sufficient for speech
    format: AudioFormat = AudioFormat.WAV
    codec: AudioCodec = AudioCodec.PCM_S16LE
    quality: Optional[float] = None  # 0.0-1.0 for lossy formats
    max_size_bytes: int = 0  # 0 means no constraint
    
    @classmethod
    def for_whisper_api(cls) -> 'AudioQualitySpecification':
        """
        Create a quality specification optimized for OpenAI's Whisper API.
        
        Returns:
            AudioQualitySpecification optimized for Whisper API
        """
        return cls(
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            format=AudioFormat.WAV,
            codec=AudioCodec.PCM_S16LE,
            max_size_bytes=24_000_000  # 24MB (below OpenAI's 25MB limit with margin)
        )
    
    @classmethod
    def for_local_processing(cls) -> 'AudioQualitySpecification':
        """
        Create a quality specification optimized for local processing.
        
        Returns:
            AudioQualitySpecification optimized for local processing
        """
        return cls(
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            format=AudioFormat.WAV,
            codec=AudioCodec.PCM_S16LE
        )
    
    @classmethod
    def for_high_quality(cls) -> 'AudioQualitySpecification':
        """
        Create a high-quality specification.
        
        Returns:
            High-quality AudioQualitySpecification
        """
        return cls(
            sample_rate=44100,
            channels=2,
            bit_depth=24,
            format=AudioFormat.WAV,
            codec=AudioCodec.PCM_S24LE
        )
    
    @classmethod
    def for_compressed(cls) -> 'AudioQualitySpecification':
        """
        Create a compressed quality specification with good quality.
        
        Returns:
            Compressed AudioQualitySpecification
        """
        return cls(
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            format=AudioFormat.MP3,
            codec=AudioCodec.MP3,
            quality=0.6  # Medium quality MP3
        )
    
    def with_sample_rate(self, sample_rate: int) -> 'AudioQualitySpecification':
        """
        Create a new specification with an updated sample rate.
        
        Args:
            sample_rate: New sample rate in Hz
            
        Returns:
            New AudioQualitySpecification with updated sample rate
        """
        return AudioQualitySpecification(
            sample_rate=sample_rate,
            channels=self.channels,
            bit_depth=self.bit_depth,
            format=self.format,
            codec=self.codec,
            quality=self.quality,
            max_size_bytes=self.max_size_bytes
        )
    
    def with_channels(self, channels: int) -> 'AudioQualitySpecification':
        """
        Create a new specification with an updated channel count.
        
        Args:
            channels: New channel count (1 for mono, 2 for stereo)
            
        Returns:
            New AudioQualitySpecification with updated channels
        """
        return AudioQualitySpecification(
            sample_rate=self.sample_rate,
            channels=channels,
            bit_depth=self.bit_depth,
            format=self.format,
            codec=self.codec,
            quality=self.quality,
            max_size_bytes=self.max_size_bytes
        )
    
    def with_max_size(self, max_size_bytes: int) -> 'AudioQualitySpecification':
        """
        Create a new specification with a maximum size constraint.
        
        Args:
            max_size_bytes: Maximum size in bytes (0 for no constraint)
            
        Returns:
            New AudioQualitySpecification with updated size constraint
        """
        return AudioQualitySpecification(
            sample_rate=self.sample_rate,
            channels=self.channels,
            bit_depth=self.bit_depth,
            format=self.format,
            codec=self.codec,
            quality=self.quality,
            max_size_bytes=max_size_bytes
        )
    
    def estimated_bytes_per_second(self) -> int:
        """
        Estimate bytes per second based on the quality specification.
        
        Returns:
            Estimated bytes per second
        """
        # For PCM formats, calculation is straightforward
        if self.codec in [AudioCodec.PCM_S16LE, AudioCodec.PCM_S24LE, AudioCodec.PCM_F32LE]:
            bytes_per_sample = 0
            
            if self.codec == AudioCodec.PCM_S16LE:
                bytes_per_sample = 2  # 16 bits = 2 bytes
            elif self.codec == AudioCodec.PCM_S24LE:
                bytes_per_sample = 3  # 24 bits = 3 bytes
            elif self.codec == AudioCodec.PCM_F32LE:
                bytes_per_sample = 4  # 32 bits = 4 bytes
                
            return self.sample_rate * self.channels * bytes_per_sample
        
        # For compressed formats, use approximate bitrates
        elif self.codec == AudioCodec.MP3:
            # Approximate MP3 bitrates based on quality
            if self.quality is not None:
                # Scale between 32kbps and 320kbps based on quality
                bitrate = 32000 + self.quality * (320000 - 32000)
                return int(bitrate / 8)  # Convert bits to bytes
            else:
                # Default to 128kbps for MP3 if no quality specified
                return 16000  # 128kbps = 16000 bytes/s
        
        elif self.codec == AudioCodec.OPUS:
            # Approximate Opus bitrates
            if self.quality is not None:
                # Scale between 8kbps and 256kbps based on quality
                bitrate = 8000 + self.quality * (256000 - 8000)
                return int(bitrate / 8)  # Convert bits to bytes
            else:
                # Default to 64kbps for Opus if no quality specified
                return 8000  # 64kbps = 8000 bytes/s
        
        elif self.codec == AudioCodec.ADPCM_MS:
            # ADPCM is approximately 4:1 compression compared to PCM
            return int(self.sample_rate * self.channels * 2 / 4)
        
        elif self.codec == AudioCodec.FLAC:
            # FLAC is approximately 50-60% of PCM size
            return int(self.sample_rate * self.channels * self.bit_depth / 8 * 0.6)
        
        else:
            # Default fallback for unknown codecs
            # Base on bit depth for approximation
            return int(self.sample_rate * self.channels * self.bit_depth / 8)
    
    @classmethod
    def high_quality(cls) -> 'AudioQualitySpecification':
        """
        Create a high-quality specification.
        
        Returns:
            High-quality AudioQualitySpecification
        """
        return cls.for_high_quality() 