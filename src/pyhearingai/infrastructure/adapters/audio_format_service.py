"""
Audio format service implementation.

This module provides a concrete implementation of the AudioFormatService interface
using FFmpeg for audio metadata extraction and segment manipulation.
"""

import logging
import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import ffmpeg

from pyhearingai.core.ports import AudioFormatService
from pyhearingai.core.domain.audio_quality import AudioQualitySpecification

logger = logging.getLogger(__name__)


class FFmpegAudioFormatService(AudioFormatService):
    """
    FFmpeg-based implementation of the AudioFormatService.
    
    This service implements audio metadata extraction and segment manipulation
    using FFmpeg, without handling conversion (which is the responsibility of
    the AudioConverter).
    """
    
    def get_audio_metadata(self, audio_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary of metadata
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            RuntimeError: If metadata extraction fails
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Use ffprobe to get metadata
            probe = ffmpeg.probe(str(audio_path))
            
            # Extract audio stream metadata
            audio_info = next((stream for stream in probe["streams"] 
                              if stream["codec_type"] == "audio"), None)
            
            if not audio_info:
                raise RuntimeError(f"No audio stream found in {audio_path}")
            
            # Build metadata dictionary
            metadata = {
                "duration": float(probe["format"].get("duration", 0)),
                "size_bytes": int(probe["format"].get("size", 0)),
                "format": probe["format"].get("format_name", ""),
                "sample_rate": int(audio_info.get("sample_rate", 0)),
                "channels": int(audio_info.get("channels", 0)),
                "codec": audio_info.get("codec_name", ""),
                "bit_rate": int(audio_info.get("bit_rate", 0)) if "bit_rate" in audio_info else None,
                "bits_per_sample": int(audio_info.get("bits_per_sample", 0)) if "bits_per_sample" in audio_info else None,
            }
            
            # Add any additional tags
            if "tags" in audio_info:
                metadata["tags"] = audio_info["tags"]
                
            return metadata
            
        except ffmpeg.Error as e:
            error_message = f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            raise RuntimeError(f"Failed to extract metadata: {str(e)}") from e
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration in seconds
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            RuntimeError: If duration extraction fails
        """
        try:
            metadata = self.get_audio_metadata(audio_path)
            return metadata["duration"]
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            raise RuntimeError(f"Failed to get audio duration: {str(e)}") from e
    
    def extract_audio_segment(
        self,
        audio_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
        quality_spec: Optional[AudioQualitySpecification] = None
    ) -> Path:
        """
        Extract a segment of audio and save to a new file.
        
        Args:
            audio_path: Path to the source audio file
            output_path: Path to save the extracted segment
            start_time: Start time in seconds
            end_time: End time in seconds
            quality_spec: Optional quality specification for the output
            
        Returns:
            Path to the extracted segment
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If start_time >= end_time
            RuntimeError: If extraction fails
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if start_time >= end_time:
            raise ValueError(f"Invalid time range: start ({start_time}) >= end ({end_time})")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Determine output format based on file extension
        output_format = output_path.suffix.lower().lstrip('.')
        
        # Prepare quality parameters
        if quality_spec:
            sample_rate = quality_spec.sample_rate
            channels = quality_spec.channels
            codec = quality_spec.codec.value if hasattr(quality_spec.codec, 'value') else None
        else:
            # Default parameters for speech
            sample_rate = 16000
            channels = 1
            codec = None
        
        try:
            # Build FFmpeg command
            input_stream = ffmpeg.input(str(audio_path), ss=start_time, to=end_time)
            
            # Build output options
            output_args = {"ar": sample_rate, "ac": channels}
            
            # Add codec if specified
            if codec:
                output_args["codec:a"] = codec
            
            # For MP3 and other lossy formats, add quality settings if specified
            if quality_spec and quality_spec.quality is not None and output_format in ["mp3", "ogg"]:
                if output_format == "mp3":
                    # Convert 0.0-1.0 quality to appropriate bitrate (8k-320k for MP3)
                    min_bitrate = 8
                    max_bitrate = 320
                    bitrate = min_bitrate + quality_spec.quality * (max_bitrate - min_bitrate)
                    output_args["b:a"] = f"{int(bitrate)}k"
            
            # Run the extraction
            logger.debug(
                f"Extracting segment from {audio_path} "
                f"({start_time}s to {end_time}s) to {output_path}"
            )
            
            (
                input_stream.output(str(output_path), **output_args)
                .overwrite_output()
                .run(quiet=True)
            )
            
            logger.debug(f"Segment extraction complete: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            error_message = f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e
        except Exception as e:
            logger.error(f"Error extracting audio segment: {str(e)}")
            raise RuntimeError(f"Failed to extract audio segment: {str(e)}") from e
    
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
            silence_threshold: Silence threshold in dB (lower values detect more silence)
            
        Returns:
            List of dictionaries with 'start' and 'end' keys for each silence region
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            RuntimeError: If silence detection fails
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create a temporary file for the silence detection output
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_json = temp_file.name
        
        try:
            # Prepare the FFmpeg command for silence detection
            # Convert threshold from dB to amplitude ratio (10^(dB/20))
            # and min_silence_duration to milliseconds
            min_silence_ms = int(min_silence_duration * 1000)
            
            # Build the silence detect filter string
            silence_filter = (
                f"silencedetect=noise={silence_threshold}dB:d={min_silence_duration}"
            )
            
            # Build the full FFmpeg command
            cmd = [
                "ffmpeg",
                "-i", str(audio_path),
                "-af", silence_filter,
                "-f", "null",
                "-"
            ]
            
            # Run the command and capture stderr where silence detection info is printed
            logger.debug(f"Detecting silence in {audio_path}")
            result = subprocess.run(
                cmd, stderr=subprocess.PIPE, text=True, check=True
            )
            
            # Parse the output to extract silence regions
            silence_output = result.stderr
            silence_regions = []
            
            # Example output: "[silencedetect @ 0x7f8d5c00] silence_start: 1.23"
            # followed by "[silencedetect @ 0x7f8d5c00] silence_end: 3.45 | silence_duration: 2.22"
            
            current_start = None
            for line in silence_output.split('\n'):
                if "silence_start" in line:
                    # Extract start time
                    current_start = float(line.split("silence_start:")[1].strip())
                elif "silence_end" in line and current_start is not None:
                    # Extract end time and duration
                    end_time = float(line.split("silence_end:")[1].split('|')[0].strip())
                    
                    # Add to regions
                    silence_regions.append({"start": current_start, "end": end_time})
                    
                    # Reset start
                    current_start = None
            
            logger.debug(f"Detected {len(silence_regions)} silence regions")
            return silence_regions
            
        except subprocess.CalledProcessError as e:
            error_message = f"FFmpeg silence detection error: {e.stderr}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e
        except Exception as e:
            logger.error(f"Error detecting silence: {str(e)}")
            raise RuntimeError(f"Failed to detect silence: {str(e)}") from e
        finally:
            # Clean up temp file
            if os.path.exists(output_json):
                os.unlink(output_json) 