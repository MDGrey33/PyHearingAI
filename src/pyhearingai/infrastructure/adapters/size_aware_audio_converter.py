"""
Size-aware audio converter implementation.

This module provides a concrete implementation of the SizeAwareAudioConverter port
using FFmpeg for audio conversion with size constraints and quality adaptation.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ffmpeg

from pyhearingai.core.domain.api_constraints import ApiProvider, ApiSizeLimitPolicy
from pyhearingai.core.domain.audio_quality import AudioCodec, AudioFormat, AudioQualitySpecification
from pyhearingai.core.domain.events import (
    AudioConversionEvent,
    AudioSizeExceededEvent,
    EventPublisher,
)
from pyhearingai.core.ports import SizeAwareAudioConverter
from pyhearingai.infrastructure.audio_converter import FFmpegAudioConverter

logger = logging.getLogger(__name__)


class SizeAwareFFmpegConverter(FFmpegAudioConverter, SizeAwareAudioConverter):
    """
    Audio converter implementation with size constraint awareness.

    This converter extends the base FFmpegAudioConverter with the ability
    to adapt quality parameters to meet size constraints.
    """

    def convert_with_size_constraint(
        self, audio_path: Path, max_size_bytes: int, target_format: str = "wav", **kwargs
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Convert audio ensuring result is under max_size_bytes.

        Args:
            audio_path: Path to the audio file to convert
            max_size_bytes: Maximum size in bytes for the output file
            target_format: Target audio format
            **kwargs: Additional conversion parameters

        Returns:
            Tuple of (path to converted file, metadata dictionary)
        """
        # Enhanced logging for debugging file path issues
        logger.info(f"Converting {audio_path} with size constraint of {max_size_bytes} bytes")
        logger.info(f"File exists check: {audio_path.exists()}")
        logger.info(f"Absolute path: {audio_path.absolute()}")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get original file size
        original_size = os.path.getsize(audio_path)
        logger.info(f"Original file size: {original_size} bytes")

        # Try initial conversion with standard parameters
        initial_args = {
            "sample_rate": kwargs.get("sample_rate", 16000),
            "channels": kwargs.get("channels", 1),
            "output_dir": kwargs.get("output_dir"),
            "codec": kwargs.get("codec"),
        }

        # Conversion metadata
        metadata = {
            "original_size": original_size,
            "original_format": audio_path.suffix.lower().lstrip("."),
            "target_format": target_format,
            "adjustments_made": [],
        }

        # Create a temporary directory for size testing conversions
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory for conversion: {temp_dir}")
            initial_args["output_dir"] = temp_dir

            # Try initial conversion
            logger.info(f"Attempting initial conversion with {initial_args}")
            initial_output = super().convert(audio_path, target_format, **initial_args)
            logger.info(f"Initial conversion output path: {initial_output}")
            logger.info(f"Initial output exists check: {initial_output.exists()}")

            # Check if result meets size constraint
            result_size = os.path.getsize(initial_output)
            logger.info(f"Initial conversion size: {result_size} bytes")

            if result_size <= max_size_bytes:
                # Size constraint met, copy to final location if needed
                logger.info(
                    f"Initial conversion meets size constraint ({result_size} <= {max_size_bytes})"
                )

                final_output = initial_output
                if kwargs.get("output_dir"):
                    output_dir = kwargs["output_dir"]
                    logger.info(f"Copying to final output directory: {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)
                    final_name = os.path.join(output_dir, initial_output.name)
                    import shutil

                    shutil.copy2(initial_output, final_name)
                    final_output = Path(final_name)
                    logger.info(f"Final output path: {final_output}")
                    logger.info(f"Final output exists check: {final_output.exists()}")
                else:
                    # No output_dir specified, copy to same directory as input file
                    # with a different name to avoid temporary file deletion
                    import shutil

                    output_dir = audio_path.parent
                    final_name = os.path.join(output_dir, f"converted_{initial_output.name}")
                    shutil.copy2(initial_output, final_name)
                    final_output = Path(final_name)
                    logger.info(f"Copied to final path (no output_dir): {final_output}")
                    logger.info(f"Final output exists check: {final_output.exists()}")

                # Add metadata
                metadata["converted_size"] = result_size
                metadata["compression_ratio"] = original_size / result_size

                # Publish conversion event
                EventPublisher.publish(
                    AudioConversionEvent(
                        source_path=audio_path,
                        target_path=final_output,
                        is_successful=True,
                        quality_spec=AudioQualitySpecification(
                            sample_rate=initial_args["sample_rate"],
                            channels=initial_args["channels"],
                            format=AudioFormat(target_format)
                            if hasattr(AudioFormat, target_format.upper())
                            else None,
                            codec=initial_args.get("codec"),
                        ),
                        original_size=original_size,
                        converted_size=result_size,
                        metadata=metadata,
                    )
                )

                return final_output, metadata

            # Initial conversion exceeds size limit, try adjustments
            logger.info(
                f"Initial conversion exceeds size limit ({result_size} > {max_size_bytes}). "
                "Attempting quality adjustments."
            )

            # Try progressive adjustments
            adjustments = [
                # Try lower sample rate (if not already 8kHz)
                {"sample_rate": 8000} if initial_args["sample_rate"] > 8000 else None,
                # Try different codec with better compression for WAV
                {"codec": "adpcm_ms"} if target_format == "wav" else None,
                # For lossy formats, try lower quality/bitrate
                {"codec": "libmp3lame", "bitrate": "32k"} if target_format == "mp3" else None,
                {"codec": "libopus", "bitrate": "16k"} if target_format == "ogg" else None,
            ]

            # Filter out None adjustments
            adjustments = [adj for adj in adjustments if adj is not None]

            # Try additional format-specific strategies
            if target_format == "wav":
                # Add more aggressive WAV compression options
                adjustments.extend(
                    [
                        {"codec": "adpcm_ms", "sample_rate": 8000},
                        {"codec": "adpcm_ms", "sample_rate": 8000, "channels": 1},
                    ]
                )
            elif target_format == "mp3":
                # Add more aggressive MP3 compression options
                adjustments.extend(
                    [
                        {"codec": "libmp3lame", "bitrate": "24k"},
                        {"codec": "libmp3lame", "bitrate": "16k"},
                        {"codec": "libmp3lame", "bitrate": "8k"},
                    ]
                )

            # Try each adjustment
            for i, adjustment in enumerate(adjustments):
                logger.debug(f"Trying adjustment {i+1}/{len(adjustments)}: {adjustment}")

                # Apply adjustment to initial args
                adjusted_args = dict(initial_args)
                adjusted_args.update(adjustment)
                metadata["adjustments_made"].append(adjustment)

                try:
                    # Convert with adjusted parameters
                    adjusted_output = super().convert(audio_path, target_format, **adjusted_args)

                    # Check size
                    result_size = os.path.getsize(adjusted_output)

                    if result_size <= max_size_bytes:
                        # Size constraint met, copy to final location if needed
                        logger.info(
                            f"Adjustment {i+1} meets size constraint "
                            f"({result_size} <= {max_size_bytes})"
                        )

                        final_output = adjusted_output
                        if kwargs.get("output_dir"):
                            os.makedirs(kwargs["output_dir"], exist_ok=True)
                            final_name = os.path.join(kwargs["output_dir"], adjusted_output.name)
                            import shutil

                            shutil.copy2(adjusted_output, final_name)
                            final_output = Path(final_name)

                        # Add metadata
                        metadata["converted_size"] = result_size
                        metadata["compression_ratio"] = original_size / result_size
                        metadata["successful_adjustment"] = adjustment

                        # Publish conversion event
                        EventPublisher.publish(
                            AudioConversionEvent(
                                source_path=audio_path,
                                target_path=final_output,
                                is_successful=True,
                                quality_spec=AudioQualitySpecification(
                                    sample_rate=adjusted_args.get("sample_rate", 16000),
                                    channels=adjusted_args.get("channels", 1),
                                    format=AudioFormat(target_format)
                                    if hasattr(AudioFormat, target_format.upper())
                                    else None,
                                    codec=adjusted_args.get("codec"),
                                ),
                                original_size=original_size,
                                converted_size=result_size,
                                metadata=metadata,
                            )
                        )

                        return final_output, metadata

                except Exception as e:
                    logger.warning(f"Adjustment {i+1} failed: {str(e)}")
                    continue

            # If we get here, all adjustments failed to meet the size constraint

            # Last resort: try converting to MP3 regardless of target format
            if target_format != "mp3":
                logger.info(
                    "All format-specific adjustments failed. " "Trying MP3 format as last resort."
                )

                try:
                    # Convert to MP3 with minimum bitrate
                    mp3_args = {
                        "sample_rate": 8000,
                        "channels": 1,
                        "codec": "libmp3lame",
                        "bitrate": "8k",
                        "output_dir": temp_dir,
                    }

                    mp3_output = super().convert(audio_path, "mp3", **mp3_args)
                    result_size = os.path.getsize(mp3_output)

                    if result_size <= max_size_bytes:
                        # Size constraint met with MP3
                        final_output = mp3_output
                        if kwargs.get("output_dir"):
                            os.makedirs(kwargs["output_dir"], exist_ok=True)
                            final_name = os.path.join(kwargs["output_dir"], mp3_output.name)
                            import shutil

                            shutil.copy2(mp3_output, final_name)
                            final_output = Path(final_name)

                        # Add metadata
                        metadata["converted_size"] = result_size
                        metadata["compression_ratio"] = original_size / result_size
                        metadata["format_changed"] = True
                        metadata["original_target_format"] = target_format
                        metadata["final_format"] = "mp3"

                        # Publish conversion event
                        EventPublisher.publish(
                            AudioConversionEvent(
                                source_path=audio_path,
                                target_path=final_output,
                                is_successful=True,
                                quality_spec=AudioQualitySpecification(
                                    sample_rate=8000,
                                    channels=1,
                                    format=AudioFormat.MP3,
                                    codec=AudioCodec.MP3,
                                ),
                                original_size=original_size,
                                converted_size=result_size,
                                metadata=metadata,
                            )
                        )

                        # Warn about format change
                        logger.warning(
                            f"Changed format from {target_format} to mp3 to meet size constraint. "
                            f"Final size: {result_size} bytes"
                        )

                        return final_output, metadata
                except Exception as e:
                    logger.warning(f"MP3 conversion failed: {str(e)}")

            # All attempts failed
            # Publish size exceeded event
            EventPublisher.publish(
                AudioSizeExceededEvent(
                    source_path=audio_path,
                    target_size_bytes=max_size_bytes,
                    best_achieved_size=result_size,
                    adjustments_tried=metadata["adjustments_made"],
                )
            )

            # Raise an error with the failure details
            raise ValueError(
                f"Unable to convert {audio_path} to meet size constraint of {max_size_bytes} bytes. "
                f"Best achieved size: {result_size} bytes after {len(metadata['adjustments_made'])} adjustments."
            )

    def convert_with_quality_spec(
        self, audio_path: Path, quality_spec: AudioQualitySpecification, **kwargs
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
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get original file size
        original_size = os.path.getsize(audio_path)

        # Convert string enum values to strings
        format_str = (
            quality_spec.format.value
            if isinstance(quality_spec.format, AudioFormat)
            else quality_spec.format
        )
        codec_str = (
            quality_spec.codec.value
            if isinstance(quality_spec.codec, AudioCodec)
            else quality_spec.codec
        )

        # Prepare conversion arguments
        conversion_args = {
            "sample_rate": quality_spec.sample_rate,
            "channels": quality_spec.channels,
            "codec": codec_str,
            "output_dir": kwargs.get("output_dir"),
        }

        # For MP3 and other lossy formats, set quality if specified
        if quality_spec.quality is not None:
            if format_str in ["mp3", "ogg"]:
                # Convert 0.0-1.0 quality to appropriate bitrate (8k-320k for MP3)
                min_bitrate = 8
                max_bitrate = 320
                bitrate = min_bitrate + quality_spec.quality * (max_bitrate - min_bitrate)
                conversion_args["bitrate"] = f"{int(bitrate)}k"

        # Check if size constraint is specified
        if quality_spec.max_size_bytes > 0:
            return self.convert_with_size_constraint(
                audio_path, quality_spec.max_size_bytes, format_str, **conversion_args
            )

        # Perform conversion without size constraint
        logger.debug(f"Converting {audio_path} with quality spec: {quality_spec}")
        output_path = super().convert(audio_path, format_str, **conversion_args)

        # Get converted file size
        converted_size = os.path.getsize(output_path)

        # Prepare metadata
        metadata = {
            "original_size": original_size,
            "converted_size": converted_size,
            "compression_ratio": original_size / converted_size if converted_size > 0 else 0,
            "original_format": audio_path.suffix.lower().lstrip("."),
            "target_format": format_str,
            "sample_rate": quality_spec.sample_rate,
            "channels": quality_spec.channels,
            "bit_depth": quality_spec.bit_depth,
        }

        # Publish conversion event
        EventPublisher.publish(
            AudioConversionEvent(
                source_path=audio_path,
                target_path=output_path,
                is_successful=True,
                quality_spec=quality_spec,
                original_size=original_size,
                converted_size=converted_size,
                metadata=metadata,
            )
        )

        return output_path, metadata

    def estimate_output_size(
        self, audio_path: Path, quality_spec: AudioQualitySpecification
    ) -> int:
        """
        Estimate the size of the output file after conversion.

        Args:
            audio_path: Path to the audio file to convert
            quality_spec: Quality specification for the conversion

        Returns:
            Estimated size in bytes
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get audio duration using ffprobe
        try:
            probe = ffmpeg.probe(str(audio_path))
            audio_info = next(
                stream for stream in probe["streams"] if stream["codec_type"] == "audio"
            )
            duration = float(audio_info.get("duration", 0))
        except Exception as e:
            logger.warning(f"Error probing audio file: {str(e)}")

            # Fallback: try getting duration using our own method
            try:
                # Approximate duration from file size and bit rate
                file_size = os.path.getsize(audio_path)
                bit_rate = audio_info.get("bit_rate", 128000)  # Default to 128kbps
                duration = file_size * 8 / float(bit_rate)
            except Exception:
                logger.error(f"Unable to determine audio duration for {audio_path}")
                # Default to a reasonable value
                duration = 60.0  # Assume 1 minute

        # Calculate estimated bytes per second based on quality spec
        bytes_per_second = quality_spec.estimated_bytes_per_second()

        # Estimate total size
        estimated_size = int(bytes_per_second * duration)

        logger.debug(
            f"Estimated output size for {audio_path}: {estimated_size} bytes "
            f"(duration: {duration}s, rate: {bytes_per_second} bytes/s)"
        )

        return estimated_size

    def check_file_size(self, file_path: Path, max_size_bytes: int) -> Tuple[bool, Optional[str]]:
        """
        Check if a file is within size constraints.

        Args:
            file_path: Path to the file to check
            max_size_bytes: Maximum allowed size in bytes

        Returns:
            Tuple of (is_within_limit, message)
        """
        if not file_path.exists():
            return False, f"File not found: {file_path}"

        file_size = os.path.getsize(file_path)
        if file_size > max_size_bytes:
            return (
                False,
                f"File size {file_size} bytes exceeds limit of {max_size_bytes} bytes "
                f"by {file_size - max_size_bytes} bytes ({(file_size / max_size_bytes - 1) * 100:.1f}%)",
            )

        return True, None

    def check_api_compatibility(
        self, file_path: Path, api_provider: ApiProvider
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a file is compatible with a specific API provider.

        Args:
            file_path: Path to the file to check
            api_provider: Target API provider

        Returns:
            Tuple of (is_compatible, message)
        """
        return ApiSizeLimitPolicy.validate_file_for_provider(file_path, api_provider)
        return ApiSizeLimitPolicy.validate_file_for_provider(file_path, api_provider)
