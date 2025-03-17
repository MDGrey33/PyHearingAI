"""
Audio validation domain services.

This module defines domain services for validating audio files against
quality specifications and API constraints, including calculation of
optimal chunk durations.
"""

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyhearingai.core.domain.api_constraints import ApiProvider, ApiSizeLimitPolicy
from pyhearingai.core.domain.audio_quality import AudioQualitySpecification

logger = logging.getLogger(__name__)


class AudioValidationService:
    """
    Domain service for validating audio files against quality specs and size limits.

    This service encapsulates the validation logic to ensure audio files
    (including chunks) meet the requirements before processing.
    """

    @staticmethod
    def validate_audio_file(
        file_path: Path, api_provider: ApiProvider
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate an audio file against API provider constraints.

        Args:
            file_path: Path to the audio file
            api_provider: Target API provider

        Returns:
            Tuple of (is_valid, error_message)
        """
        return ApiSizeLimitPolicy.validate_file_for_provider(file_path, api_provider)

    @staticmethod
    def estimate_optimal_chunk_duration(
        audio_path: Path,
        quality_spec: AudioQualitySpecification,
        api_provider: ApiProvider,
        safety_margin_percent: float = 5.0,
    ) -> float:
        """
        Calculate optimal chunk duration to meet API size constraints.

        Args:
            audio_path: Path to the original audio file
            quality_spec: Target quality specification for chunks
            api_provider: Target API provider
            safety_margin_percent: Percentage safety margin for size calculation

        Returns:
            Optimal chunk duration in seconds
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get API size limit
        size_limit = ApiSizeLimitPolicy.get_limit_for_provider(api_provider)

        # Apply safety margin
        effective_limit = int(size_limit.max_file_size_bytes * (1 - safety_margin_percent / 100))

        # Calculate bytes per second
        bytes_per_second = quality_spec.estimated_bytes_per_second()

        if bytes_per_second <= 0:
            logger.warning(
                f"Invalid bytes_per_second estimate: {bytes_per_second}. "
                "Using default chunk duration of 30 seconds."
            )
            return 30.0

        # Calculate optimal duration
        optimal_duration = effective_limit / bytes_per_second

        # Round down to nearest whole number for safety
        optimal_duration = math.floor(optimal_duration)

        logger.debug(
            f"Calculated optimal chunk duration: {optimal_duration}s "
            f"(limit: {effective_limit} bytes, rate: {bytes_per_second} bytes/s)"
        )

        # Enforce reasonable duration bounds
        if optimal_duration < 10:
            logger.warning(
                f"Calculated chunk duration is very short ({optimal_duration}s). "
                "This may result in many small chunks. Consider lowering quality."
            )
            # Still use at least 10 seconds
            return max(10.0, optimal_duration)

        if optimal_duration > 600:
            logger.info(
                f"Calculated chunk duration is very long ({optimal_duration}s). "
                "Limiting to 600 seconds for better processing."
            )
            return 600.0

        return float(optimal_duration)

    @staticmethod
    def validate_chunk_parameters(
        chunk_duration: float, overlap_duration: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate chunk parameters for reasonableness.

        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds

        Returns:
            Tuple of (is_valid, error_message)
        """
        if chunk_duration <= 0:
            return False, "Chunk duration must be positive"

        if overlap_duration < 0:
            return False, "Overlap duration cannot be negative"

        if overlap_duration >= chunk_duration:
            return (
                False,
                f"Overlap duration ({overlap_duration}s) must be less than "
                f"chunk duration ({chunk_duration}s)",
            )

        if chunk_duration < 1.0:
            return (
                False,
                f"Chunk duration ({chunk_duration}s) is too short. "
                "Minimum recommended duration is 1.0 seconds.",
            )

        # Warning threshold, but still valid
        if chunk_duration < 5.0:
            logger.warning(
                f"Chunk duration ({chunk_duration}s) is very short. "
                "This may result in many small chunks."
            )

        # All checks passed
        return True, None

    @staticmethod
    def suggest_quality_reduction(
        file_path: Path, current_spec: AudioQualitySpecification, api_provider: ApiProvider
    ) -> Optional[AudioQualitySpecification]:
        """
        Suggest reduced quality settings if a file exceeds size limits.

        Args:
            file_path: Path to the audio file
            current_spec: Current quality specification
            api_provider: Target API provider

        Returns:
            New quality specification or None if no adjustment is needed/possible
        """
        # Get API size limit
        size_limit = ApiSizeLimitPolicy.get_limit_for_provider(api_provider)

        # Check current file size
        if not file_path.exists():
            return None

        file_size = os.path.getsize(file_path)
        if file_size <= size_limit.max_file_size_bytes:
            # No reduction needed
            return None

        # Calculate how much reduction is needed
        reduction_factor = size_limit.max_file_size_bytes / file_size

        logger.info(
            f"File {file_path} exceeds size limit "
            f"({file_size} > {size_limit.max_file_size_bytes} bytes). "
            f"Need to reduce by factor of {reduction_factor:.2f}"
        )

        # Strategy 1: Reduce sample rate
        if current_spec.sample_rate > 16000:
            return current_spec.with_sample_rate(16000)

        # Strategy 2: Convert stereo to mono
        if current_spec.channels > 1:
            return AudioQualitySpecification(
                sample_rate=current_spec.sample_rate,
                channels=1,
                bit_depth=current_spec.bit_depth,
                format=current_spec.format,
                codec=current_spec.codec,
                max_size_bytes=current_spec.max_size_bytes,
            )

        # Strategy 3: Switch to MP3
        if current_spec.format != AudioQualitySpecification.AudioFormat.MP3:
            return AudioQualitySpecification(
                sample_rate=current_spec.sample_rate,
                channels=current_spec.channels,
                bit_depth=16,
                format=AudioQualitySpecification.AudioFormat.MP3,
                codec=AudioQualitySpecification.AudioCodec.MP3,
                quality=0.4,  # Low quality MP3
                max_size_bytes=current_spec.max_size_bytes,
            )

        # Strategy 4: Reduce MP3 quality if already MP3
        if (
            current_spec.format == AudioQualitySpecification.AudioFormat.MP3
            and current_spec.quality
            and current_spec.quality > 0.1
        ):
            return AudioQualitySpecification(
                sample_rate=current_spec.sample_rate,
                channels=current_spec.channels,
                bit_depth=current_spec.bit_depth,
                format=current_spec.format,
                codec=current_spec.codec,
                quality=0.1,  # Minimum quality MP3
                max_size_bytes=current_spec.max_size_bytes,
            )

        # No viable reduction strategy found
        logger.warning(
            f"No viable quality reduction found for file {file_path}. "
            "File will likely exceed size limit."
        )
        return None
