"""
ChunkingService implementation.

This module provides a concrete implementation of the ChunkingService interface,
focusing on calculating optimal chunk boundaries and coordinating with other services
for audio processing operations.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyhearingai.core.domain.api_constraints import ApiProvider, ApiSizeLimitPolicy
from pyhearingai.core.domain.audio_quality import AudioQualitySpecification
from pyhearingai.core.domain.audio_validation import AudioValidationService
from pyhearingai.core.domain.events import AudioSizeExceededEvent, ChunkingEvent, EventPublisher
from pyhearingai.core.ports import AudioFormatService, ChunkingService
from pyhearingai.infrastructure.adapters.audio_format_service import FFmpegAudioFormatService
from pyhearingai.infrastructure.adapters.size_aware_audio_converter import SizeAwareFFmpegConverter

logger = logging.getLogger(__name__)


class ChunkingServiceImpl(ChunkingService):
    """
    Implementation of the ChunkingService interface.

    This service handles logical chunking operations, focusing on calculating
    optimal chunk boundaries and delegating actual audio processing to specialized services.
    """

    def __init__(
        self,
        audio_format_service: Optional[AudioFormatService] = None,
        audio_converter: Optional[SizeAwareFFmpegConverter] = None,
    ):
        """
        Initialize the chunking service.

        Args:
            audio_format_service: Service for audio metadata and segment extraction
            audio_converter: Converter for audio format conversion
        """
        self.audio_format_service = audio_format_service or FFmpegAudioFormatService()
        self.audio_converter = audio_converter or SizeAwareFFmpegConverter()

    def calculate_chunk_boundaries(
        self,
        audio_duration: float,
        chunk_duration: float,
        overlap_duration: float = 0.0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
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

        Raises:
            ValueError: If chunk parameters are invalid
        """
        # Validate chunk parameters
        validation_result, error_message = AudioValidationService.validate_chunk_parameters(
            chunk_duration, overlap_duration
        )

        if not validation_result:
            raise ValueError(f"Invalid chunk parameters: {error_message}")

        # Apply time range constraints if specified
        effective_start = start_time if start_time is not None else 0.0
        effective_end = end_time if end_time is not None else audio_duration

        # Validate time range
        if effective_start < 0:
            effective_start = 0.0
            logger.warning(f"Negative start time corrected to 0.0")

        if effective_end > audio_duration:
            effective_end = audio_duration
            logger.warning(f"End time exceeding audio duration corrected to {audio_duration}")

        if effective_start >= effective_end:
            raise ValueError(
                f"Invalid time range: start ({effective_start}) >= end ({effective_end})"
            )

        # Calculate effective duration for chunking
        effective_duration = effective_end - effective_start

        # Handle very short audio files (shorter than chunk duration)
        if effective_duration <= chunk_duration:
            logger.debug(
                f"Audio duration ({effective_duration}s) <= chunk duration ({chunk_duration}s), "
                "creating single chunk"
            )
            return [(effective_start, effective_end)]

        # Calculate number of chunks and adjust overlap if needed
        if overlap_duration >= chunk_duration:
            logger.warning(
                f"Overlap ({overlap_duration}s) >= chunk duration ({chunk_duration}s), "
                "reducing to half chunk duration"
            )
            overlap_duration = chunk_duration / 2

        # Calculate chunk boundaries with overlap
        chunk_boundaries = []
        current_pos = effective_start

        # Set a safety limit to prevent infinite loops
        max_chunks = int(effective_duration / (chunk_duration - overlap_duration) * 2)

        while current_pos < effective_end and len(chunk_boundaries) < max_chunks:
            # Calculate chunk end position
            chunk_end = min(current_pos + chunk_duration, effective_end)

            # Add chunk boundary
            chunk_boundaries.append((current_pos, chunk_end))

            # Move to next chunk start position
            if chunk_end >= effective_end:
                # Reached the end of the audio
                break

            # Update position for next chunk
            next_pos = chunk_end - overlap_duration

            # Safety check - if we're not making progress, break
            if next_pos <= current_pos:
                logger.warning("Not making progress in chunking, breaking loop")
                break

            current_pos = next_pos

        logger.debug(
            f"Calculated {len(chunk_boundaries)} chunk boundaries for audio "
            f"(duration: {effective_duration}s, chunk: {chunk_duration}s, overlap: {overlap_duration}s)"
        )

        return chunk_boundaries

    def create_audio_chunks(
        self,
        audio_path: Path,
        output_dir: Path,
        chunk_boundaries: List[Tuple[float, float]],
        quality_spec: AudioQualitySpecification,
        api_provider: Optional[ApiProvider] = None,
        job_id: Optional[str] = None,
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

        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If chunk creation fails
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        chunk_paths = []
        oversized_chunks = []

        # Get audio format for determining output format
        format_str = (
            quality_spec.format.value
            if hasattr(quality_spec.format, "value")
            else quality_spec.format
        )

        # Use API provider or default to OpenAI Whisper
        effective_provider = api_provider or ApiProvider.OPENAI_WHISPER

        # Get API size limit
        size_limit = ApiSizeLimitPolicy.get_limit_for_provider(effective_provider)

        # Update quality spec with size limit if not already set
        if quality_spec.max_size_bytes <= 0:
            # Apply a 5% safety margin to the size limit
            safe_limit = int(size_limit.max_file_size_bytes * 0.95)
            quality_spec = quality_spec.with_max_size(safe_limit)

        # Process each chunk
        for i, (start_time, end_time) in enumerate(chunk_boundaries):
            # Generate output path for this chunk
            chunk_name = f"chunk_{i:04d}.{format_str}"
            chunk_path = output_dir / chunk_name

            try:
                # Extract audio segment with specified quality
                logger.debug(
                    f"Creating chunk {i+1}/{len(chunk_boundaries)}: {start_time}-{end_time}"
                )

                # Extract the segment
                extracted_path = self.audio_format_service.extract_audio_segment(
                    audio_path=audio_path,
                    output_path=chunk_path,
                    start_time=start_time,
                    end_time=end_time,
                    quality_spec=quality_spec,
                )

                # Check if chunk meets size constraint
                is_valid, error_message = AudioValidationService.validate_audio_file(
                    extracted_path, effective_provider
                )

                if not is_valid:
                    logger.warning(
                        f"Chunk {i+1} exceeds size limit for {effective_provider.value}: {error_message}"
                    )

                    # Try auto-adjusting quality
                    adjusted_spec = AudioValidationService.suggest_quality_reduction(
                        extracted_path, quality_spec, effective_provider
                    )

                    if adjusted_spec:
                        logger.info(f"Applying quality reduction for chunk {i+1}")

                        # Convert with adjusted quality
                        converted_path, metadata = self.audio_converter.convert_with_quality_spec(
                            extracted_path, adjusted_spec
                        )

                        # Check if now valid
                        is_valid, error_message = AudioValidationService.validate_audio_file(
                            converted_path, effective_provider
                        )

                        if is_valid:
                            logger.info(f"Successfully reduced size of chunk {i+1}")
                            chunk_paths.append(converted_path)
                        else:
                            logger.warning(
                                f"Chunk {i+1} still exceeds size limit after quality reduction: {error_message}"
                            )
                            oversized_chunks.append(i)
                    else:
                        # Try direct size constraint conversion
                        try:
                            logger.info(
                                f"Attempting direct size-constrained conversion for chunk {i+1}"
                            )
                            (
                                converted_path,
                                metadata,
                            ) = self.audio_converter.convert_with_size_constraint(
                                extracted_path,
                                size_limit.max_file_size_bytes * 0.95,  # 5% safety margin
                                format_str,
                            )

                            # Check if now valid
                            is_valid, error_message = AudioValidationService.validate_audio_file(
                                converted_path, effective_provider
                            )

                            if is_valid:
                                logger.info(
                                    f"Successfully reduced size of chunk {i+1} with direct conversion"
                                )
                                chunk_paths.append(converted_path)
                            else:
                                logger.warning(
                                    f"Chunk {i+1} still exceeds size limit after direct conversion: {error_message}"
                                )
                                oversized_chunks.append(i)
                        except Exception as e:
                            logger.error(f"Error during size-constrained conversion: {str(e)}")
                            oversized_chunks.append(i)
                else:
                    # Chunk is valid, add to paths
                    chunk_paths.append(extracted_path)

            except Exception as e:
                logger.error(f"Error creating chunk {i+1}: {str(e)}")
                raise ValueError(f"Error creating chunk {i+1}: {str(e)}")

        # Publish chunking event
        EventPublisher.publish(
            ChunkingEvent(
                source_path=audio_path,
                job_id=job_id,
                chunk_count=len(chunk_paths),
                chunk_duration=chunk_boundaries[0][1] - chunk_boundaries[0][0]
                if chunk_boundaries
                else 0,
                overlap_duration=overlap_duration if len(chunk_boundaries) > 1 else 0,
                chunk_paths=chunk_paths,
                has_oversized_chunks=len(oversized_chunks) > 0,
                oversized_chunk_indices=oversized_chunks,
            )
        )

        if len(oversized_chunks) > 0:
            logger.warning(
                f"{len(oversized_chunks)}/{len(chunk_boundaries)} chunks exceed "
                f"size limit for {effective_provider.value}"
            )

        return chunk_paths

    def detect_silence(
        self, audio_path: Path, min_silence_duration: float = 0.5, silence_threshold: float = -40
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
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Delegate to audio format service
        silence_regions = self.audio_format_service.detect_silence(
            audio_path, min_silence_duration, silence_threshold
        )

        # Convert to list of tuples
        return [(region["start"], region["end"]) for region in silence_regions]
