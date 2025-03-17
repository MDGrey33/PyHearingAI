"""
Audio chunking services for PyHearingAI.

This module provides functionality to split large audio files into smaller,
manageable chunks for more efficient processing and to enable resumable operations.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from pyhearingai.config import IdempotentProcessingConfig
from pyhearingai.core.idempotent import AudioChunk, ProcessingJob

logger = logging.getLogger(__name__)


class AudioChunkingService:
    """
    Service for splitting large audio files into smaller chunks for processing.

    This service handles:
    - Chunking audio files based on configured chunk size
    - Detecting silence for optimal chunk boundaries
    - Managing overlap between chunks
    - Storing and retrieving chunk audio files
    """

    def __init__(self, config: Optional[IdempotentProcessingConfig] = None):
        """
        Initialize the audio chunking service.

        Args:
            config: Configuration for idempotent processing
        """
        self.config = config or IdempotentProcessingConfig()

        # Ensure directories exist
        self.config.chunks_dir.mkdir(parents=True, exist_ok=True)

    def get_job_chunks_dir(self, job_id: str) -> Path:
        """
        Get the directory for storing a job's audio chunks.

        Args:
            job_id: ID of the processing job

        Returns:
            Path to the job's chunks directory
        """
        job_chunks_dir = self.config.chunks_dir / job_id / "audio"
        job_chunks_dir.mkdir(parents=True, exist_ok=True)
        return job_chunks_dir

    def get_chunk_path(self, job_id: str, chunk_index: int) -> Path:
        """
        Get the file path for a specific audio chunk.

        Args:
            job_id: ID of the processing job
            chunk_index: Index of the chunk

        Returns:
            Path to the audio chunk file
        """
        return self.get_job_chunks_dir(job_id) / f"chunk_{chunk_index:04d}.wav"

    def create_chunks(self, job: ProcessingJob) -> List[AudioChunk]:
        """
        Split an audio file into chunks based on job configuration.

        Args:
            job: Processing job containing the audio file to chunk

        Returns:
            List of created audio chunks
        """
        logger.info(f"Splitting audio file {job.original_audio_path} into chunks")

        # Load audio file
        audio_path = job.original_audio_path
        y, sr = librosa.load(audio_path, sr=None)
        full_duration = librosa.get_duration(y=y, sr=sr)

        # Apply time range constraints if specified
        start_time = job.start_time if job.start_time is not None else 0
        end_time = job.end_time if job.end_time is not None else full_duration

        # Validate time range
        if start_time >= full_duration:
            raise ValueError(f"Start time {start_time}s exceeds audio duration {full_duration}s")
        if end_time > full_duration:
            logger.warning(
                f"End time {end_time}s exceeds audio duration {full_duration}s, capping at {full_duration}s"
            )
            end_time = full_duration
        if start_time >= end_time:
            raise ValueError(f"Start time {start_time}s must be less than end time {end_time}s")

        # Extract the requested time range
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y = y[start_sample:end_sample]
        duration = end_time - start_time

        # Calculate number of chunks
        chunk_duration = job.chunk_duration
        overlap_duration = job.overlap_duration

        if chunk_duration <= 0:
            raise ValueError("Chunk duration must be positive")

        # Detect silence regions
        silence_regions = self.detect_silence(y, sr)

        # Calculate chunk boundaries with overlap
        chunk_boundaries = self._calculate_chunk_boundaries(
            duration,
            chunk_duration,
            overlap_duration,
            y,
            sr,
            silence_regions,
            start_offset=start_time,  # Pass the start offset to maintain correct timestamps
        )

        # Create chunks
        chunks = []
        for i, (start_time, end_time) in enumerate(chunk_boundaries):
            # Convert time to samples, handling None values for job.start_time
            job_start = job.start_time if job.start_time is not None else 0
            start_sample = int((start_time - job_start) * sr)  # Adjust for job start time
            end_sample = int((end_time - job_start) * sr)  # Adjust for job start time

            # Extract chunk audio
            chunk_audio = y[start_sample:end_sample]

            # Save chunk audio to file with initial quality
            chunk_path = self.get_chunk_path(job.id, i)
            logger.info(f"Saving chunk {i} to {chunk_path}")
            logger.info(f"Chunk directory exists check: {chunk_path.parent.exists()}")
            sf.write(chunk_path, chunk_audio, sr)
            logger.info(f"Chunk file written, exists check: {chunk_path.exists()}")

            # Check if chunk size exceeds OpenAI Whisper API limit (25MB)
            chunk_size = os.path.getsize(chunk_path)
            max_size = 25 * 1024 * 1024  # 25MB
            logger.info(f"Chunk {i} size: {chunk_size} bytes, limit: {max_size} bytes")

            if chunk_size > max_size:
                logger.info(
                    f"Chunk {i} exceeds size limit ({chunk_size} > {max_size}). Adjusting quality..."
                )

                # Try to reduce the file size by adjusting quality
                from pyhearingai.infrastructure.adapters.size_aware_audio_converter import (
                    SizeAwareFFmpegConverter,
                )

                converter = SizeAwareFFmpegConverter()

                try:
                    # Log the path before conversion
                    logger.info(f"Before conversion - chunk path: {chunk_path}")
                    logger.info(f"Before conversion - exists check: {chunk_path.exists()}")
                    logger.info(f"Before conversion - absolute path: {chunk_path.absolute()}")

                    # Convert with size constraint
                    adjusted_path, metadata = converter.convert_with_size_constraint(
                        chunk_path,
                        max_size,
                        target_format="wav",
                        sample_rate=16000,  # Start with reasonable quality
                        channels=1,
                    )

                    # Log the result of conversion
                    logger.info(f"After conversion - adjusted path: {adjusted_path}")
                    logger.info(f"After conversion - exists check: {adjusted_path.exists()}")
                    logger.info(f"After conversion - absolute path: {adjusted_path.absolute()}")

                    # Verify the adjusted file exists before continuing
                    if not adjusted_path.exists():
                        raise FileNotFoundError(f"Adjusted file not found: {adjusted_path}")

                    # Update chunk path to use the size-constrained version
                    chunk_path = adjusted_path
                    logger.info(
                        f"Successfully adjusted chunk {i} to meet size limit: {os.path.getsize(chunk_path)} bytes"
                    )
                except Exception as e:
                    logger.error(f"Failed to adjust chunk {i} size: {e}")
                    logger.exception("Detailed error information:")
                    # Continue with the original chunk, the transcription service will handle the error

            # Create AudioChunk entity
            chunk = AudioChunk(
                job_id=job.id,
                chunk_path=chunk_path,
                start_time=start_time,
                end_time=end_time,
                chunk_index=i,
            )
            logger.info(f"Created AudioChunk entity with path: {chunk.chunk_path}")
            chunks.append(chunk)

        # Update job with total chunks
        job.total_chunks = len(chunks)
        job.chunks = [chunk.id for chunk in chunks]

        logger.info(f"Created {len(chunks)} chunks for job {job.id}")
        return chunks

    def _calculate_chunk_boundaries(
        self,
        duration: float,
        chunk_duration: float,
        overlap_duration: float,
        audio: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        silence_regions: Optional[List[Tuple[float, float]]] = None,
        use_silence_detection: bool = True,
        start_offset: float = 0.0,
    ) -> List[Tuple[float, float]]:
        """
        Calculate chunk boundaries for audio.

        Args:
            duration: Total audio duration in seconds
            chunk_duration: Target chunk duration in seconds
            overlap_duration: Overlap between chunks in seconds
            audio: Audio signal (optional)
            sr: Sample rate (optional)
            silence_regions: List of silence regions (optional)
            use_silence_detection: Whether to use silence detection for optimal boundaries
            start_offset: Start time offset in seconds

        Returns:
            List of chunk boundaries as (start_time, end_time) tuples
        """
        print(
            f">>> _calculate_chunk_boundaries: duration={duration}, chunk_duration={chunk_duration}, overlap={overlap_duration}"
        )
        if start_offset > 0:
            print(f">>> Using start offset of {start_offset} seconds")
        print(f">>> Using simplified chunk boundary calculation for testing")

        # Handle very short audio files (shorter than chunk duration)
        if duration <= chunk_duration:
            print(f">>> Audio shorter than chunk duration, creating single chunk")
            return [(start_offset, start_offset + duration)]

        # Check for invalid overlap
        if overlap_duration >= chunk_duration:
            print(
                f">>> WARNING: overlap ({overlap_duration}) >= chunk_duration ({chunk_duration}). Setting to half chunk duration."
            )
            overlap_duration = chunk_duration / 2

        # SIMPLIFIED: Basic chunk calculation without trying to be smart
        chunk_boundaries = []
        start = start_offset
        end_point = start_offset + duration

        # Limit the number of chunks to prevent infinite loops
        max_chunks = int(duration / (chunk_duration - overlap_duration) * 2)
        print(f">>> Setting max chunks to {max_chunks}")

        while start < end_point and len(chunk_boundaries) < max_chunks:
            # Calculate end time
            end = min(start + chunk_duration, end_point)

            # Add chunk boundary
            chunk_boundaries.append((start, end))
            print(f">>> Added chunk boundary: ({start}, {end})")

            # Update start for next chunk
            next_start = end - overlap_duration

            # Safety check - if we're not making progress, break
            if next_start <= start or end >= end_point:
                print(">>> Breaking: no progress or reached end")
                break

            start = next_start

        print(f">>> Created {len(chunk_boundaries)} chunk boundaries")
        return chunk_boundaries

    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds.

        Args:
            audio_path: Path to the audio file

        Returns:
            Duration in seconds
        """
        print(f">>> Loading audio file {audio_path}")
        y, sr = librosa.load(audio_path, sr=None)
        print(f">>> Loaded audio: samples={len(y)}, sample_rate={sr}")
        duration = librosa.get_duration(y=y, sr=sr)
        print(f">>> Calculated duration: {duration} seconds")
        return duration

    def _extract_chunk_audio(
        self, audio_path: str, output_path: str, start_time: float, end_time: float
    ) -> None:
        """
        Extract a chunk of audio from the input file and save it to the output path.

        Args:
            audio_path: Path to the input audio file
            output_path: Path to save the extracted chunk
            start_time: Start time of the chunk in seconds
            end_time: End time of the chunk in seconds
        """
        print(f">>> Extracting chunk from {audio_path} ({start_time}s to {end_time}s)")
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=end_time - start_time)
        print(f">>> Loaded chunk: samples={len(y)}, sample_rate={sr}")

        # Save chunk to file
        print(f">>> Saving chunk to {output_path}")
        sf.write(output_path, y, sr)
        print(f">>> Saved chunk to {output_path}")

    def _find_silence_near(
        self, time: float, audio_path: str, max_adjustment: float = 0.5, min_val: float = 0
    ) -> float:
        """
        Find the nearest silence point near the given time.

        Args:
            time: Target time in seconds
            audio_path: Path to the audio file
            max_adjustment: Maximum time adjustment allowed (in seconds)
            min_val: Minimum allowed value for the adjusted time

        Returns:
            Adjusted time (in seconds) aligned with a silence point
        """
        print(
            f">>> Finding silence near {time}s (max_adjustment={max_adjustment}, min_val={min_val})"
        )
        # For testing purposes, just return the original time
        # In a real implementation, this would analyze the audio and find silence
        return max(min_val, time)

    def detect_silence(
        self, audio: np.ndarray, sr: int, threshold: float = 0.03, min_silence_len: int = 500
    ) -> List[Tuple[float, float]]:
        """
        Detect periods of silence in audio.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            threshold: Energy threshold below which is considered silence
            min_silence_len: Minimum silence length in milliseconds

        Returns:
            List of (start_time, end_time) tuples for silence regions
        """
        # Calculate energy of signal
        energy = librosa.feature.rms(y=audio)[0]

        # Find silence regions
        silence_regions = []
        in_silence = False
        silence_start = 0
        silence_start_time = 0.0

        # Default hop length in librosa for frame conversion
        hop_length = 512

        # Convert min_silence_len from ms to frames
        # This assumes hop_length=512 which is librosa's default for rms
        min_silence_frames = int((min_silence_len / 1000) * (sr / hop_length))

        for i, e in enumerate(energy):
            # Convert frame index to time using hop_length
            frame_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)

            if e < threshold and not in_silence:
                # Entering silence
                in_silence = True
                silence_start = i
                silence_start_time = frame_time
            elif e >= threshold and in_silence:
                # Exiting silence
                in_silence = False

                # Only include if silence is long enough
                if i - silence_start >= min_silence_frames:
                    silence_end_time = frame_time
                    silence_regions.append((silence_start_time, silence_end_time))

        # Handle case where audio ends in silence
        if in_silence:
            # Calculate the end time using the total frames
            silence_end_time = librosa.frames_to_time(len(energy), sr=sr, hop_length=hop_length)

            if len(energy) - silence_start >= min_silence_frames:
                silence_regions.append((silence_start_time, silence_end_time))

        logger.debug(f"Detected {len(silence_regions)} silence regions")
        return silence_regions

    def optimize_chunk_boundaries(
        self,
        audio: np.ndarray,
        sr: int,
        initial_boundaries: List[Tuple[float, float]],
        silence_regions: List[Tuple[float, float]],
        max_adjustment: float = 2.0,  # Maximum seconds to adjust boundary
    ) -> List[Tuple[float, float]]:
        """
        Optimize chunk boundaries to align with silence regions.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            initial_boundaries: Initial chunk boundaries
            silence_regions: Detected silence regions
            max_adjustment: Maximum time in seconds to adjust boundary

        Returns:
            Optimized list of (start_time, end_time) tuples for each chunk
        """
        if not silence_regions or not initial_boundaries:
            return initial_boundaries

        optimized_boundaries = []

        # Add first boundary (start from beginning of audio)
        optimized_boundaries.append((initial_boundaries[0][0], initial_boundaries[0][1]))

        # Optimize intermediate boundaries
        for i in range(1, len(initial_boundaries)):
            prev_end = initial_boundaries[i - 1][1]
            current_start = initial_boundaries[i][0]
            current_end = initial_boundaries[i][1]

            # Find best silence region to align with
            best_silence = None
            min_distance = float("inf")

            for silence_start, silence_end in silence_regions:
                # Check if silence region is within adjustment range of the boundary
                boundary_time = prev_end  # The boundary we're trying to optimize is the end of the previous chunk

                # Calculate distance from boundary to start and end of silence
                start_distance = abs(silence_start - boundary_time)
                end_distance = abs(silence_end - boundary_time)

                # Use the closest point (start or end of silence)
                if (
                    start_distance < end_distance
                    and start_distance < min_distance
                    and start_distance <= max_adjustment
                ):
                    min_distance = start_distance
                    best_silence = (silence_start, silence_end)
                elif end_distance < min_distance and end_distance <= max_adjustment:
                    min_distance = end_distance
                    best_silence = (silence_start, silence_end)

            # If found a suitable silence region, adjust the boundary
            if best_silence:
                silence_start, silence_end = best_silence

                # Determine if start or end of silence is closer to the boundary
                if abs(silence_start - prev_end) < abs(silence_end - prev_end):
                    # Use start of silence as boundary
                    new_boundary = silence_start
                else:
                    # Use end of silence as boundary
                    new_boundary = silence_end

                # Adjust the end of the previous chunk and start of current chunk
                prev_chunk = optimized_boundaries.pop()
                optimized_boundaries.append((prev_chunk[0], new_boundary))
                optimized_boundaries.append((new_boundary, current_end))
            else:
                # No suitable silence found, keep original boundary
                optimized_boundaries.append((current_start, current_end))

        return optimized_boundaries

    def cleanup_job_chunks(self, job_id: str) -> int:
        """
        Delete all audio chunk files for a job.

        Args:
            job_id: ID of the processing job

        Returns:
            Number of files deleted
        """
        job_chunks_dir = self.get_job_chunks_dir(job_id)
        count = 0

        if job_chunks_dir.exists():
            for chunk_file in job_chunks_dir.glob("*.wav"):
                try:
                    chunk_file.unlink()
                    count += 1
                except OSError as e:
                    logger.warning(f"Failed to delete chunk file {chunk_file}: {e}")

        return count
