"""
Reconciliation service for combining diarization and transcription results.

This module provides a service that reconciles diarization and transcription
results into a coherent final output, using GPT-4 for advanced reconciliation.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pyhearingai.core.idempotent import AudioChunk, ProcessingJob
from pyhearingai.core.models import DiarizationSegment, Segment
from pyhearingai.diarization.repositories.diarization_repository import DiarizationRepository
from pyhearingai.infrastructure.repositories.json_repositories import (
    JsonChunkRepository,
    JsonSegmentRepository,
)
from pyhearingai.reconciliation.adapters.base import BaseReconciliationAdapter
from pyhearingai.reconciliation.adapters.gpt import GPT4ReconciliationAdapter
from pyhearingai.reconciliation.adapters.responses import ResponsesReconciliationAdapter
from pyhearingai.reconciliation.repositories.reconciliation_repository import (
    ReconciliationRepository,
)
from pyhearingai.transcription.repositories.transcription_repository import TranscriptionRepository

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batched reconciliation"""

    def __init__(
        self,
        batch_size_seconds: int = 180,
        batch_overlap_seconds: int = 10,
        max_tokens_per_batch: int = 2500,  # Reduced from 3000 to be more conservative
        min_batch_size_seconds: int = 30,
    ):
        """
        Initialize batch configuration

        Args:
            batch_size_seconds: Size of each batch in seconds
            batch_overlap_seconds: Overlap between batches in seconds
            max_tokens_per_batch: Maximum tokens to send in a batch
            min_batch_size_seconds: Minimum batch size in seconds
        """
        self.batch_size_seconds = batch_size_seconds
        self.batch_overlap_seconds = batch_overlap_seconds
        self.max_tokens_per_batch = max_tokens_per_batch
        self.min_batch_size_seconds = min_batch_size_seconds


class ReconciliationService:
    """
    Service for reconciling diarization and transcription results.

    This service uses GPT-4 to combine diarization and transcription data
    into a coherent final output, handling speaker continuity across chunk
    boundaries and resolving discrepancies.
    """

    def __init__(
        self,
        reconciliation_repository=None,
        diarization_repository=None,
        transcription_repository=None,
        use_responses_api: bool = False,
    ):
        """
        Initialize the reconciliation service.

        Args:
            reconciliation_repository: Repository for storing reconciliation results
            diarization_repository: Repository for retrieving diarization results
            transcription_repository: Repository for retrieving transcription results
            use_responses_api: Whether to use the Responses API adapter (default: False)
        """
        # Create or use the repositories
        self.repository = reconciliation_repository or ReconciliationRepository()
        self.diarization_repository = diarization_repository or DiarizationRepository()
        self.transcription_repository = transcription_repository or TranscriptionRepository()

        # Select the appropriate adapter based on the feature flag
        if use_responses_api:
            logger.info("Using ResponsesReconciliationAdapter for token-efficient processing")
            self.adapter = ResponsesReconciliationAdapter()
        else:
            logger.info("Using GPT4ReconciliationAdapter as the default reconciliation adapter")
            self.adapter = GPT4ReconciliationAdapter()

        # Initialize additional repositories for chunk and segment data
        self.chunk_repository = JsonChunkRepository()
        self.segment_repository = JsonSegmentRepository()

        # Store configuration
        self.model = self.adapter.model
        self.use_responses_api = use_responses_api

        # Caching for expensive operations
        self._cached_reconciliation = {}

        logger.debug(f"Initialized ReconciliationService with model {self.model}")

    def close(self):
        """
        Close the service and release any resources.

        This method should be called when the service is no longer needed
        to ensure proper cleanup of resources.
        """
        # Release any resources used by the adapter
        try:
            if hasattr(self.adapter, "close"):
                self.adapter.close()
        except Exception as e:
            logger.error(f"Error closing reconciliation adapter: {str(e)}")

        # Clear any cached data
        if hasattr(self, "_cached_reconciliation"):
            self._cached_reconciliation = None

        logger.debug("Closed ReconciliationService")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and ensure resources are released."""
        self.close()

    def reconcile(
        self,
        job: ProcessingJob,
        options: Optional[Dict[str, Any]] = None,
        force: bool = False,
        progressive: bool = False,
    ) -> List[Segment]:
        """
        Reconcile diarization and transcription results into a coherent final transcript.

        Args:
            job: The processing job to reconcile
            options: Optional settings for the reconciliation process
            force: Whether to force reconciliation even if results exist
            progressive: Whether to use progressive reconciliation

        Returns:
            List of reconciled segments
        """
        # Check if we already have reconciled results
        if not force and self.repository.has_reconciled_result(job.id):
            logger.info(f"Using cached reconciliation results for job {job.id}")
            return self.repository.get_reconciled_result(job.id)

        # Get job chunks
        chunks = self.chunk_repository.get_by_job_id(job.id)
        if not chunks:
            logger.warning(f"No chunks found for job {job.id}")
            return []

        # Collect diarization and transcription segments
        diarization_segments = {}
        transcription_segments = {}
        segment_transcriptions = {}

        # Get all segments for all chunks
        for chunk in chunks:
            # Get diarization segments
            diarization_result = self.diarization_repository.get(job.id, chunk.id)
            if diarization_result:
                diarization_segments[chunk.id] = diarization_result

            # Get transcription segments and their text content
            transcription_result = self.transcription_repository.get_chunk_transcription(
                job.id, chunk.id
            )
            if transcription_result:
                transcription_segments[chunk.id] = transcription_result

                # Get text for each segment
                for segment in transcription_result:
                    segment_key = f"{chunk.id}:{segment.start:.2f}-{segment.end:.2f}"
                    segment_id = f"{chunk.id}_segment_{segment.start:.2f}-{segment.end:.2f}"
                    text = self.transcription_repository.get_segment_transcription(
                        job.id, segment_id
                    )
                    if text:
                        segment_transcriptions[segment_key] = text

        logger.debug(
            f"Reconciling job {job.id} with {len(diarization_segments)} chunks, "
            f"{sum(len(segs) for segs in diarization_segments.values())} diarization segments, "
            f"{sum(len(segs) for segs in transcription_segments.values())} transcription segments"
        )

        # Do the reconciliation
        try:
            # Call the adapter to reconcile
            reconciled_segments = self.adapter.reconcile(
                job=job,
                diarization_segments=diarization_segments,
                transcription_segments=transcription_segments,
                segment_transcriptions=segment_transcriptions,
                options=options,
            )

            # Save the result
            if reconciled_segments:
                # Build some metadata to store with the results
                metadata = {
                    "model": self.model,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_segments": len(reconciled_segments),
                    "version": "1.0",
                }

                # Save to the repository
                self.repository.save_reconciled_result(job.id, reconciled_segments, metadata)

            return reconciled_segments

        except Exception as e:
            logger.error(f"Error during reconciliation: {str(e)}")
            raise

    def reconcile_progressive(
        self,
        job: ProcessingJob,
        options: Optional[Dict[str, Any]] = None,
        force: bool = False,
        api_key: Optional[str] = None,
        min_chunks: int = 1,
        save_interim_results: bool = True,
    ) -> List[Segment]:
        """
        Reconcile chunks progressively as they become available.

        Args:
            job: The processing job to reconcile
            options: Optional settings for the reconciliation process
            force: Whether to force reconciliation even if already exists
            api_key: Optional OpenAI API key (only used with GPT4ReconciliationAdapter)
            min_chunks: Minimum number of chunks to process at once
            save_interim_results: Whether to save interim results

        Returns:
            List of reconciled segments with speaker information
        """
        # Get all chunks for this job
        all_chunks = self.chunk_repository.get_by_job_id(job.id)
        if not all_chunks:
            logger.warning(f"No chunks found for job {job.id}")
            return []

        # Filter chunks that have completed processing
        completed_chunks = [chunk for chunk in all_chunks if chunk.is_complete()]

        if len(completed_chunks) < min_chunks:
            logger.info(
                f"Only {len(completed_chunks)} chunks completed out of {len(all_chunks)}. "
                f"Waiting for at least {min_chunks} before reconciliation."
            )
            return []

        # Sort chunks by index to ensure proper order
        completed_chunks.sort(key=lambda c: c.index)

        # Collect all diarization segments and transcription segments
        diarization_segments = {}
        transcription_segments = {}
        segment_transcriptions = {}

        logger.debug(
            f"Collecting data for progressive reconciliation of job {job.id} "
            f"with {len(completed_chunks)}/{len(all_chunks)} chunks"
        )

        # Process each completed chunk
        for chunk in completed_chunks:
            # Get diarization segments for this chunk
            di_segments = self.diarization_repository.get(job.id, chunk.id)
            if di_segments:
                diarization_segments[chunk.id] = di_segments

                # Get transcription segments for this chunk
                tr_segments = self.transcription_repository.get_chunk_transcription(
                    job.id, chunk.id
                )
                if tr_segments:
                    transcription_segments[chunk.id] = tr_segments

                # Get segment transcriptions
                for i, segment in enumerate(di_segments):
                    segment_id = f"{chunk.id}_segment_{i}_{segment.speaker_id}"
                    text = self.transcription_repository.get_segment_transcription(
                        job.id, segment_id
                    )
                    if text:
                        segment_transcriptions[segment_id] = text

        if not diarization_segments:
            logger.warning(f"No diarization segments found for completed chunks in job {job.id}")
            return []

        if not transcription_segments:
            logger.warning(f"No transcription segments found for completed chunks in job {job.id}")
            return []

        # Call the appropriate adapter
        try:
            # Handle API key based on adapter type
            if api_key is not None and isinstance(self.adapter, GPT4ReconciliationAdapter):
                # Store original API key
                original_api_key = self.adapter._api_key
                # Set the API key on the adapter instance
                self.adapter._api_key = api_key

                try:
                    # Call reconcile without passing api_key
                    reconciled_segments = self.adapter.reconcile(
                        job=job,
                        diarization_segments=diarization_segments,
                        transcription_segments=transcription_segments,
                        segment_transcriptions=segment_transcriptions,
                        options=options,
                    )
                finally:
                    # Restore original API key
                    self.adapter._api_key = original_api_key
            else:
                # Standard call without API key handling
                reconciled_segments = self.adapter.reconcile(
                    job=job,
                    diarization_segments=diarization_segments,
                    transcription_segments=transcription_segments,
                    segment_transcriptions=segment_transcriptions,
                    options=options,
                )

            # Save the reconciled results with progress metadata
            metadata = {
                "progressive": True,
                "completed_chunks": len(completed_chunks),
                "total_chunks": len(all_chunks),
                "is_complete": len(completed_chunks) == len(all_chunks),
                "chunk_ids": [chunk.id for chunk in completed_chunks],
            }

            if save_interim_results or len(completed_chunks) == len(all_chunks):
                suffix = (
                    ""
                    if len(completed_chunks) == len(all_chunks)
                    else f"_progress_{len(completed_chunks)}"
                )
                self.repository.save_reconciled_result(
                    job.id + suffix, reconciled_segments, metadata
                )

            return reconciled_segments

        except Exception as e:
            logger.error(f"Error during progressive reconciliation for job {job.id}: {str(e)}")
            raise

    def format_output(
        self, job: ProcessingJob, segments: List[Segment], output_format: str = "txt"
    ) -> str:
        """
        Format the reconciled segments into the requested output format.

        Args:
            job: The processing job
            segments: The reconciled segments
            output_format: Format of the output (txt, json, srt, vtt, md)

        Returns:
            Formatted content as a string
        """
        # Check if we already have formatted output
        if self.repository.formatted_output_exists(job.id, output_format):
            content = self.repository.get_formatted_output(job.id, output_format)
            if content:
                return content

        # Import formatters
        from pyhearingai.application.outputs import to_json, to_markdown, to_srt, to_text, to_vtt
        from pyhearingai.core.models import TranscriptionResult

        # Create a TranscriptionResult object
        result = TranscriptionResult(
            segments=segments, audio_path=job.original_audio_path, metadata={"job_id": job.id}
        )

        # Format the output based on the requested format
        if output_format == "txt":
            content = to_text(result)
        elif output_format == "json":
            content = to_json(result)
        elif output_format == "srt":
            content = to_srt(result)
        elif output_format == "vtt":
            content = to_vtt(result)
        elif output_format == "md":
            content = to_markdown(result)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Save the formatted output
        self.repository.save_formatted_output(job.id, output_format, content)

        return content

    def save_output_file(
        self, job: ProcessingJob, output_path: Path, output_format: Optional[str] = None
    ) -> Path:
        """
        Save the reconciled results to a file.

        Args:
            job: The processing job
            output_path: Path to save the output file
            output_format: Optional format override (txt, json, srt, vtt, md)

        Returns:
            Path to the saved file
        """
        # Determine the output format from the file extension if not provided
        if not output_format:
            suffix = output_path.suffix.lower()
            if suffix in [".txt"]:
                output_format = "txt"
            elif suffix in [".json"]:
                output_format = "json"
            elif suffix in [".srt"]:
                output_format = "srt"
            elif suffix in [".vtt"]:
                output_format = "vtt"
            elif suffix in [".md", ".markdown"]:
                output_format = "md"
            else:
                # Default to txt if format can't be determined
                output_format = "txt"

        # Get the reconciled segments
        segments = self.repository.get_reconciled_result(job.id)
        if not segments:
            # Try to reconcile now if we don't have results
            segments = self.reconcile(job)

        # Format the output
        content = self.format_output(job, segments, output_format)

        # Save to the provided path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)

        logger.info(f"Saved output to {output_path}")
        return output_path

    def reconcile_batched(
        self,
        job: ProcessingJob,
        config: Optional[BatchConfig] = None,
    ) -> List[Segment]:
        """
        Reconcile transcription and diarization data in batches.

        Args:
            job: The job to process
            config: Batch configuration

        Returns:
            List of reconciled segments
        """
        if config is None:
            config = BatchConfig()

        # Get all the audio chunks
        audio_chunks = self.chunk_repository.get_by_job_id(job.id)
        if not audio_chunks:
            logger.warning(f"No audio chunks found for job {job.id}")
            return []

        # Sort chunks by start time
        audio_chunks.sort(key=lambda c: c.start_time)

        # Use diarization segments if available, otherwise use transcription segments
        diarization_data = self.diarization_repository.get_all(job.id)

        # Get job metadata
        job_metadata = self.repository.get_job_metadata(job.id)
        if job_metadata is None:
            logger.warning(f"No job metadata found for job {job.id}")
            job_metadata = {}

        # Create batches from audio chunks
        total_duration = audio_chunks[-1].end_time - audio_chunks[0].start_time
        logger.info(f"Total audio duration: {total_duration:.2f} seconds")

        # Calculate optimal batch size based on total duration and token limits
        if total_duration > 600:  # For very long recordings (>10 min)
            # Use smaller batches for very long recordings to avoid token issues
            config.batch_size_seconds = min(config.batch_size_seconds, 120)
            logger.info(
                f"Using reduced batch size of {config.batch_size_seconds} seconds for long recording"
            )

        # Create batches with proper overlap
        batches = []
        batch_start = audio_chunks[0].start_time

        # Ensure we don't create empty batches at the end
        max_end_time = audio_chunks[-1].end_time - config.min_batch_size_seconds

        while batch_start < max_end_time:
            batch_end = min(batch_start + config.batch_size_seconds, audio_chunks[-1].end_time)

            # Get chunks in this batch
            batch_chunks = []
            for chunk in audio_chunks:
                # Include chunks that overlap with the batch time range
                if chunk.end_time > batch_start and chunk.start_time < batch_end:
                    batch_chunks.append(chunk)

            batches.append((batch_start, batch_end, batch_chunks))

            # Move to next batch with overlap
            batch_start = batch_end - config.batch_overlap_seconds

        logger.info(f"Created {len(batches)} batches for reconciliation")

        # Process each batch
        all_segments = []

        for i, (start_time, end_time, batch_chunks) in enumerate(batches):
            logger.info(
                f"Processing batch {i+1}/{len(batches)}: {start_time:.2f}s - {end_time:.2f}s ({len(batch_chunks)} chunks)"
            )

            # Skip empty batches
            if not batch_chunks:
                logger.warning(f"Skipping empty batch {i+1}")
                continue

            try:
                # Collect data for this batch
                (
                    diarization_segments,
                    transcription_segments,
                    segment_transcriptions,
                ) = self._collect_data_for_batch(batch_chunks, start_time, end_time)

                # Use batch-specific options
                batch_options = {
                    "batch_index": i,
                    "batch_count": len(batches),
                    "batch_start": start_time,
                    "batch_end": end_time,
                    "is_first_batch": i == 0,
                    "is_last_batch": i == len(batches) - 1,
                    # Estimate token count to avoid hitting limits
                    "max_tokens": min(
                        config.max_tokens_per_batch,
                        7000 - min(2000, sum(len(t) for t in segment_transcriptions.values()) // 4),
                    ),
                }

                # Log batch stats
                total_segments = sum(len(segs) for segs in diarization_segments.values())
                total_text_length = sum(len(text) for text in segment_transcriptions.values())
                logger.info(
                    f"Batch {i+1} stats: {total_segments} segments, {total_text_length} chars, "
                    + f"estimated tokens: {total_text_length // 4}, max_tokens: {batch_options['max_tokens']}"
                )

                if total_text_length // 4 > batch_options["max_tokens"] * 2:
                    logger.warning(
                        f"Batch {i+1} may exceed token limits. Reducing segment count..."
                    )
                    # Dynamically reduce the batch size if it's likely to exceed token limits
                    adjusted_end_time = start_time + (end_time - start_time) * 0.6
                    logger.info(
                        f"Adjusting batch end time from {end_time:.2f}s to {adjusted_end_time:.2f}s"
                    )

                    # Recollect data with adjusted time
                    (
                        diarization_segments,
                        transcription_segments,
                        segment_transcriptions,
                    ) = self._collect_data_for_batch(batch_chunks, start_time, adjusted_end_time)

                    # Update batch end time
                    batch_options["batch_end"] = adjusted_end_time
                    end_time = adjusted_end_time

                    # Log adjusted batch stats
                    total_segments = sum(len(segs) for segs in diarization_segments.values())
                    total_text_length = sum(len(text) for text in segment_transcriptions.values())
                    logger.info(
                        f"Adjusted batch {i+1} stats: {total_segments} segments, {total_text_length} chars"
                    )

                # Process this batch
                batch_result = self.adapter.reconcile(
                    job=job,
                    diarization_segments=diarization_segments,
                    transcription_segments=transcription_segments,
                    segment_transcriptions=segment_transcriptions,
                    options=batch_options,
                )

                # Save the batch result with a unique interim job ID
                interim_job_id = f"{job.id}_batch_{i}"
                self.repository.save_reconciled_result(interim_job_id, batch_result)

                # Add segments to the combined result
                for segment in batch_result:
                    # Only include segments that are within this batch range (accounting for overlap)
                    if (
                        (segment.start >= start_time and segment.end <= end_time)
                        or (i == 0 and segment.start < start_time)
                        or (i == len(batches) - 1 and segment.end > end_time)
                    ):
                        all_segments.append(segment)
                    # For overlapping segments, only include them in the earlier batch
                    elif (
                        segment.start < end_time
                        and segment.end > start_time
                        and i < len(batches) - 1
                    ):
                        # Check if this segment is more in this batch than the next
                        segment_middle = (segment.start + segment.end) / 2
                        if segment_middle < end_time:
                            all_segments.append(segment)
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {str(e)}")

                # Try to extract segments from the error message if it contains JSON
                extracted_segments = self._extract_segments_from_error(str(e))
                if extracted_segments:
                    logger.info(f"Extracted {len(extracted_segments)} segments from error message")

                    # Add segments to the combined result
                    all_segments.extend(extracted_segments)

                    # Save the extracted segments
                    interim_job_id = f"{job.id}_batch_{i}_recovered"
                    self.repository.save_reconciled_result(interim_job_id, extracted_segments)

                # If there's still an overlap with the next batch, reduce the current batch size
                # and proceed with the next batch to ensure we don't miss any content
                if i < len(batches) - 1:
                    next_start, next_end, next_chunks = batches[i + 1]

                    # Adjust the next batch to start earlier if needed
                    if next_start > start_time + config.min_batch_size_seconds:
                        # Start the next batch right after where this one started + min size
                        new_next_start = start_time + config.min_batch_size_seconds
                        batches[i + 1] = (new_next_start, next_end, next_chunks)
                        logger.info(
                            f"Adjusted next batch to start at {new_next_start:.2f}s after batch failure"
                        )

        # Sort all segments by start time
        all_segments.sort(key=lambda s: s.start)

        # Merge overlapping segments with same speaker
        merged_segments = self._merge_overlapping_segments(all_segments)

        # Save the final reconciled result
        self.repository.save_reconciled_result(job.id, merged_segments)

        return merged_segments

    def _collect_data_for_batch(
        self, chunks: List[AudioChunk], start_time: float, end_time: float
    ) -> Tuple[Dict[str, List[DiarizationSegment]], Dict[str, List[str]], Dict[str, str]]:
        """
        Collect diarization and transcription data for a batch of chunks.

        Args:
            chunks: Chunks in the batch
            start_time: Batch start time
            end_time: Batch end time

        Returns:
            Tuple of (diarization_segments, transcription_segments, segment_transcriptions)
        """
        diarization_segments = {}
        transcription_segments = {}
        segment_transcriptions = {}

        for chunk in chunks:
            # Get diarization segments
            chunk_diarization = self.diarization_repository.get(chunk.job_id, chunk.id)
            if chunk_diarization:
                # Filter segments in the batch time range
                filtered_segments = [
                    segment
                    for segment in chunk_diarization
                    if (segment.end > start_time and segment.start < end_time)
                ]
                diarization_segments[chunk.id] = filtered_segments

            # Get transcription segments
            chunk_transcription = self.transcription_repository.get_chunk_transcription(
                chunk.job_id, chunk.id
            )
            if chunk_transcription:
                # Filter segments in the batch time range
                filtered_segments = [
                    segment
                    for segment in chunk_transcription
                    if (segment.end > start_time and segment.start < end_time)
                ]
                transcription_segments[chunk.id] = filtered_segments

                # Get segment transcriptions
                for i, segment in enumerate(filtered_segments):
                    # Generate a segment ID if not available
                    segment_id = getattr(segment, "id", f"segment_{i}")
                    segment_text = self.transcription_repository.get_segment_transcription(
                        chunk.job_id, segment_id
                    )
                    if segment_text:
                        segment_transcriptions[f"{chunk.id}_{segment_id}"] = segment_text

        return diarization_segments, transcription_segments, segment_transcriptions

    def _merge_overlapping_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Merge overlapping segments with the same speaker.

        Args:
            segments: List of segments to merge

        Returns:
            List of merged segments
        """
        if not segments:
            return []

        # Sort segments by start time
        segments.sort(key=lambda s: s.start)

        merged_segments = []
        current_segment = segments[0]

        for segment in segments[1:]:
            if segment.start <= current_segment.end:
                # Merge segments if they overlap
                current_segment.end = max(current_segment.end, segment.end)
            else:
                # Append current segment to merged list and start new current segment
                merged_segments.append(current_segment)
                current_segment = segment

        # Append the last segment
        merged_segments.append(current_segment)

        return merged_segments

    def _extract_segments_from_error(self, error_message: str) -> List[Segment]:
        """
        Extract segments from an error message that might contain JSON.

        When GPT-4 returns a valid response but the parser fails, we try to
        extract the segments from the error message.

        Args:
            error_message: The error message that might contain segments

        Returns:
            List of segments extracted from the error message, or empty list if extraction fails
        """
        try:
            import json
            import re
            from datetime import datetime

            logger.info("Attempting to extract segments from error message")

            # Look for any JSON array or object in the error message
            json_patterns = [
                r'\[\s*{\s*".*?}\s*\]',  # JSON array
                r'{\s*"segments"\s*:\s*\[\s*{.*?}\s*\]\s*}',  # JSON object with segments key
                r'{\s*".*?}\s*',  # JSON object
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, error_message, re.DOTALL)
                if matches:
                    # Try to parse the first match as JSON
                    for match in matches:
                        try:
                            data = json.loads(match)

                            # Handle both array and object formats
                            if isinstance(data, list):
                                segment_list = data
                            elif isinstance(data, dict) and "segments" in data:
                                segment_list = data["segments"]
                            elif isinstance(data, dict):
                                # Single segment
                                segment_list = [data]
                            else:
                                continue

                            # Create segments from the extracted data
                            segments = []
                            for segment_data in segment_list:
                                try:
                                    # Handle various time formats
                                    start = segment_data.get("start", 0)
                                    end = segment_data.get("end", 0)

                                    # Convert string times (HH:MM:SS) to seconds
                                    if isinstance(start, str):
                                        if ":" in start:
                                            # Time format like "00:01:23.456"
                                            time_parts = start.replace(",", ".").split(":")
                                            if len(time_parts) == 3:
                                                start = (
                                                    float(time_parts[0]) * 3600
                                                    + float(time_parts[1]) * 60
                                                    + float(time_parts[2])
                                                )
                                            elif len(time_parts) == 2:
                                                start = float(time_parts[0]) * 60 + float(
                                                    time_parts[1]
                                                )
                                        else:
                                            try:
                                                start = float(start)
                                            except ValueError:
                                                start = 0

                                    if isinstance(end, str):
                                        if ":" in end:
                                            # Time format like "00:01:23.456"
                                            time_parts = end.replace(",", ".").split(":")
                                            if len(time_parts) == 3:
                                                end = (
                                                    float(time_parts[0]) * 3600
                                                    + float(time_parts[1]) * 60
                                                    + float(time_parts[2])
                                                )
                                            elif len(time_parts) == 2:
                                                end = float(time_parts[0]) * 60 + float(
                                                    time_parts[1]
                                                )
                                        else:
                                            try:
                                                end = float(end)
                                            except ValueError:
                                                end = 0

                                    # Create segment with available fields
                                    segment = Segment(
                                        start=float(start),
                                        end=float(end),
                                        text=segment_data.get("text", ""),
                                        speaker_id=segment_data.get("speaker_id", None),
                                    )
                                    segments.append(segment)
                                except Exception as segment_error:
                                    logger.warning(
                                        f"Error creating segment from data {segment_data}: {str(segment_error)}"
                                    )
                                    continue

                            if segments:
                                logger.info(
                                    f"Successfully extracted {len(segments)} segments from error message"
                                )
                                return segments
                        except json.JSONDecodeError:
                            logger.debug(f"Failed to parse potential JSON match: {match[:100]}...")
                            continue

            logger.warning("No valid JSON segments found in error message")
            return []

        except Exception as e:
            logger.error(f"Error while extracting segments from error message: {str(e)}")
            return []
