"""
Transcription service for processing audio chunks and speaker segments.

This module provides a service that performs speech transcription on audio chunks
and speaker segments, with support for idempotent processing and resumability.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time
import concurrent.futures

from pyhearingai.core.idempotent import AudioChunk, SpeakerSegment, ProcessingJob, ChunkStatus
from pyhearingai.core.models import Segment, DiarizationSegment
from pyhearingai.transcription.adapters.whisper import WhisperAdapter
from pyhearingai.transcription.repositories.transcription_repository import TranscriptionRepository

logger = logging.getLogger(__name__)


def _process_chunk_directly(chunk_or_id, transcriber_name, **kwargs):
    """
    Process a chunk directly without a service instance.

    This function is used for parallel processing to avoid issues with
    pickling the service instance.

    Args:
        chunk_or_id: Audio chunk or chunk ID to process
        transcriber_name: Name of the transcriber to use
        **kwargs: Additional arguments for the transcriber

    Returns:
        List of transcription segments
    """
    try:
        # Create a new service instance
        from pyhearingai.transcription.repositories.transcription_repository import (
            TranscriptionRepository,
        )

        service = TranscriptionService(transcriber_name=transcriber_name)

        # Process the chunk
        segments = service.transcribe_chunk(chunk_or_id, None, **kwargs)

        # Close the service
        service.close()

        return segments
    except Exception as e:
        from traceback import format_exc

        logger.error(f"Error processing chunk directly: {str(e)}")
        logger.error(format_exc())
        return []


def _process_segment_directly(segment_or_id, audio_path, transcriber_name, **kwargs):
    """
    Process a segment directly without a service instance.

    Args:
        segment_or_id: Speaker segment or segment ID to process
        audio_path: Path to the audio file for this segment
        transcriber_name: Name of the transcriber to use
        **kwargs: Additional arguments for the transcriber

    Returns:
        Transcribed text for the segment
    """
    try:
        # Create a new service instance
        from pyhearingai.transcription.repositories.transcription_repository import (
            TranscriptionRepository,
        )

        service = TranscriptionService(transcriber_name=transcriber_name)

        # Process the segment
        text = service.transcribe_segment(segment_or_id, audio_path, None, **kwargs)

        # Close the service
        service.close()

        return text
    except Exception as e:
        from traceback import format_exc

        logger.error(f"Error processing segment directly: {str(e)}")
        logger.error(format_exc())
        return ""


class TranscriptionService:
    """
    Service for performing speech transcription on audio chunks and speaker segments.

    This service provides methods for transcribing individual audio chunks,
    speaker segments, or entire processing jobs with multiple chunks.
    """

    def __init__(
        self,
        transcriber_name: str = "whisper_openai",
        repository: Optional[TranscriptionRepository] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the transcription service.

        Args:
            transcriber_name: Name of the transcriber to use (default: "whisper_openai")
            repository: Optional repository for storing transcription results
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.transcriber_name = transcriber_name
        self.adapter = WhisperAdapter(transcriber_name)

        # Initialize repository if not provided
        if repository is None:
            logger.debug("No repository provided, creating a default one")
            repository = TranscriptionRepository()
        self.repository = repository

        # Set max workers
        if max_workers is None:
            import multiprocessing

            max_workers = max(1, multiprocessing.cpu_count() - 1)
        self.max_workers = max_workers

        logger.debug(
            f"Initialized TranscriptionService with {transcriber_name} transcriber and {max_workers} workers"
        )

    def _get_chunk_object(self, chunk_or_id: Union[AudioChunk, str]) -> Optional[AudioChunk]:
        """
        Get an AudioChunk object from a chunk or chunk ID.

        Args:
            chunk_or_id: AudioChunk object or chunk ID string

        Returns:
            AudioChunk object, or None if not found or invalid
        """
        from pyhearingai.infrastructure.repositories.json_repositories import JsonChunkRepository

        # If already an AudioChunk, return it
        if isinstance(chunk_or_id, AudioChunk):
            return chunk_or_id

        # If a string ID, try to load from repository
        elif isinstance(chunk_or_id, str):
            try:
                # Get chunk repository
                chunk_repo = JsonChunkRepository()

                # Try to load the chunk
                chunk = chunk_repo.get_by_id(chunk_or_id)
                if chunk:
                    return chunk

                # If not found, create a stub object with just the ID
                logger.warning(f"Chunk {chunk_or_id} not found in repository, creating stub object")
                return AudioChunk(id=chunk_or_id, job_id="", chunk_index=0)

            except Exception as e:
                logger.error(f"Error loading chunk {chunk_or_id}: {str(e)}")
                return None

        # Invalid input
        else:
            logger.error(f"Invalid chunk_or_id type: {type(chunk_or_id)}")
            return None

    def _get_segment_object(
        self, segment_or_id: Union[SpeakerSegment, str]
    ) -> Optional[SpeakerSegment]:
        """
        Get a SpeakerSegment object from a segment or segment ID.

        Args:
            segment_or_id: SpeakerSegment object or segment ID string

        Returns:
            SpeakerSegment object, or None if not found or invalid
        """
        from pyhearingai.infrastructure.repositories.json_repositories import JsonSegmentRepository

        # If already a SpeakerSegment, return it
        if isinstance(segment_or_id, SpeakerSegment):
            return segment_or_id

        # If a string ID, try to load from repository
        elif isinstance(segment_or_id, str):
            try:
                # Get segment repository
                segment_repo = JsonSegmentRepository()

                # Try to load the segment
                segment = segment_repo.get_by_id(segment_or_id)
                if segment:
                    return segment

                # If not found, log error and return None
                logger.warning(f"Segment {segment_or_id} not found in repository")
                return None

            except Exception as e:
                logger.error(f"Error loading segment {segment_or_id}: {str(e)}")
                return None

        # Invalid input
        else:
            logger.error(f"Invalid segment_or_id type: {type(segment_or_id)}")
            return None

    def transcribe_chunk(
        self, chunk_or_id: Union[AudioChunk, str], job: Optional[ProcessingJob] = None, **kwargs
    ) -> List[Segment]:
        """
        Transcribe a single audio chunk.

        Args:
            chunk_or_id: Audio chunk or chunk ID to process
            job: Optional processing job this chunk belongs to
            **kwargs: Additional arguments for the transcriber

        Returns:
            List of transcription segments
        """
        # Get AudioChunk object if ID was provided
        chunk = self._get_chunk_object(chunk_or_id)
        if not chunk:
            logger.warning(f"Cannot transcribe: Invalid chunk or chunk not found")
            return []

        if not chunk.chunk_path or not Path(chunk.chunk_path).exists():
            logger.warning(f"Cannot transcribe chunk {chunk.id}: Invalid path or file not found")
            return []

        job_id = job.id if job else chunk.job_id

        # Check if we have cached results
        if job_id and self.repository.chunk_exists(job_id, chunk.id):
            segments = self.repository.get_chunk_transcription(job_id, chunk.id)
            if segments:
                logger.debug(f"Using cached transcription results for chunk {chunk.id}")
                return segments

        # Perform transcription
        try:
            logger.debug(f"Transcribing chunk {chunk.id} with path {chunk.chunk_path}")
            segments = self.adapter.transcribe_chunk(chunk, **kwargs)

            # Save results if we have a job ID
            if job_id:
                self.repository.save_chunk_transcription(
                    job_id, chunk.id, segments, metadata=kwargs.get("metadata", {})
                )

            return segments

        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk.id}: {str(e)}")
            return []

    def transcribe_segment(
        self,
        segment_or_id: Union[SpeakerSegment, str],
        audio_path: Optional[Path] = None,
        job: Optional[ProcessingJob] = None,
        **kwargs,
    ) -> str:
        """
        Transcribe a single speaker segment.

        Args:
            segment_or_id: Speaker segment or segment ID to process
            audio_path: Path to the audio file for this segment (required if not in segment)
            job: Optional processing job this segment belongs to
            **kwargs: Additional arguments for the transcriber

        Returns:
            Transcribed text for the segment
        """
        # Get SpeakerSegment object if ID was provided
        segment = self._get_segment_object(segment_or_id)
        if not segment:
            logger.warning(f"Cannot transcribe: Invalid segment or segment not found")
            return ""

        # Get audio path - either provided or from segment
        if audio_path is None:
            if hasattr(segment, "audio_path") and segment.audio_path:
                audio_path = Path(segment.audio_path)
            else:
                logger.warning(f"Cannot transcribe segment {segment.id}: No audio path provided")
                return ""

        if not audio_path.exists():
            logger.warning(
                f"Cannot transcribe segment {segment.id}: Audio file not found at {audio_path}"
            )
            return ""

        job_id = job.id if job else segment.job_id if hasattr(segment, "job_id") else None

        # Check if we have cached results
        if job_id and self.repository.segment_exists(job_id, segment.id):
            text = self.repository.get_segment_transcription(job_id, segment.id)
            if text:
                logger.debug(f"Using cached transcription results for segment {segment.id}")
                return text

        # Perform transcription
        try:
            logger.debug(f"Transcribing segment {segment.id} with path {audio_path}")
            text = self.adapter.transcribe_segment(segment, audio_path, **kwargs)

            # Save results if we have a job ID
            if job_id:
                self.repository.save_segment_transcription(
                    job_id, segment.id, text, metadata=kwargs.get("metadata", {})
                )

            return text

        except Exception as e:
            logger.error(f"Error transcribing segment {segment.id}: {str(e)}")
            return ""

    def transcribe_job(
        self, job: ProcessingJob, parallel: bool = True, chunk_batch_size: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe all chunks in a processing job.

        Args:
            job: Processing job to process
            parallel: Whether to use parallel processing
            chunk_batch_size: Number of chunks to process at once in parallel mode
            **kwargs: Additional arguments for the transcriber

        Returns:
            Dictionary with results and statistics
        """
        if not job or not job.id:
            logger.error("Invalid processing job")
            return {"success": False, "error": "Invalid processing job"}

        # Get all chunks for this job
        from pyhearingai.infrastructure.repositories.json_repositories import JsonChunkRepository

        chunk_repo = JsonChunkRepository()
        chunks = chunk_repo.get_by_job_id(job.id)

        if not chunks:
            logger.warning(f"No chunks found for job {job.id}")
            return {"success": True, "chunks_processed": 0, "message": "No chunks found"}

        logger.info(f"Transcribing job {job.id} with {len(chunks)} chunks (parallel={parallel})")

        start_time = time.time()

        # Process chunks
        results = {}
        if parallel:
            results = self._transcribe_job_parallel(job, chunks, chunk_batch_size, **kwargs)
        else:
            results = self._transcribe_job_sequential(job, chunks, **kwargs)

        end_time = time.time()
        duration = end_time - start_time

        # Add statistics
        results["job_id"] = job.id
        results["duration"] = duration
        results["chunks_total"] = len(chunks)

        logger.info(f"Transcription of job {job.id} completed in {duration:.2f}s")
        return results

    def _transcribe_job_parallel(
        self,
        job: ProcessingJob,
        chunks: List[Union[AudioChunk, str]],
        batch_size: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transcribe chunks in parallel using ThreadPoolExecutor.

        Args:
            job: Processing job to process
            chunks: List of chunks to process
            batch_size: Number of chunks to process at once
            **kwargs: Additional arguments for the transcriber

        Returns:
            Dictionary with results and statistics
        """
        logger.info(
            f"Starting parallel transcription of {len(chunks)} chunks with batch size {batch_size}"
        )

        results = {"success": True, "chunks_processed": 0, "chunks_failed": 0, "chunk_results": {}}

        # Process in batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {}
            for chunk in chunks:
                # Skip non-AudioChunk objects
                if not isinstance(chunk, (AudioChunk, str)):
                    logger.warning(f"Skipping non-chunk object: {type(chunk)}")
                    continue

                # Get chunk ID
                chunk_id = chunk.id if isinstance(chunk, AudioChunk) else chunk

                # Skip if already processed (unless forced)
                if not kwargs.get("force", False) and self.repository.chunk_exists(
                    job.id, chunk_id
                ):
                    logger.debug(f"Skipping already processed chunk {chunk_id}")
                    results["chunks_processed"] += 1
                    results["chunk_results"][chunk_id] = {"status": "cached"}
                    continue

                # Submit task to executor
                future = executor.submit(
                    _process_chunk_directly, chunk, self.transcriber_name, **kwargs
                )
                future_to_chunk[future] = chunk_id

            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    segments = future.result()
                    if segments:
                        results["chunks_processed"] += 1
                        results["chunk_results"][chunk_id] = {
                            "status": "success",
                            "segments_count": len(segments),
                        }
                    else:
                        results["chunks_failed"] += 1
                        results["chunk_results"][chunk_id] = {
                            "status": "failed",
                            "error": "No segments returned",
                        }
                except Exception as e:
                    results["chunks_failed"] += 1
                    results["chunk_results"][chunk_id] = {"status": "failed", "error": str(e)}
                    logger.error(f"Error processing chunk {chunk_id}: {str(e)}")

        # Update job status if all chunks processed successfully
        if results["chunks_failed"] == 0:
            logger.info(f"All {results['chunks_processed']} chunks processed successfully")
        else:
            logger.warning(f"{results['chunks_failed']} chunks failed processing")
            results["success"] = results["chunks_failed"] < len(
                chunks
            )  # Still successful if some processed

        return results

    def _transcribe_job_sequential(
        self, job: ProcessingJob, chunks: List[Union[AudioChunk, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe chunks sequentially.

        Args:
            job: Processing job to process
            chunks: List of chunks to process
            **kwargs: Additional arguments for the transcriber

        Returns:
            Dictionary with results and statistics
        """
        logger.info(f"Starting sequential transcription of {len(chunks)} chunks")

        results = {"success": True, "chunks_processed": 0, "chunks_failed": 0, "chunk_results": {}}

        # Process chunks one by one
        for chunk in chunks:
            # Skip non-AudioChunk objects
            if not isinstance(chunk, (AudioChunk, str)):
                logger.warning(f"Skipping non-chunk object: {type(chunk)}")
                continue

            # Get chunk ID
            chunk_id = chunk.id if isinstance(chunk, AudioChunk) else chunk

            # Skip if already processed (unless forced)
            if not kwargs.get("force", False) and self.repository.chunk_exists(job.id, chunk_id):
                logger.debug(f"Skipping already processed chunk {chunk_id}")
                results["chunks_processed"] += 1
                results["chunk_results"][chunk_id] = {"status": "cached"}
                continue

            # Process chunk
            try:
                segments = self.transcribe_chunk(chunk, job, **kwargs)
                if segments:
                    results["chunks_processed"] += 1
                    results["chunk_results"][chunk_id] = {
                        "status": "success",
                        "segments_count": len(segments),
                    }
                else:
                    results["chunks_failed"] += 1
                    results["chunk_results"][chunk_id] = {
                        "status": "failed",
                        "error": "No segments returned",
                    }
            except Exception as e:
                results["chunks_failed"] += 1
                results["chunk_results"][chunk_id] = {"status": "failed", "error": str(e)}
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}")

        # Update job status if all chunks processed successfully
        if results["chunks_failed"] == 0:
            logger.info(f"All {results['chunks_processed']} chunks processed successfully")
        else:
            logger.warning(f"{results['chunks_failed']} chunks failed processing")
            results["success"] = results["chunks_failed"] < len(
                chunks
            )  # Still successful if some processed

        return results

    def close(self):
        """Clean up resources"""
        if self.adapter:
            self.adapter.close()

    def extract_and_transcribe_segments(
        self,
        job: ProcessingJob,
        chunk: Union[AudioChunk, str],
        diarization_segments: List[DiarizationSegment],
        **kwargs,
    ) -> Dict[str, str]:
        """
        Extract and transcribe individual speaker segments from a chunk.

        This method extracts audio for each diarization segment and transcribes them separately.

        Args:
            job: Processing job
            chunk: Audio chunk containing the segments
            diarization_segments: List of diarization segments from the chunk
            **kwargs: Additional arguments for the transcriber

        Returns:
            Dictionary mapping segment IDs to transcribed text
        """
        # Get AudioChunk object if ID was provided
        chunk_obj = self._get_chunk_object(chunk)
        if not chunk_obj:
            logger.warning(f"Cannot extract segments: Invalid chunk or chunk not found")
            return {}

        if not chunk_obj.chunk_path or not Path(chunk_obj.chunk_path).exists():
            logger.warning(
                f"Cannot extract segments from chunk {chunk_obj.id}: Invalid path or file not found"
            )
            return {}

        # Import audio processing utilities
        from pyhearingai.infrastructure.audio import extract_segment

        # Initialize results
        results = {}

        # Create a temp dir for extracted segments
        import tempfile

        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Process each diarization segment
            for i, segment in enumerate(diarization_segments):
                # Create a unique segment ID
                segment_id = f"{chunk_obj.id}_segment_{i}_{segment.speaker_id}"

                # Extract audio for this segment
                segment_path = temp_dir / f"{segment_id}.wav"
                try:
                    extract_segment(chunk_obj.chunk_path, segment_path, segment.start, segment.end)

                    # Create a SpeakerSegment object
                    speaker_segment = SpeakerSegment(
                        id=segment_id,
                        job_id=job.id,
                        chunk_id=chunk_obj.id,
                        speaker_id=segment.speaker_id,
                        start_time=segment.start + chunk_obj.start_time,
                        end_time=segment.end + chunk_obj.start_time,
                        confidence=segment.score if segment.score is not None else 0.0,
                    )

                    # Transcribe the segment
                    text = self.transcribe_segment(speaker_segment, segment_path, job, **kwargs)

                    # Store the result
                    results[segment_id] = text

                except Exception as e:
                    logger.error(f"Error extracting or transcribing segment {segment_id}: {str(e)}")
                    results[segment_id] = ""

            return results

        finally:
            # Clean up temp dir
            import shutil

            shutil.rmtree(temp_dir)

    def transcribe_diarized_chunk(
        self,
        job: ProcessingJob,
        chunk: Union[AudioChunk, str],
        diarization_segments: List[DiarizationSegment],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transcribe a chunk with existing diarization segments.

        This method performs both whole-chunk transcription and segment-level
        transcription, and reconciles the results if needed.

        Args:
            job: Processing job
            chunk: Audio chunk to process
            diarization_segments: List of diarization segments from the chunk
            **kwargs: Additional arguments for the transcriber

        Returns:
            Dictionary with results and statistics
        """
        # Get AudioChunk object if ID was provided
        chunk_obj = self._get_chunk_object(chunk)
        if not chunk_obj:
            logger.warning(f"Cannot transcribe diarized chunk: Invalid chunk or chunk not found")
            return {"success": False, "error": "Invalid chunk"}

        if not chunk_obj.chunk_path or not Path(chunk_obj.chunk_path).exists():
            logger.warning(
                f"Cannot transcribe chunk {chunk_obj.id}: Invalid path or file not found"
            )
            return {"success": False, "error": "Invalid chunk path"}

        # Initialize results
        results = {
            "chunk_id": chunk_obj.id,
            "job_id": job.id,
            "whole_chunk": {"success": False, "segments_count": 0},
            "speaker_segments": {"success": False, "segments_count": 0, "segments": {}},
        }

        # 1. Perform whole-chunk transcription
        try:
            chunk_segments = self.transcribe_chunk(chunk_obj, job, **kwargs)
            results["whole_chunk"]["success"] = True
            results["whole_chunk"]["segments_count"] = len(chunk_segments)
        except Exception as e:
            logger.error(f"Error in whole-chunk transcription: {str(e)}")
            results["whole_chunk"]["error"] = str(e)

        # 2. Perform segment-level transcription
        try:
            segment_results = self.extract_and_transcribe_segments(
                job, chunk_obj, diarization_segments, **kwargs
            )
            results["speaker_segments"]["success"] = True
            results["speaker_segments"]["segments_count"] = len(segment_results)
            results["speaker_segments"]["segments"] = segment_results
        except Exception as e:
            logger.error(f"Error in segment-level transcription: {str(e)}")
            results["speaker_segments"]["error"] = str(e)

        # Overall success if at least one method succeeded
        results["success"] = (
            results["whole_chunk"]["success"] or results["speaker_segments"]["success"]
        )

        return results

    def process_job(
        self,
        job: ProcessingJob,
        chunks: List[AudioChunk],
        show_progress: bool = False,
        chunk_progress_callback=None,
        **kwargs,
    ) -> Dict[str, List[Segment]]:
        """
        Process all chunks for a job with progress tracking.

        Args:
            job: The processing job
            chunks: List of audio chunks to process
            show_progress: Whether to show progress information
            chunk_progress_callback: Callback for per-chunk progress updates
            **kwargs: Additional arguments for the transcriber

        Returns:
            Dictionary mapping chunk IDs to lists of transcript segments
        """
        logger.info(f"Processing transcription for {len(chunks)} chunks in job {job.id}")

        # Initialize results dictionary
        results: Dict[str, List[Segment]] = {}

        # Determine the processing mode
        if self.max_workers == 1 or len(chunks) == 1:
            # Sequential processing
            for i, chunk in enumerate(chunks):
                if show_progress:
                    logger.info(f"Transcribing chunk {i+1}/{len(chunks)}: {chunk.id}")

                # Skip already transcribed chunks
                if chunk.status in [ChunkStatus.TRANSCRIBED, ChunkStatus.COMPLETED]:
                    logger.info(f"Chunk {chunk.id} already transcribed, loading results")
                    segments = self.repository.get_chunk_transcription(job.id, chunk.id)
                    results[chunk.id] = segments

                    if chunk_progress_callback:
                        chunk_progress_callback(chunk.id, 1.0, "Already transcribed")
                    continue

                try:
                    # Process the chunk with progress updates
                    if chunk_progress_callback:
                        # Create a wrapper that updates progress
                        def progress_wrapper(progress, message):
                            chunk_progress_callback(chunk.id, progress, message)

                        # Process with progress tracking (adding progress_callback to kwargs)
                        kwargs_with_progress = dict(kwargs)
                        kwargs_with_progress["progress_callback"] = progress_wrapper
                        segments = self.transcribe_chunk(chunk, job=job, **kwargs_with_progress)
                    else:
                        # Process without progress tracking
                        segments = self.transcribe_chunk(chunk, job=job, **kwargs)

                    results[chunk.id] = segments

                    if chunk_progress_callback:
                        chunk_progress_callback(chunk.id, 1.0, "Transcription complete")
                except Exception as e:
                    logger.error(f"Error transcribing chunk {chunk.id}: {str(e)}")
                    if chunk_progress_callback:
                        chunk_progress_callback(chunk.id, 0.5, f"Error: {str(e)}")
                    raise
        else:
            # Parallel processing
            logger.info(f"Using parallel processing with {self.max_workers} workers")

            # Create a list of tasks to process
            tasks = []
            for chunk in chunks:
                # Skip already transcribed chunks
                if chunk.status in [ChunkStatus.TRANSCRIBED, ChunkStatus.COMPLETED]:
                    logger.info(f"Chunk {chunk.id} already transcribed, loading results")
                    segments = self.repository.get_chunk_transcription(job.id, chunk.id)
                    results[chunk.id] = segments

                    if chunk_progress_callback:
                        chunk_progress_callback(chunk.id, 1.0, "Already transcribed")
                    continue

                # Add the chunk to the task list
                tasks.append(chunk)

            # Process tasks in parallel
            if tasks:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    # Submit all tasks
                    future_to_chunk = {
                        executor.submit(
                            _process_chunk_directly,
                            chunk_or_id=chunk,
                            transcriber_name=self.transcriber_name,
                            job_id=job.id,
                            **kwargs,
                        ): chunk
                        for chunk in tasks
                    }

                    # Process results as they complete
                    for i, future in enumerate(concurrent.futures.as_completed(future_to_chunk)):
                        chunk = future_to_chunk[future]
                        try:
                            segments = future.result()
                            results[chunk.id] = segments

                            if show_progress:
                                logger.info(f"Completed {i+1}/{len(tasks)} chunks")

                            if chunk_progress_callback:
                                chunk_progress_callback(
                                    chunk.id, 1.0, f"Completed {i+1}/{len(tasks)}"
                                )
                        except Exception as e:
                            logger.error(f"Error transcribing chunk {chunk.id}: {str(e)}")
                            if chunk_progress_callback:
                                chunk_progress_callback(chunk.id, 0.5, f"Error: {str(e)}")
                            raise

        return results

    def has_all_chunk_data(self, job_id: str) -> bool:
        """
        Check if all chunks for a job have been transcribed.

        Args:
            job_id: ID of the job to check

        Returns:
            True if all chunks have been transcribed, False otherwise
        """
        # Get all chunks for this job
        from pyhearingai.infrastructure.repositories.json_repositories import JsonChunkRepository

        chunk_repo = JsonChunkRepository()
        chunks = chunk_repo.get_by_job_id(job_id)

        if not chunks:
            return False

        # Check if all chunks have transcription data
        for chunk in chunks:
            if not self.repository.chunk_exists(job_id, chunk.id):
                return False

        return True
