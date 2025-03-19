"""
Test helpers for PyHearingAI tests.

This module provides fixtures, mocks, and utilities for unit and integration tests.
"""

import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf  # Use soundfile instead of librosa.output

from pyhearingai.core.idempotent import (
    AudioChunk,
    ChunkStatus,
    ProcessingJob,
    ProcessingStatus,
    SpeakerSegment,
)
from pyhearingai.core.models import DiarizationSegment, Segment
from pyhearingai.diarization.repositories.diarization_repository import DiarizationRepository
from pyhearingai.diarization.service import DiarizationService
from pyhearingai.reconciliation.repositories.reconciliation_repository import (
    ReconciliationRepository,
)
from pyhearingai.reconciliation.service import ReconciliationService
from pyhearingai.transcription.repositories.transcription_repository import TranscriptionRepository
from pyhearingai.transcription.service import TranscriptionService


class TestFixtures:
    """Class for creating test fixtures and sample data."""

    @staticmethod
    def create_test_audio(path, duration=5.0):
        """Create a test audio file for testing."""
        # Generate a simple sine wave
        sr = 16000  # Sample rate
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Save as WAV file using soundfile
        sf.write(path, y, sr, format="WAV")
        return path

    @staticmethod
    def create_test_job(audio_path=None):
        """Create a test processing job."""
        # If no audio path provided, create a temporary one
        if audio_path is None:
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, "test_audio.wav")
            TestFixtures.create_test_audio(audio_path)

        # Create a job with a unique ID
        job_id = f"test-job-{uuid.uuid4()}"

        # Create the job object
        job = ProcessingJob(
            id=job_id,
            original_audio_path=audio_path,
            chunk_duration=5.0,  # Set a positive chunk duration
            overlap_duration=1.0,  # Set a reasonable overlap
        )

        # Status is already set to PENDING in the constructor
        # No need to explicitly set it here

        return job

    @staticmethod
    def create_test_chunk(job_id, index=0, start_time=0.0, end_time=5.0):
        """Create a test audio chunk."""
        chunk_id = f"{job_id}_chunk_{index:04d}"

        # Create a temporary chunk path
        chunk_path = Path(f"/tmp/test_chunks/{job_id}/chunk_{index:04d}.wav")

        chunk = AudioChunk(
            id=chunk_id,
            job_id=job_id,
            chunk_index=index,  # Changed from index to chunk_index
            chunk_path=chunk_path,  # Added chunk_path parameter
            start_time=start_time,
            end_time=end_time,
            status=ChunkStatus.PENDING,  # Use the enum instead of string
        )

        return chunk


class MockServices:
    """Class for creating mock service instances for testing."""

    @staticmethod
    def mock_diarization_service():
        """Create a mock DiarizationService."""
        # Create a mock service
        mock_service = MagicMock(spec=DiarizationService)

        # Define a mock process_chunk function
        def mock_process_chunk(chunk_id_or_obj, **kwargs):
            # Return mock diarization results
            return [
                DiarizationSegment(speaker_id="SPEAKER_1", start=0.0, end=2.0),
                DiarizationSegment(speaker_id="SPEAKER_2", start=2.0, end=5.0),
            ]

        # Define a mock diarize_audio function
        def mock_diarize_audio(audio_path, **kwargs):
            # Return mock diarization results
            return [
                DiarizationSegment(speaker_id="SPEAKER_1", start=0.0, end=2.0),
                DiarizationSegment(speaker_id="SPEAKER_2", start=2.0, end=5.0),
            ]

        # Define a mock diarize_chunk function
        def mock_diarize_chunk(chunk, job=None, **kwargs):
            # Return mock diarization results as DiarizationSegment objects
            return [
                DiarizationSegment(speaker_id="SPEAKER_1", start=0.0, end=2.0),
                DiarizationSegment(speaker_id="SPEAKER_2", start=2.0, end=5.0),
            ]

        # Define a mock diarize_job function
        def mock_diarize_job(job, chunk_repo=None, **kwargs):
            # Return mock diarization results for each chunk
            result = {}
            # Use the chunks from the job if available
            if hasattr(job, "chunks") and job.chunks:
                for chunk_id in job.chunks:
                    result[chunk_id] = [
                        DiarizationSegment(speaker_id="SPEAKER_1", start=0.0, end=2.0),
                        DiarizationSegment(speaker_id="SPEAKER_2", start=2.0, end=5.0),
                    ]
            else:
                # Otherwise use a default chunk ID
                result["test_chunk"] = [
                    DiarizationSegment(speaker_id="SPEAKER_1", start=0.0, end=2.0),
                    DiarizationSegment(speaker_id="SPEAKER_2", start=2.0, end=5.0),
                ]
            return result

        # Define mock process_job function
        def mock_process_job(
            job, chunks, show_progress=False, chunk_progress_callback=None, **kwargs
        ):
            # Return mock diarization results for each chunk
            result = {}
            for chunk in chunks:
                chunk_id = chunk.id if hasattr(chunk, "id") else chunk
                result[chunk_id] = [
                    DiarizationSegment(speaker_id="SPEAKER_1", start=0.0, end=2.0),
                    DiarizationSegment(speaker_id="SPEAKER_2", start=2.0, end=5.0),
                ]
            return result

        # Set up the mock service
        mock_service.process_chunk = MagicMock(side_effect=mock_process_chunk)
        mock_service.diarize_audio = MagicMock(side_effect=mock_diarize_audio)
        mock_service.diarize_chunk = MagicMock(side_effect=mock_diarize_chunk)
        mock_service.diarize_job = MagicMock(side_effect=mock_diarize_job)
        mock_service.process_job = MagicMock(side_effect=mock_process_job)

        # Add close method
        mock_service.close = MagicMock()

        # Add a repository attribute
        mock_service.repository = MagicMock(spec=DiarizationRepository)

        return mock_service

    @staticmethod
    def mock_transcription_service():
        """Create a mock TranscriptionService."""
        # Create a mock service
        mock_service = MagicMock(spec=TranscriptionService)

        # Define a mock process_chunk function
        def mock_process_chunk(chunk_id_or_obj, diarization_result=None, **kwargs):
            # Mock transcription results
            if diarization_result:
                # Transcribe each segment
                segments = []
                for segment in diarization_result:
                    segments.append(
                        {
                            "speaker": segment["speaker"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": f"This is a mock transcription for {segment['speaker']}.",
                        }
                    )
                return segments
            else:
                # Return a default segment if no diarization
                return [
                    {
                        "speaker": "UNKNOWN",
                        "start": 0.0,
                        "end": 5.0,
                        "text": "This is a mock transcription without diarization.",
                    }
                ]

        # Define a mock transcribe_chunk method that returns Segment objects
        def mock_transcribe_chunk(chunk_or_id, job=None, **kwargs):
            return [Segment(text="This is a mock transcription for chunk.", start=0.0, end=5.0)]

        # Set up the mock service
        mock_service.process_chunk = MagicMock(side_effect=mock_process_chunk)
        mock_service.transcribe_chunk = MagicMock(side_effect=mock_transcribe_chunk)

        # Also mock the transcribe_audio method which might be used
        def mock_transcribe_audio(audio_path, **kwargs):
            return [
                {
                    "speaker": "UNKNOWN",
                    "start": 0.0,
                    "end": 5.0,
                    "text": "This is a mock transcription from transcribe_audio.",
                }
            ]

        mock_service.transcribe_audio = MagicMock(side_effect=mock_transcribe_audio)

        # Define process_job method
        def mock_process_job(
            job, chunks, show_progress=False, chunk_progress_callback=None, **kwargs
        ):
            result = {}
            for chunk in chunks:
                chunk_id = chunk.id if hasattr(chunk, "id") else chunk
                result[chunk_id] = [
                    Segment(
                        text=f"This is a mock transcription for chunk {chunk_id}.",
                        start=0.0,
                        end=5.0,
                    )
                ]
            return result

        mock_service.process_job = MagicMock(side_effect=mock_process_job)

        # Define transcribe_job method
        def mock_transcribe_job(job, parallel=True, chunk_batch_size=5, **kwargs):
            return {
                "segments": [
                    Segment(text="This is a mock transcription for job.", start=0.0, end=5.0)
                ],
                "job_id": job.id if hasattr(job, "id") else job,
            }

        mock_service.transcribe_job = MagicMock(side_effect=mock_transcribe_job)

        # Define transcribe_diarized_chunk method
        def mock_transcribe_diarized_chunk(job, chunk, diarization_segments, **kwargs):
            segments = []
            for i, diarization_segment in enumerate(diarization_segments):
                segments.append(
                    Segment(
                        text=f"This is a mock transcription for segment {i}.",
                        start=diarization_segment.start,
                        end=diarization_segment.end,
                        speaker_id=diarization_segment.speaker_id,
                    )
                )
            return {"segments": segments}

        mock_service.transcribe_diarized_chunk = MagicMock(
            side_effect=mock_transcribe_diarized_chunk
        )

        # Define extract_and_transcribe_segments method
        def mock_extract_and_transcribe_segments(job, chunk, diarization_segments, **kwargs):
            result = {}
            for i, diarization_segment in enumerate(diarization_segments):
                segment_id = f"{chunk.id if hasattr(chunk, 'id') else chunk}_segment_{i}"
                result[segment_id] = f"This is a mock transcription for segment {i}."
            return result

        mock_service.extract_and_transcribe_segments = MagicMock(
            side_effect=mock_extract_and_transcribe_segments
        )

        # Add close method
        mock_service.close = MagicMock()

        # Add repository attribute
        mock_service.repository = MagicMock(spec=TranscriptionRepository)

        return mock_service

    @staticmethod
    def mock_reconciliation_service():
        """Create a mock ReconciliationService."""
        # Create a mock service
        mock_service = MagicMock(spec=ReconciliationService)

        # Define a mock reconcile function
        def mock_reconcile(job_or_id, **kwargs):
            # Create SpeakerSegment objects directly
            return [
                SpeakerSegment(
                    job_id="test-job",
                    chunk_id="test-chunk-0",
                    speaker_id="Speaker 1",
                    start_time=0.0,
                    end_time=2.0,
                    metadata={"text": "This is speaker one."},
                ),
                SpeakerSegment(
                    job_id="test-job",
                    chunk_id="test-chunk-0",
                    speaker_id="Speaker 2",
                    start_time=2.0,
                    end_time=5.0,
                    metadata={"text": "This is speaker two."},
                ),
            ]

        # Set up the mock service
        mock_service.reconcile = MagicMock(side_effect=mock_reconcile)

        # Add repository attributes
        mock_service.repository = MagicMock(spec=ReconciliationRepository)
        mock_service.diarization_repository = MagicMock(spec=DiarizationRepository)
        mock_service.transcription_repository = MagicMock(spec=TranscriptionRepository)

        return mock_service
