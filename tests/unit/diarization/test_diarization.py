"""
Tests for the DiarizationService class.

This module tests the functionality of the diarization service,
including initialization, chunk processing, job processing, error handling,
and more.
"""

import os
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
from pytest import fixture

from pyhearingai.core.idempotent import AudioChunk, ChunkStatus, ProcessingJob, ProcessingStatus
from pyhearingai.core.models import DiarizationSegment, Segment
from pyhearingai.diarization.repositories.diarization_repository import DiarizationRepository
from pyhearingai.diarization.service import DiarizationService
from pyhearingai.infrastructure.repositories.json_repositories import (
    JsonChunkRepository,
    JsonJobRepository,
)
from tests.conftest import create_processing_job_func


def create_test_job(
    job_id, audio_path, chunks=None, status=None, parallel=False, force_reprocess=False
):
    """
    Create a ProcessingJob instance with the given parameters.
    Handles differences between constructor signatures in different versions.
    """
    job = create_processing_job_func(audio_path=str(audio_path), job_id=job_id, status=status)

    # Add custom attributes
    job.parallel = parallel
    job.force_reprocess = force_reprocess

    # Set chunks if provided
    if chunks is not None:
        job.chunks = chunks

    return job
