"""
Root conftest.py for PyHearingAI tests.

This module sets up global fixtures and configuration for all tests, ensuring a consistent
test environment and behavior.
"""

import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

# Add project root to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Constants for tests
TEST_RESOURCE_DIR = Path(__file__).parent / "fixtures" / "resources"


@pytest.fixture(scope="session")
def test_resource_dir():
    """Return the path to the test resources directory."""
    return TEST_RESOURCE_DIR


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    # Cleanup temp files/directories
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def reset_logging():
    """Reset logging configuration before and after each test."""
    import logging

    # Store original logging config
    original_loggers = {}
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        original_loggers[logger_name] = {
            "level": logger.level,
            "handlers": logger.handlers.copy(),
            "disabled": logger.disabled,
            "propagate": logger.propagate,
        }

    # Store root logger config
    root_logger = logging.getLogger()
    original_root_logger = {
        "level": root_logger.level,
        "handlers": root_logger.handlers.copy(),
        "disabled": root_logger.disabled,
    }

    # Reset for test
    yield

    # Restore original logging config
    for logger_name, config in original_loggers.items():
        logger = logging.getLogger(logger_name)
        logger.level = config["level"]
        logger.handlers = config["handlers"]
        logger.disabled = config["disabled"]
        logger.propagate = config["propagate"]

    # Restore root logger
    root_logger = logging.getLogger()
    root_logger.level = original_root_logger["level"]
    root_logger.handlers = original_root_logger["handlers"]
    root_logger.disabled = original_root_logger["disabled"]


@pytest.fixture
def mock_environment():
    """Mock environment variables needed for tests and restore original values afterward."""
    import os

    # Store original environment values
    original_env = {}
    test_env_vars = [
        "OPENAI_API_KEY",
        "ASSEMBLYAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "WHISPERAPI_API_KEY",
        "PYHEARING_LOG_LEVEL",
        "PYHEARING_CONFIG_PATH",
    ]

    for var in test_env_vars:
        original_env[var] = os.environ.get(var)

    # Set test values
    os.environ["OPENAI_API_KEY"] = "test-openai-key"
    os.environ["ASSEMBLYAI_API_KEY"] = "test-assemblyai-key"
    os.environ["DEEPGRAM_API_KEY"] = "test-deepgram-key"
    os.environ["WHISPERAPI_API_KEY"] = "test-whisperapi-key"
    os.environ["PYHEARING_LOG_LEVEL"] = "DEBUG"

    yield

    # Restore original values
    for var, value in original_env.items():
        if value is None:
            if var in os.environ:
                del os.environ[var]
        else:
            os.environ[var] = value


@pytest.fixture(scope="session")
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def example_audio_path(fixtures_dir):
    """Return the path to the example audio file."""
    return fixtures_dir / "resources" / "example_audio.m4a"


@pytest.fixture(scope="session")
def reference_transcript_path(fixtures_dir):
    """Return the path to the reference transcript file."""
    return fixtures_dir / "resources" / "labeled_transcript.txt"


def create_temp_audio_file(suffix=".wav"):
    """Helper function to create a temporary audio file."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    file_handle = os.fdopen(fd, "wb")
    return Path(path), file_handle


@pytest.fixture
def temp_audio_file():
    """Create a temporary WAV file for testing."""
    path, file_handle = create_temp_audio_file(suffix=".wav")

    # Write some dummy content to make it a valid file
    file_handle.write(
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    file_handle.flush()

    yield path

    # Clean up
    file_handle.close()
    if path.exists():
        path.unlink()


def create_segment(start, end, text, speaker_id=None):
    """Helper function to create a transcript segment."""
    try:
        from pyhearingai.core.models import Segment

        return Segment(text=text, start=start, end=end, speaker=speaker_id)
    except ImportError:
        # Simple dict-based fallback if the module is not available
        return {"text": text, "start": start, "end": end, "speaker": speaker_id}


@pytest.fixture
def transcript_segments():
    """Return a list of transcript segments for testing."""
    return [
        create_segment(start=0.0, end=2.0, text="This is a test."),
        create_segment(start=2.5, end=4.5, text="Testing the transcriber."),
        create_segment(start=5.0, end=7.0, text="Final test segment."),
    ]


@pytest.fixture
def diarization_segments():
    """Return a list of diarization segments for testing."""
    return [
        create_segment(start=0.0, end=2.0, text="", speaker_id="SPEAKER_00"),
        create_segment(start=2.5, end=4.5, text="", speaker_id="SPEAKER_01"),
        create_segment(start=5.0, end=7.0, text="", speaker_id="SPEAKER_00"),
    ]


@pytest.fixture
def labeled_segments():
    """Return a list of segments with speaker labels for testing."""
    return [
        create_segment(start=0.0, end=2.0, text="This is a test.", speaker_id="SPEAKER_00"),
        create_segment(
            start=2.5, end=4.5, text="Testing the transcriber.", speaker_id="SPEAKER_01"
        ),
        create_segment(start=5.0, end=7.0, text="Final test segment.", speaker_id="SPEAKER_00"),
    ]


@pytest.fixture
def formatter_segments_with_speakers():
    """Return a list of segments with speaker IDs for formatter testing."""
    return [
        create_segment(start=0.0, end=2.0, text="This is segment one.", speaker_id="Speaker 1"),
        create_segment(start=2.5, end=4.5, text="This is segment two.", speaker_id="Speaker 2"),
        create_segment(start=5.0, end=7.0, text="This is segment three.", speaker_id="Speaker 1"),
    ]


@pytest.fixture
def formatter_segments_without_speakers():
    """Return a list of segments without speaker IDs for formatter testing."""
    return [
        create_segment(start=0.0, end=2.0, text="This is segment one.", speaker_id=None),
        create_segment(start=2.5, end=4.5, text="This is segment two.", speaker_id=None),
    ]


@pytest.fixture
def formatter_single_segment():
    """Return a single segment for simple formatter tests."""
    return [
        create_segment(start=0.0, end=2.0, text="This is a test segment.", speaker_id="Speaker 1"),
    ]


# Import at the end to avoid circular dependencies
from tests.fixtures.audio_fixtures import create_multi_speaker_audio, create_test_audio


def create_processing_job_func(audio_path, job_id=None, status=None):
    """
    Create a ProcessingJob instance with the given parameters.

    Args:
        audio_path: Path to the audio file
        job_id: Optional job ID, will be generated if not provided
        status: Optional processing status

    Returns:
        ProcessingJob: A new processing job instance
    """
    from pyhearingai.core.idempotent import ProcessingJob, ProcessingStatus

    if job_id is None:
        job_id = str(uuid.uuid4())

    # Create the job with the correct constructor parameters
    job = ProcessingJob(
        original_audio_path=str(audio_path),
        id=job_id,
    )

    # Set the status if provided
    if status is not None:
        job.status = status

    return job
