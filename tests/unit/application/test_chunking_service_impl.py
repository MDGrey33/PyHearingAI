"""
Unit tests for the ChunkingServiceImpl class.

This module tests the functionality of the chunking service implementation,
focusing on chunk boundary calculation, audio chunking operations, and silence detection.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyhearingai.application.chunking_service_impl import ChunkingServiceImpl
from pyhearingai.core.domain.api_constraints import ApiProvider
from pyhearingai.core.domain.audio_quality import AudioQualitySpecification


@pytest.fixture
def reset_event_publisher():
    """Reset the event publisher before each test."""
    from pyhearingai.core.domain.events import EventPublisher

    # Save original subscribers
    original_subscribers = EventPublisher._subscribers.copy()

    # Clear subscribers
    EventPublisher._subscribers.clear()

    yield

    # Restore original subscribers
    EventPublisher._subscribers = original_subscribers


@pytest.fixture
def example_audio_path(tmp_path):
    """Create a sample audio file for testing."""
    audio_path = tmp_path / "test_audio.wav"
    with open(audio_path, "wb") as f:
        # Write minimal WAV header + some audio data
        f.write(
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
    return audio_path


@pytest.fixture
def mock_audio_format_service():
    """Create a mock AudioFormatService for testing."""
    mock_service = MagicMock()

    # Configure extract_audio_segment method
    def mock_extract(audio_path, output_path, start_time, end_time, quality_spec):
        """Mock implementation that writes a dummy file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt")
        return output_path

    mock_service.extract_audio_segment.side_effect = mock_extract

    # Configure detect_silence method
    mock_service.detect_silence.return_value = [
        {"start": 0.5, "end": 1.0},
        {"start": 2.5, "end": 3.0},
    ]

    return mock_service


@pytest.fixture
def mock_audio_converter():
    """Create a mock SizeAwareFFmpegConverter for testing."""
    mock_converter = MagicMock()

    # Configure convert_with_quality_spec method
    def mock_convert(input_path, quality_spec):
        """Mock implementation that returns the input path and dummy metadata."""
        return input_path, {"duration": 10.0, "format": "wav", "size_bytes": 1024}

    mock_converter.convert_with_quality_spec.side_effect = mock_convert

    # Configure convert_with_size_constraint method
    mock_converter.convert_with_size_constraint.side_effect = (
        lambda input_path, max_size, format_str: (
            input_path,
            {"duration": 10.0, "format": format_str, "size_bytes": max_size // 2},
        )
    )

    # Configure estimate_output_size method
    mock_converter.estimate_output_size.return_value = 1024

    return mock_converter


@pytest.fixture
def mock_event_publisher():
    """Mock event publisher to capture events."""
    events = []

    def capture_event(event):
        events.append(event)

    with patch("pyhearingai.core.domain.events.EventPublisher.publish") as mock_publish:
        mock_publish.side_effect = capture_event
        yield mock_publish, events


class TestChunkingServiceImpl:
    """Unit tests for ChunkingServiceImpl class."""

    def test_init(self, mock_audio_format_service, mock_audio_converter):
        """Test initialization of the chunking service."""
        # Initialize with provided dependencies
        service = ChunkingServiceImpl(
            audio_format_service=mock_audio_format_service, audio_converter=mock_audio_converter
        )

        # Verify dependencies are set
        assert service.audio_format_service == mock_audio_format_service
        assert service.audio_converter == mock_audio_converter

        # Initialize with default dependencies
        service = ChunkingServiceImpl()

        # Verify defaults are set
        assert service.audio_format_service is not None
        assert service.audio_converter is not None

    def test_calculate_chunk_boundaries_basic(self):
        """Test basic chunk boundary calculation without overlap."""
        service = ChunkingServiceImpl()

        # Test with basic parameters
        boundaries = service.calculate_chunk_boundaries(audio_duration=10.0, chunk_duration=2.0)

        # Verify correct boundaries
        assert len(boundaries) == 5
        assert boundaries[0] == (0.0, 2.0)
        assert boundaries[1] == (2.0, 4.0)
        assert boundaries[2] == (4.0, 6.0)
        assert boundaries[3] == (6.0, 8.0)
        assert boundaries[4] == (8.0, 10.0)

    def test_calculate_chunk_boundaries_with_overlap(self):
        """Test chunk boundary calculation with overlap."""
        service = ChunkingServiceImpl()

        # Test with overlap
        boundaries = service.calculate_chunk_boundaries(
            audio_duration=10.0, chunk_duration=3.0, overlap_duration=1.0
        )

        # Verify correct boundaries with overlap
        assert len(boundaries) == 5
        assert boundaries[0] == (0.0, 3.0)
        assert boundaries[1] == (2.0, 5.0)
        assert boundaries[2] == (4.0, 7.0)
        assert boundaries[3] == (6.0, 9.0)
        assert boundaries[4] == (8.0, 10.0)

    def test_calculate_chunk_boundaries_with_time_range(self):
        """Test chunk boundary calculation with time range constraints."""
        service = ChunkingServiceImpl()

        # Test with time range
        boundaries = service.calculate_chunk_boundaries(
            audio_duration=10.0, chunk_duration=2.0, start_time=2.0, end_time=8.0
        )

        # Verify correct boundaries within time range
        assert len(boundaries) == 3
        assert boundaries[0] == (2.0, 4.0)
        assert boundaries[1] == (4.0, 6.0)
        assert boundaries[2] == (6.0, 8.0)

    @pytest.mark.skip(reason="Test needs to be reimplemented")
    def test_create_audio_chunks(
        self,
        example_audio_path,
        mock_audio_format_service,
        mock_audio_converter,
        mock_event_publisher,
        tmp_path,
        reset_event_publisher,
    ):
        """
        Verify that audio can be chunked according to specified boundaries.

        The test should verify:
        - Audio chunks are created at specified boundaries
        - Output files are created in the correct location
        - Events are published for chunking operations
        - Returns list of paths to created chunks
        """
        pass

    @pytest.mark.skip(reason="Test needs to be reimplemented")
    def test_create_audio_chunks_with_api_provider(
        self,
        example_audio_path,
        mock_audio_format_service,
        mock_audio_converter,
        mock_event_publisher,
        tmp_path,
        reset_event_publisher,
    ):
        """
        Verify that chunking respects API provider constraints.

        The test should verify:
        - API provider constraints are applied during chunking
        - Size limits are properly checked against provider requirements
        - File format constraints are correctly applied
        """
        pass

    @pytest.mark.skip(reason="Test needs to be reimplemented")
    def test_create_audio_chunks_with_size_constraint(
        self,
        example_audio_path,
        mock_audio_format_service,
        mock_audio_converter,
        mock_event_publisher,
        tmp_path,
        reset_event_publisher,
    ):
        """
        Verify that chunking handles size constraints correctly.

        The test should verify:
        - Chunks exceeding size limits are properly handled
        - Quality reduction is applied when needed to meet size constraints
        - Size-constrained conversion is attempted for oversized chunks
        """
        pass

    @pytest.mark.skip(reason="Test needs to be reimplemented")
    def test_create_audio_chunks_exceeding_size(
        self,
        example_audio_path,
        mock_audio_format_service,
        mock_audio_converter,
        mock_event_publisher,
        tmp_path,
        reset_event_publisher,
    ):
        """
        Verify behavior when chunks exceed size limits even after reduction.

        The test should verify:
        - Oversized chunks are properly reported
        - Events include information about oversized chunks
        - System continues processing remaining chunks
        """
        pass

    def test_detect_silence(
        self, example_audio_path, mock_audio_format_service, mock_audio_converter
    ):
        """Test silence detection functionality."""
        # Initialize service with mocks
        service = ChunkingServiceImpl(
            audio_format_service=mock_audio_format_service, audio_converter=mock_audio_converter
        )

        # Patch Path.exists to return True for our test file path
        with patch("pathlib.Path.exists", return_value=True):
            # Call detect_silence with default parameters
            silence_regions = service.detect_silence(example_audio_path)

            # Verify results
            assert silence_regions == [(0.5, 1.0), (2.5, 3.0)]
            mock_audio_format_service.detect_silence.assert_called_once_with(
                example_audio_path, 0.5, -40
            )

    def test_detect_silence_custom_params(
        self, example_audio_path, mock_audio_format_service, mock_audio_converter
    ):
        """Test silence detection with custom parameters."""
        # Initialize service with mocks
        service = ChunkingServiceImpl(
            audio_format_service=mock_audio_format_service, audio_converter=mock_audio_converter
        )

        # Patch Path.exists to return True for our test file path
        with patch("pathlib.Path.exists", return_value=True):
            # Call detect_silence with custom parameters
            silence_regions = service.detect_silence(
                example_audio_path, min_silence_duration=1.0, silence_threshold=-50
            )

            # Verify results with custom parameters
            assert silence_regions == [(0.5, 1.0), (2.5, 3.0)]
            mock_audio_format_service.detect_silence.assert_called_once_with(
                example_audio_path, 1.0, -50
            )
