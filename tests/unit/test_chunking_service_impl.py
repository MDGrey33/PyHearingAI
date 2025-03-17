"""
Tests for the ChunkingServiceImpl class.

These tests verify the functionality for calculating chunk boundaries,
creating audio chunks, and detecting silence in audio files.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pytest
import soundfile as sf

from pyhearingai.application.chunking_service_impl import ChunkingServiceImpl
from pyhearingai.core.ports import AudioFormatService
from pyhearingai.core.domain.audio_quality import (
    AudioQualitySpecification,
    AudioFormat,
    AudioCodec
)
from pyhearingai.core.domain.api_constraints import ApiProvider, ApiSizeLimitPolicy
from pyhearingai.core.domain.events import (
    AudioSizeExceededEvent,
    ChunkingEvent,
    EventPublisher
)
from pyhearingai.infrastructure.adapters.audio_format_service import FFmpegAudioFormatService
from pyhearingai.infrastructure.adapters.size_aware_audio_converter import SizeAwareFFmpegConverter

# Reset EventPublisher before each test
@pytest.fixture(autouse=True)
def reset_event_publisher():
    EventPublisher.clear_subscribers()
    yield

@pytest.fixture
def example_audio_path(tmp_path):
    """Create a sample audio file for testing."""
    audio_file = tmp_path / "example.wav"
    with open(audio_file, 'w') as f:
        f.write("fake audio content")
    return audio_file

@pytest.fixture
def mock_audio_format_service():
    """Create a mock audio format service."""
    # Create a MagicMock without spec to allow any method
    mock_service = MagicMock()
    mock_service.get_audio_duration.return_value = 10.0  # 10 seconds
    
    # Mock extract_audio_segment to return a path
    def mock_extract(audio_path, output_path, start_time, end_time, quality_spec):
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a simple dummy file
        with open(output_path, 'w') as f:
            f.write(f"Chunk from {start_time} to {end_time}")
        return output_path
    
    mock_service.extract_audio_segment.side_effect = mock_extract
    
    # Mock detect_silence to return some silence regions
    # The ChunkingServiceImpl.detect_silence method expects a list of dicts with 'start' and 'end' keys
    mock_service.detect_silence.return_value = [
        {'start': 1.0, 'end': 2.0},
        {'start': 5.0, 'end': 6.0}
    ]
    
    return mock_service

@pytest.fixture
def mock_audio_converter():
    """Create a mock size-aware audio converter."""
    mock_converter = MagicMock(spec=SizeAwareFFmpegConverter)
    
    # Mock convert_with_quality_spec to return a path and metadata
    def mock_convert(audio_path, quality_spec, **kwargs):
        output_dir = kwargs.get('output_dir', Path(tempfile.gettempdir()))
        output_path = output_dir / f"converted_{audio_path.name}"
        
        # Create parent directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a simple dummy file
        with open(output_path, 'w') as f:
            f.write(f"Converted {audio_path}")
        
        metadata = {
            'original_size': 100000,
            'converted_size': 50000,
            'compression_ratio': 2.0,
            'original_format': 'wav',
            'adjustments_made': ['bitrate', 'channels']
        }
        
        return output_path, metadata
    
    mock_converter.convert_with_quality_spec.side_effect = mock_convert
    
    # Mock convert_with_size_constraint with similar functionality
    def mock_convert_with_constraint(audio_path, max_size, target_format="wav", **kwargs):
        output_dir = kwargs.get('output_dir', Path(tempfile.gettempdir()))
        output_path = output_dir / f"converted_constrained_{audio_path.name}"
        
        # Create parent directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a simple dummy file
        with open(output_path, 'w') as f:
            f.write(f"Converted with constraint {audio_path}")
        
        metadata = {
            'original_size': 100000,
            'converted_size': 40000,
            'compression_ratio': 2.5,
            'original_format': 'wav',
            'adjustments_made': ['bitrate', 'channels']
        }
        
        return output_path, metadata
    
    mock_converter.convert_with_size_constraint.side_effect = mock_convert_with_constraint
    
    # Mock estimate_output_size to return a reasonable size estimate
    mock_converter.estimate_output_size.return_value = 50000
    
    return mock_converter

@pytest.fixture
def mock_event_publisher():
    """Create a mock event publisher."""
    with patch.object(EventPublisher, 'publish', autospec=True) as mock_publish:
        yield mock_publish

class TestChunkingServiceImpl:
    """Test suite for ChunkingServiceImpl."""
    
    def test_init(self):
        """Test initializing ChunkingServiceImpl."""
        audio_format_service = MagicMock()
        audio_converter = MagicMock()
        
        service = ChunkingServiceImpl(audio_format_service, audio_converter)
        
        assert service.audio_format_service == audio_format_service
        assert service.audio_converter == audio_converter
    
    def test_calculate_chunk_boundaries_basic(self):
        """Test basic chunk boundary calculation."""
        service = ChunkingServiceImpl(MagicMock(), MagicMock())
        
        # Mock duration
        duration = 10.0  # 10 seconds
        
        # Calculate chunk boundaries with a chunk duration of 5 seconds
        boundaries = service.calculate_chunk_boundaries(duration, 5.0)
        
        # Expect 2 chunks: 0-5, 5-10
        assert len(boundaries) == 2
        assert boundaries[0] == (0.0, 5.0)
        assert boundaries[1] == (5.0, 10.0)
    
    def test_calculate_chunk_boundaries_with_overlap(self):
        """Test chunk boundary calculation with overlap."""
        service = ChunkingServiceImpl(MagicMock(), MagicMock())
        
        # Mock duration
        duration = 10.0  # 10 seconds
        
        # Calculate chunk boundaries with a chunk duration of 5 seconds and 1 second overlap
        boundaries = service.calculate_chunk_boundaries(duration, 5.0, 1.0)
        
        # Expect 3 chunks: 0-5, 4-9, 8-10
        assert len(boundaries) == 3
        assert boundaries[0] == (0.0, 5.0)
        assert boundaries[1] == (4.0, 9.0)
        assert boundaries[2] == (8.0, 10.0)
    
    def test_calculate_chunk_boundaries_with_time_range(self):
        """Test chunk boundary calculation with a time range."""
        service = ChunkingServiceImpl(MagicMock(), MagicMock())
        
        # Mock duration
        duration = 20.0  # 20 seconds
        
        # Calculate chunk boundaries for a specific range (5s-15s)
        boundaries = service.calculate_chunk_boundaries(
            duration, 
            chunk_duration=5.0, 
            overlap_duration=0.0, 
            start_time=5.0, 
            end_time=15.0
        )
        
        # Expect 2 chunks: 5-10, 10-15
        assert len(boundaries) == 2
        assert boundaries[0] == (5.0, 10.0)
        assert boundaries[1] == (10.0, 15.0)
    
    def test_create_audio_chunks(self, example_audio_path, tmp_path, 
                               mock_audio_format_service, mock_audio_converter, mock_event_publisher):
        """Test creation of audio chunks."""
        # Create service with mocked dependencies
        service = ChunkingServiceImpl(mock_audio_format_service, mock_audio_converter)
        
        # Define chunk boundaries
        boundaries = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
        
        # Create output directory
        output_dir = tmp_path / "chunks"
        
        # Use a quality specification
        quality_spec = AudioQualitySpecification.for_whisper_api()
        
        # Create chunks
        chunk_paths = service.create_audio_chunks(
            example_audio_path,
            output_dir,
            boundaries,
            quality_spec
        )
        
        # Verify that extract_audio_segment was called for each boundary
        assert mock_audio_format_service.extract_audio_segment.call_count == len(boundaries)
        
        # Verify that the chunk paths list has the correct length
        assert len(chunk_paths) == len(boundaries)
        
        # Verify that ChunkingEvent was published
        mock_event_publisher.assert_called()
        event = mock_event_publisher.call_args[0][0]
        assert isinstance(event, ChunkingEvent)
        assert event.source_path == example_audio_path
        assert event.chunk_count == len(boundaries)
        assert len(event.chunk_paths) == len(boundaries)
    
    @patch('pyhearingai.core.domain.audio_validation.AudioValidationService.validate_audio_file')
    @patch('pyhearingai.core.domain.audio_validation.AudioValidationService.suggest_quality_reduction')
    def test_create_audio_chunks_with_api_provider(self, mock_suggest_quality, mock_validate_file,
                                                example_audio_path, tmp_path, 
                                                mock_audio_format_service, mock_audio_converter, 
                                                mock_event_publisher):
        """Test creation of audio chunks with API provider constraints."""
        # Mock validation to fail for the first chunk
        mock_validate_file.side_effect = [
            (False, "File too large"),  # First chunk fails validation
            (True, None),               # Second chunk passes
            (True, None)                # Third chunk passes
        ]
        
        # Mock quality reduction to return a new spec
        reduced_spec = AudioQualitySpecification.for_whisper_api()
        mock_suggest_quality.return_value = reduced_spec
        
        # Create service with mocked dependencies
        service = ChunkingServiceImpl(mock_audio_format_service, mock_audio_converter)
        
        # Define chunk boundaries
        boundaries = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
        
        # Create output directory
        output_dir = tmp_path / "chunks"
        
        # Use a quality specification with standard settings
        quality_spec = AudioQualitySpecification.for_whisper_api()
        
        # Create chunks with API provider
        chunk_paths = service.create_audio_chunks(
            example_audio_path,
            output_dir,
            boundaries,
            quality_spec,
            ApiProvider.OPENAI_WHISPER
        )
        
        # Verify that convert_with_quality_spec was called for the first chunk
        assert mock_audio_converter.convert_with_quality_spec.call_count > 0
        
        # Verify that the chunk paths list has the correct length
        assert len(chunk_paths) == len(boundaries)
    
    @patch.object(ApiSizeLimitPolicy, 'get_limit_for_provider')
    def test_create_audio_chunks_with_size_constraint(self, mock_get_limit,
                                                   example_audio_path, tmp_path,
                                                   mock_audio_format_service, mock_audio_converter,
                                                   mock_event_publisher):
        """Test creation of audio chunks with size constraints."""
        # Create service with mocked dependencies
        service = ChunkingServiceImpl(mock_audio_format_service, mock_audio_converter)
        
        # Mock size limit for OPENAI_WHISPER
        mock_limit = MagicMock()
        mock_limit.check_file_size.return_value = (True, None)  # Files within limit
        mock_limit.max_file_size_bytes = 50000
        mock_get_limit.return_value = mock_limit
        
        # Define chunk boundaries
        boundaries = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
        
        # Create output directory
        output_dir = tmp_path / "chunks"
        
        # Use a quality specification
        quality_spec = AudioQualitySpecification.for_whisper_api()
        
        # Create chunks with API provider
        chunk_paths = service.create_audio_chunks(
            example_audio_path,
            output_dir,
            boundaries,
            quality_spec,
            ApiProvider.OPENAI_WHISPER
        )
        
        # Verify that the size limit was checked
        assert mock_get_limit.called
        
        # Verify that the chunk paths list has the correct length
        assert len(chunk_paths) == len(boundaries)
    
    @patch.object(ApiSizeLimitPolicy, 'get_limit_for_provider')
    def test_create_audio_chunks_exceeding_size(self, mock_get_limit,
                                              example_audio_path, tmp_path,
                                              mock_audio_format_service, mock_audio_converter,
                                              mock_event_publisher):
        """Test creation of audio chunks with size constraints that can't be met."""
        # Create service with mocked dependencies
        service = ChunkingServiceImpl(mock_audio_format_service, mock_audio_converter)
        
        # Mock size limit for OPENAI_WHISPER
        mock_limit = MagicMock()
        mock_limit.check_file_size.side_effect = [
            (False, "File too large"),  # First chunk exceeds limit
            (True, None),               # Second chunk within limit
            (True, None)                # Third chunk within limit
        ]
        mock_limit.max_file_size_bytes = 50000  # Add max size property
        mock_get_limit.return_value = mock_limit
        
        # Define chunk boundaries
        boundaries = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
        
        # Create output directory
        output_dir = tmp_path / "chunks"
        
        # Use a quality specification
        quality_spec = AudioQualitySpecification.for_whisper_api()
        
        # Create chunks with API provider
        chunk_paths = service.create_audio_chunks(
            example_audio_path,
            output_dir,
            boundaries,
            quality_spec,
            ApiProvider.OPENAI_WHISPER
        )
        
        # Verify that the size limit was checked
        assert mock_get_limit.called
        
        # Verify that a ChunkingEvent with oversized_chunks was published
        for call in mock_event_publisher.call_args_list:
            event = call[0][0]
            if isinstance(event, ChunkingEvent):
                assert event.has_oversized_chunks
                assert len(event.oversized_chunk_indices) > 0
                break
        else:
            pytest.fail("No ChunkingEvent with oversized chunks was published")
    
    def test_detect_silence(self, example_audio_path, mock_audio_format_service):
        """Test detecting silence in audio."""
        # Create service with mocked dependencies
        service = ChunkingServiceImpl(mock_audio_format_service, MagicMock())
        
        # Detect silence with default parameters
        silence_regions = service.detect_silence(example_audio_path)
        
        # Verify that silence was detected
        assert len(silence_regions) == 2
        assert silence_regions[0] == (1.0, 2.0)
        assert silence_regions[1] == (5.0, 6.0)
        
        # Verify that detect_silence was called with correct parameters
        mock_audio_format_service.detect_silence.assert_called_once_with(
            example_audio_path, 0.5, -40
        )
    
    def test_detect_silence_custom_params(self, example_audio_path, mock_audio_format_service):
        """Test detecting silence with custom parameters."""
        # Create service with mocked dependencies
        service = ChunkingServiceImpl(mock_audio_format_service, MagicMock())
        
        # Detect silence with custom parameters
        silence_regions = service.detect_silence(
            example_audio_path,
            min_silence_duration=1.0,
            silence_threshold=-60
        )
        
        # Verify that silence was detected
        assert len(silence_regions) == 2
        
        # Verify that detect_silence was called with custom parameters
        mock_audio_format_service.detect_silence.assert_called_once_with(
            example_audio_path, 1.0, -60
        ) 