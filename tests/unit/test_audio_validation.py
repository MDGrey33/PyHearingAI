"""
Tests for the AudioValidationService class.

These tests verify the functionality for validating audio files against
format, quality, and size constraints.
"""

import os
import tempfile
import math
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pytest
import soundfile as sf

from pyhearingai.core.domain.audio_validation import AudioValidationService
from pyhearingai.core.domain.audio_quality import AudioQualitySpecification, AudioFormat, AudioCodec
from pyhearingai.core.domain.api_constraints import ApiProvider, ApiSizeLimitPolicy, ApiSizeLimit
from pyhearingai.core.domain.events import AudioValidationEvent, EventPublisher
from pyhearingai.core.ports import AudioFormatService
from pyhearingai.infrastructure.adapters.audio_format_service import FFmpegAudioFormatService


@pytest.fixture
def example_audio_path(tmp_path):
    """Create a simple test audio file."""
    audio_path = tmp_path / "example.wav"
    
    # Create a synthetic audio file (1 second, 44100 Hz, mono)
    duration = 1.0
    sr = 44100
    samples = np.random.uniform(-0.1, 0.1, int(duration * sr))
    
    # Use numpy to write a simple WAV file
    try:
        sf.write(audio_path, samples, sr)
    except Exception:
        # Fallback if soundfile fails
        import wave
        import struct
        
        # Create a WAV file in write mode
        wf = wave.open(str(audio_path), 'wb')
        try:
            # Using wave module's methods for writing WAV files
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sr)  # Sample rate
            wf.writeframes(struct.pack('h' * len(samples), *[int(s * 32767) for s in samples]))
        finally:
            wf.close()
    
    return audio_path


@pytest.fixture
def mp3_audio_path(tmp_path):
    """Create a mock MP3 audio file."""
    audio_path = tmp_path / "example.mp3"
    
    # Just create an empty file with MP3 extension
    # The actual content doesn't matter for our tests since we'll mock the validation
    with open(audio_path, 'w') as f:
        f.write("Mock MP3 content")
    
    return audio_path


@pytest.fixture
def invalid_audio_path(tmp_path):
    """Create an invalid audio file (not actually audio)."""
    invalid_path = tmp_path / "invalid.wav"
    
    # Just create a text file with .wav extension
    with open(invalid_path, 'w') as f:
        f.write("This is not a valid audio file")
    
    return invalid_path


@pytest.fixture
def mock_audio_format_service():
    """Create a mock audio format service."""
    mock_service = MagicMock(spec=AudioFormatService)
    
    # Mock get_audio_metadata to return different metadata for different files
    def mock_get_metadata(audio_path):
        if audio_path.suffix.lower() == '.wav':
            return {
                'duration': 10.0,
                'sample_rate': 44100,
                'channels': 2,
                'codec': 'pcm_s16le',
                'bit_rate': 1411000,
                'format': 'wav',
                'file_size': 1000000
            }
        elif audio_path.suffix.lower() == '.mp3':
            return {
                'duration': 10.0,
                'sample_rate': 44100,
                'channels': 2,
                'codec': 'mp3',
                'bit_rate': 320000,
                'format': 'mp3',
                'file_size': 400000
            }
        else:
            # For invalid files, return limited metadata with an error
            return {
                'file_size': os.path.getsize(audio_path),
                'file_extension': audio_path.suffix.lower().lstrip('.'),
                'error': 'Invalid audio file'
            }
    
    mock_service.get_audio_metadata.side_effect = mock_get_metadata
    
    return mock_service


@pytest.fixture
def mock_event_publisher():
    """Mock EventPublisher for testing event emission."""
    with patch.object(EventPublisher, 'publish', autospec=True) as mock_publish:
        yield mock_publish


class TestAudioValidationService:
    """Test suite for AudioValidationService."""
    
    def test_init(self):
        """Test class existence and expected static methods."""
        # Verify that the class exists and can be accessed
        assert AudioValidationService is not None
        
        # Test that the class has the expected static methods
        assert hasattr(AudioValidationService, 'validate_audio_file')
        assert hasattr(AudioValidationService, 'estimate_optimal_chunk_duration')
        assert hasattr(AudioValidationService, 'validate_chunk_parameters')
        assert hasattr(AudioValidationService, 'suggest_quality_reduction')
    
    @patch.object(ApiSizeLimitPolicy, 'validate_file_for_provider')
    def test_validate_audio_file(self, mock_validate):
        """Test validation of audio files against API provider constraints."""
        # Setup mock return values
        mock_validate.return_value = (True, None)  # Valid file
        
        # Test with valid file
        file_path = Path("/path/to/test.wav")
        is_valid, error_msg = AudioValidationService.validate_audio_file(file_path, ApiProvider.OPENAI_WHISPER)
        
        # Verify results
        assert is_valid is True
        assert error_msg is None
        mock_validate.assert_called_once_with(file_path, ApiProvider.OPENAI_WHISPER)
        
        # Test with invalid file
        mock_validate.reset_mock()
        mock_validate.return_value = (False, "File too large")
        
        is_valid, error_msg = AudioValidationService.validate_audio_file(file_path, ApiProvider.OPENAI_WHISPER)
        
        # Verify results
        assert is_valid is False
        assert error_msg == "File too large"
        mock_validate.assert_called_once_with(file_path, ApiProvider.OPENAI_WHISPER)
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch.object(ApiSizeLimitPolicy, 'get_limit_for_provider')
    def test_estimate_optimal_chunk_duration(self, mock_get_limit, mock_getsize, mock_exists, example_audio_path):
        """Test estimation of optimal chunk duration."""
        # Setup mocks
        mock_exists.return_value = True
        mock_getsize.return_value = 1000000  # 1MB
        
        # Create mock API size limit
        mock_size_limit = ApiSizeLimit(
            provider=ApiProvider.OPENAI_WHISPER,
            max_file_size_bytes=250000,  # 250KB
            max_duration_seconds=600
        )
        mock_get_limit.return_value = mock_size_limit
        
        # Create quality spec
        quality_spec = AudioQualitySpecification.for_whisper_api()
        
        # Test estimation with standard parameters
        duration = AudioValidationService.estimate_optimal_chunk_duration(
            example_audio_path,
            quality_spec,
            ApiProvider.OPENAI_WHISPER
        )
        
        # Verify result is a reasonable duration
        assert isinstance(duration, float)
        assert duration > 0
        
        # The actual value depends on the estimated_bytes_per_second implementation
        # but we can verify the calculation logic is applied
        bytes_per_second = quality_spec.estimated_bytes_per_second()
        if bytes_per_second > 0:
            expected_duration = int(237500 / bytes_per_second)  # 250000 * 0.95 (5% safety margin)
            expected_duration = math.floor(expected_duration)
            expected_duration = min(600, max(10, expected_duration))
            assert duration == expected_duration
        else:
            assert duration == 30.0  # Default for invalid bytes_per_second
        
        # Test with non-existent file
        mock_exists.return_value = False
        
        # For Path objects, we need to patch the Path.exists method
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                AudioValidationService.estimate_optimal_chunk_duration(
                    example_audio_path,
                    quality_spec,
                    ApiProvider.OPENAI_WHISPER
                )
    
    def test_validate_chunk_parameters(self):
        """Test validation of chunk parameters."""
        # Test valid parameters
        is_valid, error_msg = AudioValidationService.validate_chunk_parameters(30.0, 5.0)
        assert is_valid is True
        assert error_msg is None
        
        # Test invalid chunk duration
        is_valid, error_msg = AudioValidationService.validate_chunk_parameters(0, 0)
        assert is_valid is False
        assert "Chunk duration must be positive" in error_msg
        
        # Test negative overlap
        is_valid, error_msg = AudioValidationService.validate_chunk_parameters(30.0, -5.0)
        assert is_valid is False
        assert "Overlap duration cannot be negative" in error_msg
        
        # Test overlap >= chunk duration
        is_valid, error_msg = AudioValidationService.validate_chunk_parameters(30.0, 30.0)
        assert is_valid is False
        assert "Overlap duration" in error_msg
        assert "must be less than" in error_msg
        
        # Test very short chunk duration
        is_valid, error_msg = AudioValidationService.validate_chunk_parameters(0.5, 0)
        assert is_valid is False
        assert "too short" in error_msg
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch.object(ApiSizeLimitPolicy, 'get_limit_for_provider')
    def test_suggest_quality_reduction(self, mock_get_limit, mock_getsize, mock_exists, example_audio_path):
        """Test suggestion of quality reduction for large files."""
        # Setup mocks
        mock_exists.return_value = True
        
        # Create mock API size limit
        mock_size_limit = ApiSizeLimit(
            provider=ApiProvider.OPENAI_WHISPER,
            max_file_size_bytes=250000,  # 250KB
            max_duration_seconds=600
        )
        mock_get_limit.return_value = mock_size_limit
        
        # Test when file is already within limits
        mock_getsize.return_value = 200000  # 200KB
        
        current_spec = AudioQualitySpecification(
            sample_rate=48000,
            channels=2,
            bit_depth=24,
            format=AudioFormat.WAV,
            codec=AudioCodec.PCM_S24LE
        )
        
        result = AudioValidationService.suggest_quality_reduction(
            example_audio_path,
            current_spec,
            ApiProvider.OPENAI_WHISPER
        )
        
        # No reduction needed
        assert result is None
        
        # Test when file exceeds limits - should reduce sample rate first
        mock_getsize.return_value = 300000  # 300KB
        
        result = AudioValidationService.suggest_quality_reduction(
            example_audio_path,
            current_spec,
            ApiProvider.OPENAI_WHISPER
        )
        
        # Should reduce sample rate to 16kHz
        assert result is not None
        assert result.sample_rate == 16000
        assert result.channels == 2  # Unchanged
        assert result.bit_depth == 24  # Unchanged
        
        # Test with already reduced sample rate - should reduce to mono next
        current_spec = AudioQualitySpecification(
            sample_rate=16000,
            channels=2,
            bit_depth=24,
            format=AudioFormat.WAV,
            codec=AudioCodec.PCM_S24LE
        )
        
        result = AudioValidationService.suggest_quality_reduction(
            example_audio_path,
            current_spec,
            ApiProvider.OPENAI_WHISPER
        )
        
        # Should reduce to mono
        assert result is not None
        assert result.sample_rate == 16000  # Unchanged
        assert result.channels == 1  # Reduced to mono
        assert result.bit_depth == 24  # Unchanged
        
        # Test with mono and 16kHz - should reduce bit depth next
        current_spec = AudioQualitySpecification(
            sample_rate=16000,
            channels=1,
            bit_depth=24,
            format=AudioFormat.WAV,
            codec=AudioCodec.PCM_S24LE
        )
        
        result = AudioValidationService.suggest_quality_reduction(
            example_audio_path,
            current_spec,
            ApiProvider.OPENAI_WHISPER
        )
        
        # Should reduce bit depth
        assert result is not None
        assert result.sample_rate == 16000  # Unchanged
        assert result.channels == 1  # Unchanged
        assert result.bit_depth == 16  # Reduced bit depth
        
        # Test with already minimal settings - should return None as no further reduction is possible
        current_spec = AudioQualitySpecification(
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            format=AudioFormat.WAV,
            codec=AudioCodec.PCM_S16LE
        )
        
        result = AudioValidationService.suggest_quality_reduction(
            example_audio_path,
            current_spec,
            ApiProvider.OPENAI_WHISPER
        )
        
        # No further reduction possible
        assert result is None
        
        # Test with non-existent file
        mock_exists.return_value = False
        
        result = AudioValidationService.suggest_quality_reduction(
            example_audio_path,
            current_spec,
            ApiProvider.OPENAI_WHISPER
        )
        
        # Should return None for non-existent file
        assert result is None 