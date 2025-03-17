"""
Tests for the SizeAwareFFmpegConverter class.

These tests verify the size-constrained audio conversion capabilities,
including adapting quality to meet size requirements.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY, mock_open

import pytest
import numpy as np

from pyhearingai.infrastructure.adapters.size_aware_audio_converter import SizeAwareFFmpegConverter
from pyhearingai.core.domain.audio_quality import (
    AudioQualitySpecification, 
    AudioCodec, 
    AudioFormat
)
from pyhearingai.core.domain.api_constraints import ApiProvider, ApiSizeLimitPolicy
from pyhearingai.core.domain.events import (
    AudioConversionEvent, 
    AudioSizeExceededEvent,
    EventPublisher
)


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
        import soundfile as sf
        sf.write(audio_path, samples, sr)
    except ImportError:
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
def mock_event_publisher():
    """Mock EventPublisher for testing event emission."""
    with patch.object(EventPublisher, 'publish', autospec=True) as mock_publish:
        yield mock_publish


class TestSizeAwareFFmpegConverter:
    """Test suite for SizeAwareFFmpegConverter."""
    
    def test_init(self):
        """Test initialization of the converter."""
        converter = SizeAwareFFmpegConverter()
        assert converter is not None
        
    @patch("ffmpeg.probe")
    @patch("ffmpeg.input")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("tempfile.mkstemp")
    @patch("shutil.copy2")
    @patch("shutil.move")
    def test_convert_with_quality_spec(
        self, mock_move, mock_copy2, mock_mkstemp, mock_getsize, mock_exists, 
        mock_input, mock_probe, example_audio_path, tmp_path, mock_event_publisher
    ):
        """Test conversion using a quality specification."""
        # Setup mock for temporary file creation
        mock_mkstemp.return_value = (1, str(tmp_path / "temp_output.wav"))
        
        # Setup mock for file existence check
        mock_exists.return_value = True
        
        # Setup mock for file size
        mock_getsize.return_value = 50000  # 50KB
        
        # Setup mock for file operations
        mock_copy2.return_value = None
        mock_move.return_value = None
        
        # Setup mock ffmpeg probe result
        mock_probe.return_value = {
            'format': {'duration': '1.0', 'size': '88200'},
            'streams': [
                {
                    'codec_type': 'audio',
                    'sample_rate': '44100',
                    'channels': '1'
                }
            ]
        }
        
        # Setup mock ffmpeg input and output
        mock_output = MagicMock()
        mock_input.return_value.output.return_value = mock_output
        mock_output.run.return_value = (None, None)
        
        # Create converter
        converter = SizeAwareFFmpegConverter()
        
        # Create quality spec
        quality_spec = AudioQualitySpecification.for_whisper_api()
        
        # Create output directory
        output_dir = tmp_path / "converted"
        output_dir.mkdir(exist_ok=True)
        
        # Convert with quality spec
        with patch("builtins.open", mock_open()):
            result_path, metadata = converter.convert_with_quality_spec(
                example_audio_path,
                quality_spec,
                output_dir=output_dir
            )
        
        # Verify the result
        assert "original_size" in metadata
        assert "converted_size" in metadata
        assert "compression_ratio" in metadata
        assert "original_format" in metadata
        assert "adjustments_made" in metadata
        
        # Verify that mock_input was called with the correct path
        mock_input.assert_called_once_with(str(example_audio_path))
        
        # Verify that output was called with expected arguments
        mock_input.return_value.output.assert_called_once()
        
        # Verify that an event was published
        mock_event_publisher.assert_called_once()
        event = mock_event_publisher.call_args[0][0]
        assert isinstance(event, AudioConversionEvent)
        assert event.is_successful is True
        assert event.source_path == example_audio_path
    
    @patch("ffmpeg.probe")
    @patch("ffmpeg.input")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("tempfile.mkstemp")
    @patch("shutil.copy2")
    @patch("shutil.move")
    def test_convert_with_size_constraint_within_limit(
        self, mock_move, mock_copy2, mock_mkstemp, mock_getsize, mock_exists, 
        mock_input, mock_probe, example_audio_path, tmp_path, mock_event_publisher
    ):
        """Test conversion with size constraint when the file is already within limits."""
        # Setup mock for temporary file creation
        mock_mkstemp.return_value = (1, str(tmp_path / "temp_output.wav"))
        
        # Setup mock for file existence check
        mock_exists.return_value = True
        
        # Setup mock file size (under the limit)
        mock_getsize.return_value = 10000
        
        # Setup mock for file operations
        mock_copy2.return_value = None
        mock_move.return_value = None
        
        # Setup mock ffmpeg probe result
        mock_probe.return_value = {
            'format': {'duration': '1.0', 'size': '10000'},
            'streams': [
                {
                    'codec_type': 'audio',
                    'sample_rate': '44100',
                    'channels': '1'
                }
            ]
        }
        
        # Setup mock ffmpeg input and output
        mock_output = MagicMock()
        mock_input.return_value.output.return_value = mock_output
        mock_output.run.return_value = (None, None)
        
        # Create converter
        converter = SizeAwareFFmpegConverter()
        
        # Create output directory
        output_dir = tmp_path / "converted"
        output_dir.mkdir(exist_ok=True)
        
        # Convert with size constraint
        max_size = 20000  # Bytes
        with patch("builtins.open", mock_open()):
            result_path, metadata = converter.convert_with_size_constraint(
                example_audio_path,
                max_size,
                output_dir=output_dir
            )
        
        # Verify the result
        assert "original_size" in metadata
        assert "converted_size" in metadata
        assert metadata["converted_size"] <= max_size
        
        # Verify that an event was published
        mock_event_publisher.assert_called_once()
        event = mock_event_publisher.call_args[0][0]
        assert isinstance(event, AudioConversionEvent)
        assert event.is_successful is True
    
    @patch("ffmpeg.probe")
    @patch("ffmpeg.input")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("tempfile.mkstemp")
    @patch("shutil.copy2")
    @patch("shutil.move")
    def test_convert_with_size_constraint_needs_reduction(
        self, mock_move, mock_copy2, mock_mkstemp, mock_getsize, mock_exists, 
        mock_input, mock_probe, example_audio_path, tmp_path, mock_event_publisher
    ):
        """Test conversion with size constraint when quality reduction is needed."""
        # Setup mock for temporary file creation
        mock_mkstemp.return_value = (1, str(tmp_path / "temp_output.wav"))
        
        # Setup mock for file existence check
        mock_exists.return_value = True
        
        # Setup mock file sizes (first above limit, second below limit)
        mock_getsize.side_effect = [100000, 20000]
        
        # Setup mock for file operations
        mock_copy2.return_value = None
        mock_move.return_value = None
        
        # Setup mock ffmpeg probe result
        mock_probe.return_value = {
            'format': {'duration': '1.0', 'size': '100000'},
            'streams': [
                {
                    'codec_type': 'audio',
                    'sample_rate': '44100',
                    'channels': '1'
                }
            ]
        }
        
        # Setup mock ffmpeg input and output
        mock_output = MagicMock()
        mock_input.return_value.output.return_value = mock_output
        mock_output.run.return_value = (None, None)
        
        # Create converter
        converter = SizeAwareFFmpegConverter()
        
        # Create output directory
        output_dir = tmp_path / "converted"
        output_dir.mkdir(exist_ok=True)
        
        # Convert with size constraint
        max_size = 50000  # Bytes
        with patch("builtins.open", mock_open()):
            result_path, metadata = converter.convert_with_size_constraint(
                example_audio_path,
                max_size,
                output_dir=output_dir
            )
        
        # Verify the result
        assert "original_size" in metadata
        assert "converted_size" in metadata
        assert metadata["converted_size"] <= max_size
        
        # Verify that multiple output calls were made (quality reduction attempts)
        assert mock_input.return_value.output.call_count >= 1
        
        # Verify that events were published
        assert mock_event_publisher.call_count >= 1
        
        # Verify the final event is a successful conversion
        last_event = mock_event_publisher.call_args[0][0]
        assert isinstance(last_event, AudioConversionEvent)
        assert last_event.is_successful is True
    
    @patch("ffmpeg.probe")
    @patch("ffmpeg.input")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("tempfile.mkstemp")
    @patch("shutil.copy2")
    @patch("shutil.move")
    def test_convert_with_size_constraint_cannot_meet_limit(
        self, mock_move, mock_copy2, mock_mkstemp, mock_getsize, mock_exists, 
        mock_input, mock_probe, example_audio_path, tmp_path, mock_event_publisher
    ):
        """Test conversion with size constraint when the limit cannot be met."""
        # Setup mock for temporary file creation
        mock_mkstemp.return_value = (1, str(tmp_path / "temp_output.wav"))
        
        # Setup mock for file existence check
        mock_exists.return_value = True
        
        # Setup mock file sizes (always above limit)
        mock_getsize.return_value = 100000
        
        # Setup mock for file operations
        mock_copy2.return_value = None
        mock_move.return_value = None
        
        # Setup mock ffmpeg probe result
        mock_probe.return_value = {
            'format': {'duration': '1.0', 'size': '100000'},
            'streams': [
                {
                    'codec_type': 'audio',
                    'sample_rate': '44100',
                    'channels': '1'
                }
            ]
        }
        
        # Setup mock ffmpeg input and output
        mock_output = MagicMock()
        mock_input.return_value.output.return_value = mock_output
        mock_output.run.return_value = (None, None)
        
        # Create converter
        converter = SizeAwareFFmpegConverter()
        
        # Create output directory
        output_dir = tmp_path / "converted"
        output_dir.mkdir(exist_ok=True)
        
        # Convert with size constraint
        max_size = 1000  # Very small limit in bytes
        
        # Should raise ValueError because limit cannot be met
        with pytest.raises(ValueError) as exc_info:
            with patch("builtins.open", mock_open()):
                converter.convert_with_size_constraint(
                    example_audio_path,
                    max_size,
                    output_dir=output_dir
                )
        
        assert "Unable to convert" in str(exc_info.value)
        assert "to meet size constraint" in str(exc_info.value)
        
        # Verify that an audio size exceeded event was published
        has_size_exceeded_event = False
        for call in mock_event_publisher.call_args_list:
            event = call[0][0]
            if isinstance(event, AudioSizeExceededEvent):
                has_size_exceeded_event = True
                assert event.source_path == example_audio_path
                assert event.target_size_bytes == max_size
                assert event.best_achieved_size == 100000
                break
        
        assert has_size_exceeded_event, "No AudioSizeExceededEvent was published"
    
    @patch("ffmpeg.probe")
    @patch("os.path.exists")
    def test_estimate_output_size(self, mock_exists, mock_probe, example_audio_path):
        """Test estimation of output size based on quality specification."""
        # Setup mock for file existence check
        mock_exists.return_value = True
        
        # Setup mock ffmpeg probe result with complete structure
        mock_probe.return_value = {
            'format': {'duration': '10.0'},  # 10 seconds
            'streams': [
                {
                    'codec_type': 'audio',
                    'sample_rate': '44100',
                    'channels': '1',
                    'duration': '10.0'  # Ensure duration is in the stream info too
                }
            ]
        }
        
        # Create converter
        converter = SizeAwareFFmpegConverter()
        
        # Test with different quality specs
        high_quality = AudioQualitySpecification.high_quality()
        medium_quality = AudioQualitySpecification.for_local_processing()
        low_quality = AudioQualitySpecification.for_whisper_api()
        
        # Calculate expected sizes based on the quality specs
        expected_high_size = high_quality.estimated_bytes_per_second() * 10
        expected_medium_size = medium_quality.estimated_bytes_per_second() * 10
        expected_low_size = low_quality.estimated_bytes_per_second() * 10
        
        # Mock Path.exists to return True
        with patch.object(Path, 'exists', return_value=True):
            # Get estimated sizes
            high_size = converter.estimate_output_size(Path(example_audio_path), high_quality)
            medium_size = converter.estimate_output_size(Path(example_audio_path), medium_quality)
            low_size = converter.estimate_output_size(Path(example_audio_path), low_quality)
        
        # Verify that the function returns a value for each quality spec
        assert isinstance(high_size, int)
        assert isinstance(medium_size, int)
        assert isinstance(low_size, int)
        
        # Verify that the estimated sizes match our expectations
        assert high_size == expected_high_size
        assert medium_size == expected_medium_size
        assert low_size == expected_low_size
    
    @patch("ffmpeg.probe")
    @patch("os.path.exists")
    @patch("pyhearingai.infrastructure.audio_converter.FFmpegAudioConverter.convert")
    def test_ffmpeg_error_handling(self, mock_convert, mock_exists, mock_probe, example_audio_path, tmp_path):
        """Test handling of FFmpeg errors during conversion."""
        # Setup mock for file existence check
        mock_exists.return_value = True
        
        # Setup mock ffmpeg probe result
        mock_probe.return_value = {
            'format': {'duration': '1.0', 'size': '88200'},
            'streams': [
                {
                    'codec_type': 'audio',
                    'sample_rate': '44100',
                    'channels': '1'
                }
            ]
        }
        
        # Setup mock for parent class convert method to raise RuntimeError
        mock_convert.side_effect = RuntimeError("Simulated FFmpeg error")
        
        # Create converter
        converter = SizeAwareFFmpegConverter()
        
        # Create quality spec
        quality_spec = AudioQualitySpecification.for_whisper_api()
        
        # Attempt conversion with quality spec
        with pytest.raises(RuntimeError) as exc_info:
            converter.convert_with_quality_spec(
                Path(example_audio_path),
                quality_spec,
                output_dir=tmp_path
            )
        
        # Verify the error message
        assert "Simulated FFmpeg error" in str(exc_info.value)
    
    @patch("os.path.exists")
    def test_file_not_found_error(self, mock_exists, tmp_path):
        """Test handling of file not found errors."""
        # Setup mock for file existence check
        mock_exists.return_value = False
        
        # Create converter
        converter = SizeAwareFFmpegConverter()
        
        # Create quality spec
        quality_spec = AudioQualitySpecification.for_whisper_api()
        
        # Attempt conversion with non-existent file
        non_existent_path = tmp_path / "non_existent.wav"
        
        with pytest.raises(FileNotFoundError):
            converter.convert_with_quality_spec(
                non_existent_path,
                quality_spec,
                output_dir=tmp_path
            ) 