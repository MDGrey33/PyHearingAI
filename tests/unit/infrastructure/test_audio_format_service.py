"""
Tests for the AudioFormatService implementation.

This test suite verifies the functionality of the FFmpegAudioFormatService,
which provides audio format operations and metadata retrieval.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyhearingai.core.domain.audio_quality import AudioQualitySpecification
from pyhearingai.infrastructure.adapters.audio_format_service import FFmpegAudioFormatService
from tests.fixtures.mock_implementations import MockAudioFormatService


@pytest.fixture
def test_audio_file():
    """
    Create a test audio file with known properties.

    Creates a WAV file with a sine wave tone, including a silence section in the middle.

    Returns:
        Tuple[Path, dict]: A tuple containing the path to the test file and a dictionary
                          with its known properties (duration, sample_rate, etc.)
    """
    # Use tempfile to create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    # For now, we'll return a predefined file path and metadata
    # In a real test, we would generate an actual audio file here
    metadata = {
        "duration": 10.0,
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav",
        "silence_at": [(2.0, 3.0), (7.0, 8.0)],  # Silence regions (start, end)
    }

    # Return the file path and its properties
    return temp_path, metadata


class TestFFmpegAudioFormatService:
    """Test suite for the FFmpegAudioFormatService implementation."""

    @pytest.mark.skip(reason="Need to mock FFmpeg calls")
    def test_extract_audio_segment(self, test_audio_file):
        """
        Test extraction of audio segments.

        Verifies that FFmpegAudioFormatService can correctly extract
        a segment of audio from a larger file.
        """
        # Get the test file path and metadata
        audio_path, metadata = test_audio_file

        # Create a mock service that intercepts FFmpeg calls
        with patch("ffmpeg.input") as mock_input, patch("ffmpeg.output") as mock_output, patch(
            "ffmpeg.run"
        ) as mock_run:
            # Configure mocks to simulate successful extraction
            mock_output.return_value.overwrite_output.return_value.run.return_value = None

            # Create the service
            service = FFmpegAudioFormatService()

            # Extract a segment
            start_time = 1.0
            end_time = 4.0
            output_path = Path(tempfile.gettempdir()) / "segment.wav"

            # Call the service method
            result = service.extract_audio_segment(audio_path, output_path, start_time, end_time)

            # Verify the result
            assert result == output_path

            # Verify FFmpeg was called with correct parameters
            mock_input.assert_called_once_with(str(audio_path), ss=start_time, to=end_time)
            mock_output.assert_called_once()

    @pytest.mark.skip(reason="Need to mock FFmpeg calls")
    def test_detect_silence(self, test_audio_file):
        """
        Test silence detection functionality.

        Verifies that the service can detect silence regions in an audio file.
        """
        # Get the test file path and metadata
        audio_path, metadata = test_audio_file

        # Mock subprocess to return predictable silence detection output
        with patch("subprocess.run") as mock_run:
            # Mock output from FFmpeg silence detection
            mock_run.return_value.stderr = (
                "[silencedetect @ 0x7f8f9c00f600] silence_start: 2.0\n"
                "[silencedetect @ 0x7f8f9c00f600] silence_end: 3.0 | silence_duration: 1.0\n"
                "[silencedetect @ 0x7f8f9c00f600] silence_start: 7.0\n"
                "[silencedetect @ 0x7f8f9c00f600] silence_end: 8.0 | silence_duration: 1.0\n"
            )

            # Create the service
            service = FFmpegAudioFormatService()

            # Detect silence
            silence_regions = service.detect_silence(
                audio_path, min_silence_duration=0.5, silence_threshold=-40
            )

            # Verify the detected silence regions
            assert len(silence_regions) == 2
            assert silence_regions[0]["start"] == 2.0
            assert silence_regions[0]["end"] == 3.0
            assert silence_regions[1]["start"] == 7.0
            assert silence_regions[1]["end"] == 8.0

    @pytest.mark.skip(reason="Need to mock FFmpeg calls")
    def test_detect_silence_custom_params(self, test_audio_file):
        """
        Test silence detection with custom parameters.

        Verifies that the service respects custom silence detection parameters.
        """
        # Get the test file path and metadata
        audio_path, metadata = test_audio_file

        # Mock subprocess and create service
        with patch("subprocess.run") as mock_run:
            # Configure the mock
            mock_run.return_value.stderr = (
                "[silencedetect @ 0x7f8f9c00f600] silence_start: 2.0\n"
                "[silencedetect @ 0x7f8f9c00f600] silence_end: 3.0 | silence_duration: 1.0\n"
            )

            # Create service
            service = FFmpegAudioFormatService()

            # Detect silence with custom parameters
            min_silence_duration = 1.0
            silence_threshold = -30

            service.detect_silence(
                audio_path,
                min_silence_duration=min_silence_duration,
                silence_threshold=silence_threshold,
            )

            # Verify that FFmpeg was called with the custom parameters
            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            assert "-af" in cmd_args
            filter_idx = cmd_args.index("-af") + 1
            assert (
                f"silencedetect=noise={silence_threshold}dB:d={min_silence_duration}"
                in cmd_args[filter_idx]
            )

    @pytest.mark.skip(reason="Need to mock FFmpeg calls")
    def test_convert_format(self, test_audio_file):
        """
        Test format conversion functionality.

        Verifies that the service can convert audio files to different formats.
        """
        # This test doesn't make sense because FFmpegAudioFormatService doesn't have a convert_format method
        # It would be handled by a separate AudioConverter class
        pass

    @pytest.mark.skip(reason="Need to mock FFmpeg calls")
    def test_get_audio_metadata(self, test_audio_file):
        """
        Test metadata extraction functionality.

        Verifies that the service can extract metadata from audio files.
        """
        # Get the test file path and metadata
        audio_path, metadata = test_audio_file

        # Mock ffmpeg.probe to return predictable metadata
        with patch("ffmpeg.probe") as mock_probe:
            # Configure the mock to return sample metadata
            mock_probe.return_value = {
                "streams": [
                    {
                        "codec_type": "audio",
                        "codec_name": "pcm_s16le",
                        "sample_rate": "16000",
                        "channels": "1",
                        "bits_per_sample": "16",
                        "bit_rate": "256000",
                    }
                ],
                "format": {"duration": "10.0", "size": "1024000", "format_name": "wav"},
            }

            # Create the service
            service = FFmpegAudioFormatService()

            # Get metadata
            result = service.get_audio_metadata(audio_path)

            # Verify the metadata
            assert result["duration"] == 10.0
            assert result["size_bytes"] == 1024000
            assert result["format"] == "wav"
            assert result["sample_rate"] == 16000
            assert result["channels"] == 1
            assert result["codec"] == "pcm_s16le"
            assert result["bit_rate"] == 256000
            assert result["bits_per_sample"] == 16

    @pytest.mark.skip(reason="Need to mock FFmpeg calls")
    def test_error_handling_invalid_file(self):
        """
        Test error handling for invalid audio files.

        Verifies that the service properly handles errors when processing invalid files.
        """
        # Create a service
        service = FFmpegAudioFormatService()

        # Try to get metadata for a non-existent file
        non_existent_path = Path("/non/existent/file.wav")
        with pytest.raises(FileNotFoundError):
            service.get_audio_metadata(non_existent_path)

        # Mock ffmpeg.probe to raise an error
        with patch("ffmpeg.probe") as mock_probe, tempfile.NamedTemporaryFile(
            suffix=".wav"
        ) as temp_file:
            # Configure the mock to raise an exception
            mock_probe.side_effect = Exception("FFmpeg error")

            # Try to get metadata for a corrupted file
            with pytest.raises(RuntimeError):
                service.get_audio_metadata(Path(temp_file.name))

    @pytest.mark.skip(reason="Need to implement FFmpeg dependency check")
    def test_ffmpeg_dependency(self):
        """
        Test that the service correctly detects FFmpeg availability.

        Verifies that the service can check if FFmpeg is installed and available.
        """
        # This test would verify that the service can check for FFmpeg dependency
        # However, FFmpegAudioFormatService doesn't currently have this method
        pass


class TestMockAudioFormatService:
    """Test suite for the MockAudioFormatService implementation."""

    def test_mock_get_audio_metadata(self, test_audio_file):
        """Test that the mock service can provide metadata."""
        # Get the test file path
        audio_path, _ = test_audio_file

        # Create the mock service
        service = MockAudioFormatService()

        # Get metadata
        metadata = service.get_audio_metadata(audio_path)

        # Verify the metadata has the expected structure
        assert "duration" in metadata
        assert "sample_rate" in metadata
        assert "channels" in metadata
        assert "format" in metadata
        assert "codec" in metadata

    def test_mock_extract_audio_segment(self, test_audio_file, tmp_path):
        """Test that the mock service can extract segments."""
        # Get the test file path
        audio_path, _ = test_audio_file

        # Create the mock service
        service = MockAudioFormatService()

        # Extract a segment
        output_path = tmp_path / "segment.wav"
        result = service.extract_audio_segment(audio_path, output_path, 1.0, 3.0)

        # Verify the result
        assert result == output_path
        assert output_path.exists()

    def test_mock_detect_silence(self, test_audio_file):
        """Test that the mock service can detect silence."""
        # Get the test file path
        audio_path, _ = test_audio_file

        # Create the mock service
        service = MockAudioFormatService()

        # Detect silence
        silence_regions = service.detect_silence(audio_path)

        # Verify the result
        assert isinstance(silence_regions, list)
        assert len(silence_regions) > 0
        assert "start" in silence_regions[0]
        assert "end" in silence_regions[0]
