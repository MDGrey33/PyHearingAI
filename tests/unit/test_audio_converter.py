"""
Tests for the FFmpegAudioConverter class.

These tests verify the functionality of the audio conversion capabilities.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import ffmpeg
import pytest

from pyhearingai.infrastructure.audio_converter import FFmpegAudioConverter


def test_audio_converter_basic(example_audio_path, temp_dir):
    """Test basic functionality of the audio converter."""
    # Create output directory
    output_dir = temp_dir / "audio_conversion"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create converter
    converter = FFmpegAudioConverter()

    # Convert audio
    converted_path = converter.convert(example_audio_path, output_dir=output_dir)

    # Verify the conversion
    assert converted_path.exists(), f"Converted file {converted_path} does not exist"
    assert (
        converted_path.suffix.lower() == ".wav"
    ), f"Expected .wav format, got {converted_path.suffix}"
    assert converted_path.stat().st_size > 0, f"Converted file {converted_path} is empty"


def test_audio_converter_same_format(example_audio_path, tmp_path):
    """Test handling when the input is already in the target format."""
    # Prepare a WAV file (assuming example_audio_path is already WAV)
    wav_file = tmp_path / "already.wav"
    with open(wav_file, "wb") as f:
        f.write(b"dummy wav data")

    converter = FFmpegAudioConverter()

    # Mock exists() to return True
    with patch.object(Path, "exists", return_value=True):
        # Convert the file that's already in the target format
        result_path = converter.convert(wav_file, target_format="wav")

        # It should return the original file
        assert result_path == wav_file


def test_audio_converter_different_formats(example_audio_path, tmp_path):
    """Test conversion to different audio formats."""
    converter = FFmpegAudioConverter()

    # Mock ffmpeg to avoid actual conversion
    with patch("ffmpeg.input") as mock_input:
        # Create mock for chain of calls
        mock_stream = MagicMock()
        mock_output = MagicMock()
        mock_input.return_value = mock_stream
        mock_stream.output.return_value = mock_output

        # Test conversion to mp3
        converter.convert(example_audio_path, "mp3", output_dir=tmp_path)

        # Check if ffmpeg was called with correct parameters
        mock_input.assert_called_with(str(example_audio_path))
        mock_stream.output.assert_called()
        mock_output.run.assert_called_with(quiet=True, overwrite_output=True)


def test_audio_converter_custom_options(example_audio_path, tmp_path):
    """Test conversion with custom options."""
    converter = FFmpegAudioConverter()

    # Mock ffmpeg to avoid actual conversion
    with patch("ffmpeg.input") as mock_input:
        # Create mock for chain of calls
        mock_stream = MagicMock()
        mock_output = MagicMock()
        mock_input.return_value = mock_stream
        mock_stream.output.return_value = mock_output

        # Test conversion with custom options
        converter.convert(
            example_audio_path,
            "wav",
            output_dir=tmp_path,
            sample_rate=44100,
            channels=2,
            codec="pcm_s24le",
        )

        # Check if output was called with custom params
        called_args = mock_stream.output.call_args[1]
        assert called_args["ar"] == 44100
        assert called_args["ac"] == 2
        assert called_args["codec:a"] == "pcm_s24le"


def test_audio_converter_default_codec_for_wav(example_audio_path, tmp_path):
    """Test that the default codec is set for WAV format when not specified."""
    converter = FFmpegAudioConverter()

    # Mock ffmpeg to avoid actual conversion
    with patch("ffmpeg.input") as mock_input:
        # Create mock for chain of calls
        mock_stream = MagicMock()
        mock_output = MagicMock()
        mock_input.return_value = mock_stream
        mock_stream.output.return_value = mock_output

        # Test conversion to wav without specifying codec
        converter.convert(example_audio_path, "wav", output_dir=tmp_path)

        # Check if default codec was set
        called_args = mock_stream.output.call_args[1]
        assert called_args["codec:a"] == "pcm_s16le"


def test_audio_converter_file_not_found():
    """Test handling of non-existent input files."""
    converter = FFmpegAudioConverter()

    # Try to convert a non-existent file
    with pytest.raises(FileNotFoundError):
        converter.convert(Path("/non/existent/audio.mp3"))


def test_audio_converter_ffmpeg_error(example_audio_path, tmp_path):
    """Test handling of FFmpeg errors."""
    converter = FFmpegAudioConverter()

    # Mock ffmpeg.input to raise an error
    with patch("ffmpeg.input") as mock_input:
        mock_stream = MagicMock()
        mock_input.return_value = mock_stream

        # Make the run method raise an error
        error = ffmpeg.Error(
            cmd=["ffmpeg", "-i", "input.mp3", "output.wav"],
            stdout=b"",
            stderr=b"Mock error from FFmpeg",
        )
        mock_stream.output.return_value.run.side_effect = error

        # Test that the error is properly handled
        with pytest.raises(RuntimeError) as excinfo:
            converter.convert(example_audio_path, output_dir=tmp_path)

        # Check that the error message contains the FFmpeg error
        assert "FFmpeg error" in str(excinfo.value)
        assert "Mock error from FFmpeg" in str(excinfo.value)


def test_audio_converter_general_exception(example_audio_path, tmp_path):
    """Test handling of general exceptions."""
    converter = FFmpegAudioConverter()

    # Mock ffmpeg.input to raise a general exception
    with patch("ffmpeg.input") as mock_input:
        mock_input.side_effect = Exception("General error")

        # Test that the error is properly handled
        with pytest.raises(Exception) as excinfo:
            converter.convert(example_audio_path, output_dir=tmp_path)

        # Check that the exception is passed through
        assert "General error" in str(excinfo.value)


def test_audio_converter_temp_dir_creation(example_audio_path):
    """Test that a temporary directory is created when output_dir is not specified."""
    converter = FFmpegAudioConverter()

    # Mock tempfile.mkdtemp and ffmpeg to avoid actual conversion
    with patch("tempfile.mkdtemp") as mock_mkdtemp, patch("ffmpeg.input") as mock_input:
        # Set up the mocks
        mock_mkdtemp.return_value = "/tmp/mock_temp_dir"
        mock_stream = MagicMock()
        mock_output = MagicMock()
        mock_input.return_value = mock_stream
        mock_stream.output.return_value = mock_output

        # Ensure Path.exists returns True to avoid FileNotFoundError
        with patch.object(Path, "exists", return_value=True):
            # Test conversion without specifying output_dir
            result = converter.convert(example_audio_path)

            # Verify temp dir was created
            mock_mkdtemp.assert_called_once()

            # Verify the output path uses the temp dir
            expected_output = Path("/tmp/mock_temp_dir") / f"{example_audio_path.stem}.wav"
            mock_stream.output.assert_called_once()
            assert str(expected_output) in mock_stream.output.call_args[0][0]
