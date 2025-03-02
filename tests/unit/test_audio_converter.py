from pathlib import Path

import pytest


def test_audio_converter_basic(example_audio_path, temp_dir):
    """Test basic functionality of the audio converter."""
    from pyhearingai.infrastructure.audio_converter import FFmpegAudioConverter

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
