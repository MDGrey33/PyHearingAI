"""Test that verifies our fixtures work properly."""

from pathlib import Path

import pytest


def test_create_test_audio_fixture(create_test_audio):
    """Test that the create_test_audio fixture works properly."""
    # Call the fixture to create an audio file
    audio_path = create_test_audio(duration=1.0, sample_rate=16000)

    # Verify the file exists
    assert audio_path.exists()
    assert audio_path.is_file()

    # Verify the filename
    assert audio_path.name.endswith(".wav")


def test_create_multi_speaker_audio_fixture(create_multi_speaker_audio):
    """Test that the create_multi_speaker_audio fixture works properly."""
    # Call the fixture to create an audio file with speaker segments
    speech_segments = [
        {"speaker": 0, "start": 0.0, "end": 1.0},
        {"speaker": 1, "start": 1.5, "end": 2.5},
    ]

    audio_path = create_multi_speaker_audio(
        duration=3.0, num_speakers=2, speech_segments=speech_segments
    )

    # Verify the file exists
    assert audio_path.exists()
    assert audio_path.is_file()

    # Verify the filename
    assert audio_path.name.endswith(".wav")
