"""
Audio fixtures for testing.

This module provides fixtures for creating test audio files with various properties,
useful for testing audio processing functionality.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def create_test_audio():
    """
    Create a test audio file with configurable parameters.

    This fixture returns a function that can create audio files with various parameters,
    making it flexible for different test scenarios.

    Returns:
        function: A function that creates and returns a path to a test audio file

    Usage:
        def test_something(create_test_audio):
            # Create a default 1-second audio file
            audio_path = create_test_audio()

            # Create a 3-second stereo file with custom parameters
            audio_path = create_test_audio(
                duration=3.0,
                sample_rate=44100,
                channels=2,
                frequency=440,
                amplitude=0.5
            )
    """

    def _create_audio(
        duration=1.0,
        sample_rate=16000,
        channels=1,
        frequency=440,
        amplitude=0.1,
        noise_level=0.01,
        filename=None,
        output_dir=None,
    ):
        """
        Create a test audio file with the specified parameters.

        Args:
            duration (float): Duration of audio in seconds
            sample_rate (int): Sample rate in Hz
            channels (int): Number of audio channels (1=mono, 2=stereo)
            frequency (float): Frequency of test tone in Hz
            amplitude (float): Amplitude of test tone (0.0-1.0)
            noise_level (float): Level of background noise to add
            filename (str): Optional filename, otherwise a temporary file is created
            output_dir (str): Optional output directory, otherwise a temporary dir is used

        Returns:
            Path: Path to the created audio file
        """
        import soundfile as sf

        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # Generate sine wave
        audio = amplitude * np.sin(2 * np.pi * frequency * t)

        # Add some noise
        if noise_level > 0:
            noise = noise_level * np.random.normal(0, 1, len(t))
            audio = audio + noise

        # Create stereo if needed
        if channels == 2:
            # For stereo, create a slightly different second channel
            audio2 = amplitude * np.sin(2 * np.pi * (frequency * 1.01) * t)
            if noise_level > 0:
                noise2 = noise_level * np.random.normal(0, 1, len(t))
                audio2 = audio2 + noise2
            audio = np.column_stack((audio, audio2))

        # Create output path
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir

        if filename is None:
            filename = f"test_audio_{int(frequency)}hz_{int(duration * 1000)}ms.wav"

        audio_path = Path(output_dir) / filename

        # Save to disk
        sf.write(audio_path, audio, sample_rate)

        return audio_path

    return _create_audio


@pytest.fixture
def create_multi_speaker_audio():
    """
    Create a test audio file with multiple speakers.

    This fixture returns a function that creates audio files with simulated speech
    from multiple speakers at specified time intervals. Useful for testing
    diarization functionality.

    Returns:
        function: A function that creates and returns a path to a test audio file

    Usage:
        def test_diarization(create_multi_speaker_audio):
            audio_path = create_multi_speaker_audio(
                speech_segments=[
                    {"speaker": 0, "start": 0.0, "end": 2.0},
                    {"speaker": 1, "start": 2.5, "end": 4.0},
                    {"speaker": 0, "start": 4.5, "end": 6.0},
                ]
            )
    """

    def _create_multi_speaker_audio(
        duration=10.0,
        sample_rate=16000,
        speech_segments=None,
        num_speakers=2,
        filename=None,
        output_dir=None,
    ):
        """
        Create a test audio file with multiple speakers.

        Args:
            duration (float): Total duration of audio in seconds
            sample_rate (int): Sample rate in Hz
            speech_segments (list): List of dicts with speaker, start, end times
            num_speakers (int): Number of speakers to simulate if speech_segments not provided
            filename (str): Optional filename, otherwise a temporary file is created
            output_dir (str): Optional output directory

        Returns:
            Path: Path to the created audio file
        """
        import soundfile as sf

        # Initialize silent audio
        samples = np.zeros(int(sample_rate * duration))

        # Define speech segments if not provided
        if speech_segments is None:
            # Create default segments with alternating speakers
            segment_duration = 2.0
            gap_duration = 0.5
            speech_segments = []

            current_time = 0.0
            current_speaker = 0

            while current_time + segment_duration <= duration:
                speech_segments.append(
                    {
                        "speaker": current_speaker,
                        "start": current_time,
                        "end": current_time + segment_duration,
                    }
                )

                current_time += segment_duration + gap_duration
                current_speaker = (current_speaker + 1) % num_speakers

        # Generate audio for each speech segment
        for segment in speech_segments:
            speaker_id = segment["speaker"]
            start_time = segment["start"]
            end_time = segment["end"]

            # Convert times to sample indices
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)

            if end_idx > len(samples):
                end_idx = len(samples)

            # Generate speech-like audio for this segment
            # Use different frequency ranges for different speakers
            base_freq = 120 + (speaker_id * 60)  # Different fundamental frequency per speaker

            t = np.arange(start_idx, end_idx) / sample_rate
            segment_audio = np.zeros(end_idx - start_idx)

            # Add fundamental frequency
            segment_audio += 0.5 * np.sin(2 * np.pi * base_freq * t)

            # Add some harmonics to make it sound more speech-like
            segment_audio += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
            segment_audio += 0.1 * np.sin(2 * np.pi * base_freq * 3 * t)

            # Add amplitude modulation to simulate syllables
            syllable_rate = 4  # syllables per second
            segment_audio *= 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t)

            # Add to main audio
            samples[start_idx:end_idx] = segment_audio

        # Normalize audio
        if np.max(np.abs(samples)) > 0:
            samples = samples / np.max(np.abs(samples)) * 0.8

        # Create output path
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir

        if filename is None:
            filename = f"multi_speaker_{num_speakers}_{int(duration)}s.wav"

        audio_path = Path(output_dir) / filename

        # Save to disk
        sf.write(audio_path, samples, sample_rate)

        return audio_path

    return _create_multi_speaker_audio
