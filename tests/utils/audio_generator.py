"""
Utilities for generating test audio files.

This module provides functions to create synthetic audio files for testing
diarization, transcription, and reconciliation functionality.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


def create_test_audio_file(
    path: str,
    duration: float = 10.0,
    sample_rate: int = 16000,
    num_speakers: int = 2,
    speech_segments: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Create a test audio file with specified speech segments.

    Args:
        path: Path where the audio file will be saved
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz
        num_speakers: Number of distinct speakers
        speech_segments: List of dictionaries with speech segment data:
                         [{"speaker": 0, "start": 1.0, "end": 2.0, "text": "Hello"}]

    Returns:
        Path to the created audio file
    """
    # Create base silence
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples)

    # Create speaker tones (different frequencies for each speaker)
    speaker_tones = {}
    base_freq = 440  # A4 note
    for i in range(num_speakers):
        # Each speaker gets a different frequency
        freq = base_freq * (1.2**i)
        speaker_tones[i] = freq

    # Add speech segments if provided
    if speech_segments:
        for segment in speech_segments:
            speaker = segment["speaker"]
            start_time = segment["start"]
            end_time = segment["end"]

            # Convert times to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_duration = end_sample - start_sample

            # Generate a tone for this speaker
            freq = speaker_tones[speaker]
            t = np.linspace(0, (end_time - start_time), segment_duration)
            tone = 0.5 * np.sin(2 * np.pi * freq * t)

            # Apply a simple envelope
            envelope = np.ones_like(tone)
            fade_samples = min(int(0.05 * segment_duration), 1000)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

            # Apply envelope to tone
            tone = tone * envelope

            # Insert into the audio
            audio[start_sample:end_sample] = tone

    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Save the audio file
    sf.write(path, audio, sample_rate)

    return path


def create_multilingual_test_audio(
    path: str,
    duration: float = 10.0,
    sample_rate: int = 16000,
    languages: List[str] = ["en", "es"],
    speech_segments: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Create a test audio file with multilingual speech segments.

    Args:
        path: Path where the audio file will be saved
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz
        languages: List of language codes to simulate
        speech_segments: List of dictionaries with speech segment data including language:
                         [{"speaker": 0, "start": 1.0, "end": 2.0, "text": "Hello", "language": "en"}]

    Returns:
        Path to the created audio file
    """
    # Create base silence
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples)

    # Create speaker tones with language variations
    speaker_tones = {}
    base_freq = 440  # A4 note

    # Get unique speakers from speech segments
    if speech_segments:
        speakers = set(segment["speaker"] for segment in speech_segments)
        num_speakers = len(speakers)

        for i, speaker in enumerate(speakers):
            # Base frequency for this speaker
            speaker_base_freq = base_freq * (1.2**i)

            # Different frequencies for different languages
            speaker_tones[speaker] = {}
            for j, lang in enumerate(languages):
                # Small variation for different languages
                speaker_tones[speaker][lang] = speaker_base_freq * (1.05**j)

        # Add speech segments
        for segment in speech_segments:
            speaker = segment["speaker"]
            start_time = segment["start"]
            end_time = segment["end"]
            language = segment.get("language", languages[0])

            # Convert times to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_duration = end_sample - start_sample

            # Generate a tone for this speaker and language
            freq = speaker_tones[speaker].get(language, speaker_tones[speaker].get(languages[0]))
            t = np.linspace(0, (end_time - start_time), segment_duration)
            tone = 0.5 * np.sin(2 * np.pi * freq * t)

            # Apply a simple envelope
            envelope = np.ones_like(tone)
            fade_samples = min(int(0.05 * segment_duration), 1000)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

            # Apply envelope to tone
            tone = tone * envelope

            # Insert into the audio
            audio[start_sample:end_sample] = tone

    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Save the audio file
    sf.write(path, audio, sample_rate)

    return path


def create_audio_with_silence(
    path: str,
    duration: float = 10.0,
    sample_rate: int = 16000,
    num_speakers: int = 1,
    speech_segments: Optional[List[Dict[str, Any]]] = None,
    silence_regions: Optional[List[Tuple[float, float]]] = None,
) -> str:
    """
    Create a test audio file with explicit silence regions.

    Args:
        path: Path where the audio file will be saved
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz
        num_speakers: Number of distinct speakers
        speech_segments: List of dictionaries with speech segment data
        silence_regions: List of tuples with (start_time, end_time) for explicit silence regions

    Returns:
        Path to the created audio file
    """
    # First create a normal test audio file
    create_test_audio_file(
        path=path,
        duration=duration,
        sample_rate=sample_rate,
        num_speakers=num_speakers,
        speech_segments=speech_segments,
    )

    # If no silence regions specified, just return the created file
    if not silence_regions:
        return path

    # Read the created audio file
    audio, sr = sf.read(path)

    # Apply explicit silence (zero out the regions)
    for start_time, end_time in silence_regions:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Ensure we're within bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        # Zero out the region
        audio[start_sample:end_sample] = 0

    # Save the modified audio
    sf.write(path, audio, sample_rate)

    return path
