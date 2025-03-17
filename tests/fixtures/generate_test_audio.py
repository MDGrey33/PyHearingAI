#!/usr/bin/env python3
"""
Generate synthetic multilingual audio files for testing.

This script creates synthetic audio files with known content for testing
transcription and diarization services. It can generate multilingual audio
with different speakers and predefined text.
"""

import argparse
import logging
import math
import os
import random
import struct
import sys
import uuid
import wave
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Sample rates and durations
DEFAULT_SAMPLE_RATE = 16000  # 16kHz
DEFAULT_DURATION = 5.0  # 5 seconds


def generate_sine_wave(
    frequency: float, duration: float, sample_rate: int, amplitude: float = 0.5
) -> bytes:
    """
    Generate a simple sine wave.

    Args:
        frequency: Frequency of the sine wave in Hz
        duration: Duration of the audio in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude of the sine wave (0.0-1.0)

    Returns:
        Audio data as bytes
    """
    num_samples = int(duration * sample_rate)
    data = b""

    for i in range(num_samples):
        t = float(i) / sample_rate
        value = amplitude * math.sin(2 * math.pi * frequency * t)
        # Convert to 16-bit PCM
        sample = int(value * 32767)
        data += struct.pack("<h", sample)

    return data


def generate_silent_audio(duration: float, sample_rate: int) -> bytes:
    """
    Generate silent audio.

    Args:
        duration: Duration of the audio in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio data as bytes
    """
    num_samples = int(duration * sample_rate)
    return struct.pack("<" + "h" * num_samples, *([0] * num_samples))


def generate_white_noise(duration: float, sample_rate: int, amplitude: float = 0.1) -> bytes:
    """
    Generate white noise.

    Args:
        duration: Duration of the audio in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude of the noise (0.0-1.0)

    Returns:
        Audio data as bytes
    """
    num_samples = int(duration * sample_rate)
    data = b""

    for _ in range(num_samples):
        value = amplitude * (random.random() * 2 - 1)
        # Convert to 16-bit PCM
        sample = int(value * 32767)
        data += struct.pack("<h", sample)

    return data


def create_wav_file(path: Path, data: bytes, sample_rate: int) -> None:
    """
    Create a WAV file with the given data.

    Args:
        path: Path to the output WAV file
        data: Audio data as bytes
        sample_rate: Sample rate in Hz
    """
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data)


def generate_test_audio(
    output_dir: Path,
    duration: float = DEFAULT_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    language: str = "en",
) -> Path:
    """
    Generate a test audio file with synthetic content.

    Args:
        output_dir: Directory to save the audio file
        duration: Duration of the audio in seconds
        sample_rate: Sample rate in Hz
        language: Language code for metadata

    Returns:
        Path to the generated audio file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename
    filename = f"test_audio_{language}_{uuid.uuid4().hex[:8]}.wav"
    output_path = output_dir / filename

    # Create different audio patterns based on language
    if language == "en":
        # English: sine wave at 440 Hz (A4 note)
        data = generate_sine_wave(440, duration, sample_rate)
    elif language == "fr":
        # French: sine wave at 329.63 Hz (E4 note)
        data = generate_sine_wave(329.63, duration, sample_rate)
    elif language == "de":
        # German: sine wave at 392 Hz (G4 note)
        data = generate_sine_wave(392, duration, sample_rate)
    elif language == "es":
        # Spanish: sine wave at 261.63 Hz (C4 note)
        data = generate_sine_wave(261.63, duration, sample_rate)
    elif language == "multi":
        # Multilingual: alternating tones
        data = b""
        segment_duration = duration / 4
        data += generate_sine_wave(440, segment_duration, sample_rate)  # English
        data += generate_sine_wave(329.63, segment_duration, sample_rate)  # French
        data += generate_sine_wave(392, segment_duration, sample_rate)  # German
        data += generate_sine_wave(261.63, segment_duration, sample_rate)  # Spanish
    else:
        # Unknown language: white noise
        data = generate_white_noise(duration, sample_rate)

    # Create the WAV file
    create_wav_file(output_path, data, sample_rate)

    logger.info(f"Generated {duration}s {language} test audio: {output_path}")
    return output_path


def generate_test_audio_with_speakers(
    output_dir: Path,
    num_speakers: int = 2,
    duration_per_speaker: float = 2.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    language: str = "en",
) -> Path:
    """
    Generate a test audio file with multiple synthetic speakers.

    Args:
        output_dir: Directory to save the audio file
        num_speakers: Number of speakers in the audio
        duration_per_speaker: Duration for each speaker in seconds
        sample_rate: Sample rate in Hz
        language: Language code for metadata

    Returns:
        Path to the generated audio file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename
    filename = f"test_audio_{language}_{num_speakers}speakers_{uuid.uuid4().hex[:8]}.wav"
    output_path = output_dir / filename

    # Base frequencies for each speaker (different notes)
    speaker_frequencies = [
        261.63,  # C4
        293.66,  # D4
        329.63,  # E4
        349.23,  # F4
        392.00,  # G4
        440.00,  # A4
        493.88,  # B4
        523.25,  # C5
    ]

    # Ensure we have enough frequencies
    while len(speaker_frequencies) < num_speakers:
        speaker_frequencies.extend(speaker_frequencies)

    # Create audio with different speakers
    data = b""
    for i in range(num_speakers):
        # Add a short silence between speakers (0.2 seconds)
        if i > 0:
            data += generate_silent_audio(0.2, sample_rate)

        # Add speaker audio
        frequency = speaker_frequencies[i]
        data += generate_sine_wave(frequency, duration_per_speaker, sample_rate)

    # Create the WAV file
    create_wav_file(output_path, data, sample_rate)

    logger.info(f"Generated {num_speakers} speaker test audio: {output_path}")
    return output_path


def main():
    """Run the script."""
    parser = argparse.ArgumentParser(description="Generate synthetic audio for testing")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/fixtures/transcription",
        help="Directory to save the generated audio files",
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION, help="Duration of the audio in seconds"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Sample rate in Hz"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "fr", "de", "es", "multi"],
        help="Language code for the audio",
    )
    parser.add_argument(
        "--num-speakers", type=int, default=2, help="Number of speakers in the audio"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all test files (single and multi-speaker, all languages)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.all:
        # Generate all language variants
        for lang in ["en", "fr", "de", "es", "multi"]:
            generate_test_audio(
                output_dir=output_dir,
                duration=args.duration,
                sample_rate=args.sample_rate,
                language=lang,
            )

        # Generate speaker variants (2, 3, 4 speakers)
        for num_speakers in [2, 3, 4]:
            generate_test_audio_with_speakers(
                output_dir=output_dir,
                num_speakers=num_speakers,
                duration_per_speaker=args.duration / num_speakers,
                sample_rate=args.sample_rate,
                language="en",
            )
    else:
        # Generate a single audio file
        generate_test_audio(
            output_dir=output_dir,
            duration=args.duration,
            sample_rate=args.sample_rate,
            language=args.language,
        )

        # Generate multi-speaker audio if requested
        if args.num_speakers > 1:
            generate_test_audio_with_speakers(
                output_dir=output_dir,
                num_speakers=args.num_speakers,
                duration_per_speaker=args.duration / args.num_speakers,
                sample_rate=args.sample_rate,
                language=args.language,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
