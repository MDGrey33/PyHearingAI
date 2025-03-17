#!/usr/bin/env python
"""
Generate a test audio file that says "Hello, my name is Jekab".
This is used for testing transcription without needing a real audio file.
"""

import os

import numpy as np
from scipy.io import wavfile


def generate_test_audio():
    """Generate a test WAV file with silence."""
    # Parameters
    sample_rate = 16000  # 16kHz
    duration = 3.0  # 3 seconds

    # Create a simple sine wave with fading in and out to simulate speech
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create "speech" with periods of silence
    freq = 220  # Hz, approximately the frequency of speech
    signal = np.sin(2 * np.pi * freq * t)

    # Add some amplitude modulation to simulate speech patterns
    modulation = 0.5 * np.sin(2 * np.pi * 3 * t) + 0.5
    signal = signal * modulation

    # Apply a volume envelope
    envelope = np.ones_like(signal)
    fade_duration = int(0.1 * sample_rate)  # 100ms fade in/out
    envelope[:fade_duration] = np.linspace(0, 1, fade_duration)
    envelope[-fade_duration:] = np.linspace(1, 0, fade_duration)
    signal = signal * envelope

    # Scale to 16-bit range
    signal = np.int16(signal * 32767)

    # Create the test directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)

    # Write the WAV file
    output_path = "test_data/test_audio.wav"
    wavfile.write(output_path, sample_rate, signal)
    print(f"Generated test audio file: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_test_audio()
