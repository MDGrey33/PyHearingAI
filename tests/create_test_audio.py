#!/usr/bin/env python3
"""
Generate a test audio file for integration tests.

This script creates a simple WAV file with two distinct "speakers"
represented by different frequency tones, with some silence in between.
This allows us to test the diarization functionality without requiring
large real-world audio files.
"""

import numpy as np
from scipy.io import wavfile
import os
from pathlib import Path

# Configuration
SAMPLE_RATE = 16000  # 16kHz
DURATION = 10  # seconds
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Create fixtures directory if it doesn't exist
os.makedirs(FIXTURES_DIR, exist_ok=True)
output_file = FIXTURES_DIR / "test_audio.wav"

# Generate time array
t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, False)

# Create a signal with two different "speakers" (different frequencies)
# Speaker 1: 440 Hz tone (0-2s and 6-8s)
# Speaker 2: 880 Hz tone (3-5s and 8-10s)
# With some silence in between
signal = np.zeros_like(t)

# Speaker 1 segments
speaker1_mask = ((t >= 0) & (t < 2)) | ((t >= 6) & (t < 8))
signal[speaker1_mask] = 0.5 * np.sin(2 * np.pi * 440 * t[speaker1_mask])

# Speaker 2 segments
speaker2_mask = ((t >= 3) & (t < 5)) | ((t >= 8) & (t < 10))
signal[speaker2_mask] = 0.5 * np.sin(2 * np.pi * 880 * t[speaker2_mask])

# Convert to int16 for WAV file
signal_int16 = (signal * 32767).astype(np.int16)

# Write WAV file
wavfile.write(output_file, SAMPLE_RATE, signal_int16)

print(f"Created test audio file: {output_file}")
print(f"Duration: {DURATION} seconds")
print(f"Sample rate: {SAMPLE_RATE} Hz")
print(f"File size: {os.path.getsize(output_file)} bytes")
print("Speaker segments:")
print("  Speaker 1: 0-2s and 6-8s (440 Hz)")
print("  Speaker 2: 3-5s and 8-10s (880 Hz)")
print("  Silence: 2-3s and 5-6s")
