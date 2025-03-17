#!/usr/bin/env python
"""
Generate a synthetic audio file for testing.

This script creates a synthetic audio file with multiple tones at different
frequencies to simulate different speakers or segments.
"""

import numpy as np
import soundfile as sf

# Parameters
sample_rate = 16000  # Hz
duration = 180.0  # 3 minutes total
output_file = "test_audio.wav"

# Create time array
t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

# Start with silence
audio = np.zeros_like(t)

# Add different tones to simulate different speakers/segments
# Speaker 1: 440 Hz (A4)
mask1 = ((t >= 0) & (t < 30)) | ((t >= 60) & (t < 90)) | ((t >= 120) & (t < 150))
audio[mask1] += 0.3 * np.sin(2 * np.pi * 440 * t[mask1])

# Speaker 2: 330 Hz (E4)
mask2 = ((t >= 30) & (t < 60)) | ((t >= 90) & (t < 120)) | ((t >= 150) & (t < 180))
audio[mask2] += 0.3 * np.sin(2 * np.pi * 330 * t[mask2])

# Add some silence gaps to test silence detection
for i in range(18):
    gap_start = i * 10
    gap_end = gap_start + 0.5
    gap_mask = (t >= gap_start) & (t < gap_end)
    audio[gap_mask] = 0

print(f"Generating {duration} seconds of audio at {sample_rate} Hz sample rate")
sf.write(output_file, audio, sample_rate)
print(f"Audio saved to {output_file}")
