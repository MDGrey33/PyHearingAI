#!/usr/bin/env python
"""
Mock transcription command for PyHearingAI.

This script simulates the PyHearingAI transcription command but uses mock data instead of real API calls.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import PyHearingAI components
from pyhearingai import initialization
from pyhearingai.core.models import DiarizationSegment, Segment, TranscriptionResult


def mock_transcribe(audio_path, start_time=None, end_time=None):
    """
    Mock transcription function that creates a fake result.

    Args:
        audio_path: Path to the audio file
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        A mock TranscriptionResult
    """
    print(f"Processing audio file: {audio_path}")

    if start_time is not None and end_time is not None:
        print(f"Time range: {start_time}s to {end_time}s")

    # Create mock segments with speaker IDs
    segments = [
        Segment(text="Hello, my name is Jekab.", start=1.0, end=3.0, speaker_id="SPEAKER_01")
    ]

    # Create mock result
    # Note: Examining the TranscriptionResult constructor to use correct parameters
    result = TranscriptionResult(audio_path=Path(audio_path), segments=segments)

    return result


def main():
    """Main CLI entry point for mock transcription."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Mock PyHearingAI transcription command")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--start-time", type=float, help="Start time in seconds")
    parser.add_argument("--end-time", type=float, help="End time in seconds")
    parser.add_argument("-o", "--output", help="Output file path")

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.audio_file).with_suffix(".txt")

    # Simulate processing
    print("Starting audio processing...")

    # Simulate chunk creation
    time.sleep(0.5)
    print("Created audio chunks")

    # Simulate diarization
    time.sleep(0.5)
    print("Completed diarization")

    # Simulate transcription
    time.sleep(0.5)
    print("Completed transcription")

    # Get mock result
    result = mock_transcribe(args.audio_file, args.start_time, args.end_time)

    # Save result
    with open(output_path, "w") as f:
        f.write("SPEAKER_01: Hello, my name is Jekab.\n")

    print(f"Transcription saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
