#!/usr/bin/env python
"""
Example script demonstrating how to process a specific time range of an audio file.

This script shows how to use the time range parameters in PyHearingAI to process
only a portion of an audio file, which is useful for testing or when you're only
interested in a specific part of a recording.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def process_time_range(audio_path, start_time=None, end_time=None, sample_duration=None):
    """
    Process a specific time range of an audio file using the CLI.

    Args:
        audio_path: Path to the audio file
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        sample_duration: Duration to process in seconds (optional)
    """
    print(f"Processing audio file: {audio_path}")

    # Build the command
    cmd = ["python", "-m", "pyhearingai", audio_path, "--verbose"]

    # Add time range parameters
    if start_time is not None:
        cmd.extend(["--start-time", str(start_time)])
        print(f"Starting at: {start_time} seconds")

    if end_time is not None:
        cmd.extend(["--end-time", str(end_time)])
        print(f"Ending at: {end_time} seconds")

    if sample_duration is not None:
        start = start_time or 0
        cmd.extend(["--start-time", str(start), "--end-time", str(start + sample_duration)])
        print(f"Processing {sample_duration} seconds from {start}s to {start + sample_duration}s")

    # Set output path
    output_path = f"{Path(audio_path).stem}_sample.txt"
    cmd.extend(["-o", output_path])

    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    subprocess.run(cmd)
    end_time = time.time()

    # Print processing time
    print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
    print(f"Saved transcript to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python time_range_example.py audio_file.mp3 [start_time] [end_time]")
        print("   or: python time_range_example.py audio_file.mp3 --sample DURATION")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Check for sample duration flag
    if len(sys.argv) > 2 and sys.argv[2] == "--sample" and len(sys.argv) > 3:
        sample_duration = float(sys.argv[3])
        process_time_range(audio_path, sample_duration=sample_duration)

    # Check for start and end times
    elif len(sys.argv) > 2:
        start_time = float(sys.argv[2])
        end_time = float(sys.argv[3]) if len(sys.argv) > 3 else None
        process_time_range(audio_path, start_time=start_time, end_time=end_time)

    # Process the entire file
    else:
        process_time_range(audio_path)
