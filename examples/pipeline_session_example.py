#!/usr/bin/env python
"""
Example script demonstrating the pipeline_session context manager.

This script shows how to use the pipeline_session context manager to efficiently
process multiple audio files by reusing resources.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path if running the script directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhearingai import pipeline_session


def process_audio_files(audio_files, output_dir=None, verbose=False):
    """
    Process multiple audio files using a pipeline session.

    Args:
        audio_files: List of paths to audio files
        output_dir: Directory to save output files (optional)
        verbose: Whether to enable verbose logging

    Returns:
        List of TranscriptionResult objects
    """
    start_time = time.time()
    results = []

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process files with resource reuse
    with pipeline_session(
        verbose=verbose,
        chunk_size_seconds=30.0,  # Process in 30-second chunks
        overlap_seconds=2.0,  # Overlap chunks by 2 seconds
        max_workers=4,  # Use 4 worker threads
    ) as session:
        for i, audio_path in enumerate(audio_files):
            print(f"Processing file {i+1}/{len(audio_files)}: {audio_path}")

            # Determine output path if needed
            output_path = None
            if output_dir:
                output_filename = f"{Path(audio_path).stem}_transcript.txt"
                output_path = os.path.join(output_dir, output_filename)

            # Process the file
            result = session.transcribe(audio_path=audio_path, output_path=output_path)

            results.append(result)

            # Print a summary of the result
            print(f"  Found {len(result.segments)} segments")
            print(f"  Total duration: {result.duration:.2f} seconds")
            if output_path:
                print(f"  Saved transcript to: {output_path}")
            print()

    # Print overall statistics
    total_time = time.time() - start_time
    print(f"Processed {len(audio_files)} files in {total_time:.2f} seconds")
    print(f"Average time per file: {total_time / len(audio_files):.2f} seconds")

    return results


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python pipeline_session_example.py audio_file1.mp3 [audio_file2.mp3 ...]")
        sys.exit(1)

    # Get audio files from command line arguments
    audio_files = sys.argv[1:]

    # Process the files
    output_dir = "transcripts"
    results = process_audio_files(audio_files, output_dir=output_dir, verbose=True)

    # Print a summary of all results
    print("\nSummary of all transcriptions:")
    for i, result in enumerate(results):
        print(f"File {i+1}: {len(result.segments)} segments, {result.duration:.2f} seconds")
