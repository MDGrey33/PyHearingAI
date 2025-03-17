#!/usr/bin/env python
"""
Example script demonstrating memory management in PyHearingAI.

This script shows how to use the memory management features to control
resource usage when processing large audio files.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path if running the script directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhearingai import cleanup_resources, set_memory_limit, transcribe_chunked


def with_memory_limit(audio_path, limit_mb=1024):
    """
    Process an audio file with a memory limit.

    Args:
        audio_path: Path to the audio file
        limit_mb: Memory limit in MB
    """
    print(f"Processing audio with {limit_mb}MB memory limit: {audio_path}")

    # Set memory limit before processing
    set_memory_limit(limit_mb)

    # Start timing
    start_time = time.time()

    # Process with chunking to manage memory usage
    result = transcribe_chunked(
        audio_path=audio_path,
        chunk_size_seconds=60.0,  # Process in 1-minute chunks
        overlap_seconds=5.0,  # 5 seconds overlap between chunks
        verbose=True,
        show_chunks=True,
    )

    # Calculate processing time
    duration = time.time() - start_time
    print(f"Processing completed in {duration:.2f} seconds")
    print(f"Processed {result.duration:.2f} seconds of audio")
    print(f"Found {len(result.segments)} segments")

    # Show a sample of the transcript
    print("\nSample transcript:")
    sample_length = min(500, len(result.full_text))
    print(f"{result.full_text[:sample_length]}...")


def process_with_cleanup(audio_path, output_path=None):
    """
    Process an audio file with periodic resource cleanup.

    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save the output
    """
    print(f"Processing with manual cleanup: {audio_path}")

    # Set a memory limit
    set_memory_limit(2048)  # 2GB limit

    # Start timing
    start_time = time.time()

    try:
        # Process with chunking
        result = transcribe_chunked(
            audio_path=audio_path,
            output_path=output_path,
            chunk_size_seconds=300.0,  # 5-minute chunks
            overlap_seconds=10.0,  # 10 seconds overlap
            verbose=True,
        )

        print(f"Transcription complete, found {len(result.segments)} segments")

        # Explicitly clean up resources after processing
        print("Manually cleaning up resources...")
        freed_mb = cleanup_resources()
        print(f"Freed {freed_mb:.2f}MB of memory")

        # Calculate processing time
        duration = time.time() - start_time
        print(f"Total processing time: {duration:.2f} seconds")

        return result

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        # Clean up even if there was an error
        cleanup_resources()
        raise


def print_memory_usage():
    """Print current memory usage if psutil is available."""
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.2f}MB")
    except ImportError:
        print("psutil not available, cannot measure memory usage")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python memory_management_example.py audio_file.mp3 [memory_limit_mb]")
        sys.exit(1)

    # Get audio file path from command line argument
    audio_path = sys.argv[1]

    # Get optional memory limit
    memory_limit = 1024  # Default to 1GB
    if len(sys.argv) >= 3:
        try:
            memory_limit = int(sys.argv[2])
        except ValueError:
            print(f"Invalid memory limit: {sys.argv[2]}. Using default: {memory_limit}MB")

    # Print initial memory usage
    print("Before processing:")
    print_memory_usage()

    # Process with memory limit
    output_path = None
    if len(sys.argv) >= 4:
        output_path = sys.argv[3]

    # Option 1: Simple processing with memory limit
    with_memory_limit(audio_path, memory_limit)

    # Print memory usage after processing
    print("\nAfter processing:")
    print_memory_usage()

    # Option 2: Process with manual cleanup
    print("\nDemonstrating manual cleanup:")
    process_with_cleanup(audio_path, output_path)

    # Print memory usage after cleanup
    print("\nAfter manual cleanup:")
    print_memory_usage()
