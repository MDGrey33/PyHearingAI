#!/usr/bin/env python
"""
Example script demonstrating the transcribe_chunked function.

This script shows how to use the transcribe_chunked function to process
large audio files in manageable chunks.
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path if running the script directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhearingai import transcribe_chunked


def process_large_audio_file(audio_path, output_path=None, verbose=False):
    """
    Process a large audio file using chunked transcription.

    Args:
        audio_path: Path to the audio file
        output_path: Path for the output file (optional)
        verbose: Whether to enable verbose logging

    Returns:
        TranscriptionResult object
    """
    print(f"Processing large audio file: {audio_path}")
    print(f"This will process the file in chunks to manage memory usage")

    start_time = time.time()

    # Process the file in chunks
    result = transcribe_chunked(
        audio_path=audio_path,
        output_path=output_path,
        chunk_size_seconds=600,  # Process in 10-minute chunks
        overlap_seconds=30,  # Overlap chunks by 30 seconds
        verbose=verbose,
        show_chunks=True,  # Show progress for each chunk
    )

    # Print statistics
    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    print(f"Found {len(result.segments)} segments")
    print(f"Total duration: {result.duration:.2f} seconds")

    if output_path:
        print(f"Saved transcript to: {output_path}")

    return result


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python transcribe_chunked_example.py audio_file.mp3 [output_file.txt]")
        sys.exit(1)

    # Get audio file path from command line argument
    audio_path = sys.argv[1]

    # Get output path if provided
    output_path = None
    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        # Create default output path
        output_path = str(Path(audio_path).with_suffix(".txt"))

    # Process the file
    result = process_large_audio_file(audio_path, output_path=output_path, verbose=True)

    # Print a sample of the transcript
    print("\nSample of transcript:")
    sample_text = (
        result.full_text[:500] + "..." if len(result.full_text) > 500 else result.full_text
    )
    print(sample_text)
