#!/usr/bin/env python
"""
Command-line interface for PyHearingAI.

This module provides a CLI for the PyHearingAI library, allowing users to
transcribe audio files with speaker diarization from the command line.
"""

import argparse
import os
import sys
from pathlib import Path

from pyhearingai import __version__
from pyhearingai.application.transcribe import transcribe


def main():
    """Main CLI entry point for PyHearingAI."""
    # Configure the argument parser with helpful description
    parser = argparse.ArgumentParser(
        description="PyHearingAI - Transcribe audio with speaker diarization",
        epilog="""
Examples:
  transcribe recording.mp3                  # Transcribe using default settings
  transcribe -s recording.mp3 -o output.txt # Specify source and output
  transcribe recording.mp3 -f json          # Output in JSON format

Supported models:
  Transcriber: whisper_openai (default) - Uses OpenAI's Whisper API
  Diarizer: pyannote (default) - Uses Pyannote for speaker diarization

For more information, visit: https://github.com/MDGrey33/PyHearingAI
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "audio_file", type=str, nargs="?", help="Path to the audio file to transcribe"
    )
    input_group.add_argument(
        "-s", "--source", type=str, help="Path to the audio file to transcribe"
    )

    # Output options
    parser.add_argument(
        "-o", "--output", type=str, help="Output file path (default: based on input file)"
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="txt",
        choices=["txt", "json", "srt", "vtt", "md"],
        help="Output format (default: txt)",
    )

    # API keys
    parser.add_argument(
        "--openai-key",
        type=str,
        help="OpenAI API key (default: from OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--huggingface-key",
        type=str,
        help="Hugging Face API key (default: from HUGGINGFACE_API_KEY environment variable)",
    )

    # Other options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # Get the audio file path
    audio_file = args.audio_file if args.audio_file else args.source
    audio_path = Path(audio_file)

    # Validate that the audio file exists
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_file}", file=sys.stderr)
        return 1

    # Determine the output path if not specified
    output_path = None
    if args.output:
        output_path = Path(args.output)
    else:
        # Default output path: replace input extension with format
        output_path = audio_path.with_suffix(f".{args.format}")

    # Prepare kwargs for API keys
    kwargs = {}

    # Check OpenAI API key
    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print(
            "Warning: OpenAI API key not found. Please set it using one of these methods:",
            file=sys.stderr,
        )
        print("  1. Set OPENAI_API_KEY environment variable", file=sys.stderr)
        print("  2. Provide it with --openai-key parameter", file=sys.stderr)
    else:
        # Pass directly to the API
        kwargs["api_key"] = openai_key
        # Also set environment variable for components that use it directly
        os.environ["OPENAI_API_KEY"] = openai_key

    # Check Hugging Face API key
    huggingface_key = args.huggingface_key or os.environ.get("HUGGINGFACE_API_KEY")
    if not huggingface_key:
        print(
            "Warning: Hugging Face API key not found. Please set it using one of these methods:",
            file=sys.stderr,
        )
        print("  1. Set HUGGINGFACE_API_KEY environment variable", file=sys.stderr)
        print("  2. Provide it with --huggingface-key parameter", file=sys.stderr)
    else:
        # Pass directly to the API
        kwargs["huggingface_api_key"] = huggingface_key
        # Also set environment variable for components that use it directly
        os.environ["HUGGINGFACE_API_KEY"] = huggingface_key

    # Call the transcribe function
    try:
        result = transcribe(audio_path=audio_file, verbose=args.verbose, **kwargs)

        # Save the result
        result.save(output_path, format=args.format)
        print(f"Transcription saved to: {output_path}")

        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
