#!/usr/bin/env python
"""
Command-line interface for PyHearingAI.

This module provides a CLI for the PyHearingAI library, allowing users to
transcribe audio files with speaker diarization from the command line.
"""

import argparse
import sys
from pathlib import Path

from pyhearingai import __version__
from pyhearingai.application.transcribe import transcribe


def main():
    """Main CLI entry point for PyHearingAI."""
    parser = argparse.ArgumentParser(
        description="PyHearingAI - Transcribe audio with speaker diarization"
    )
    parser.add_argument("audio_file", type=str, help="Path to the audio file to transcribe")
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
    parser.add_argument(
        "-t",
        "--transcriber",
        type=str,
        default="whisper_openai",
        help="Transcription model to use (default: whisper_openai)",
    )
    parser.add_argument(
        "-d",
        "--diarizer",
        type=str,
        default="pyannote",
        help="Diarization model to use (default: pyannote)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # Validate that the audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        return 1

    # Determine the output path if not specified
    output_path = None
    if args.output:
        output_path = Path(args.output)

    # Call the transcribe function
    try:
        result = transcribe(
            audio_path=args.audio_file,
            transcriber=args.transcriber,
            diarizer=args.diarizer,
            verbose=args.verbose,
        )

        # Save the result
        if output_path:
            result.save(output_path, format=args.format)
        else:
            # Default output path: replace input extension with format
            default_output = audio_path.with_suffix(f".{args.format}")
            result.save(default_output)
            print(f"Transcription saved to: {default_output}")

        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
