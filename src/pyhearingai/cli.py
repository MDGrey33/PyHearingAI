#!/usr/bin/env python
"""
Command-line interface for PyHearingAI.

This module provides a CLI for the PyHearingAI library, allowing users to
transcribe audio files with speaker diarization from the command line.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Import initialization module to ensure components are registered
from pyhearingai import __version__, initialization
from pyhearingai.application.transcribe import transcribe
from pyhearingai.config import IdempotentProcessingConfig
from pyhearingai.core.idempotent import ProcessingStatus
from pyhearingai.infrastructure.repositories.json_repositories import JsonJobRepository


def list_jobs():
    """List all jobs in the repository."""
    job_repo = JsonJobRepository()
    jobs = job_repo.list_all()

    if not jobs:
        print("No jobs found.")
        return

    print(f"Found {len(jobs)} jobs:")
    print("-" * 80)
    print(f"{'ID':<36} | {'Status':<12} | {'Created':<19} | {'Audio File'}")
    print("-" * 80)

    for job in jobs:
        print(
            f"{job.id:<36} | {job.status.name:<12} | {job.created_at:<19} | {job.original_audio_path}"
        )


def find_job_by_audio_path(audio_path):
    """Find a job by its audio path."""
    job_repo = JsonJobRepository()
    jobs = job_repo.list_all()

    audio_path = Path(audio_path).resolve()

    for job in jobs:
        if Path(job.original_audio_path).resolve() == audio_path:
            return job

    return None


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
  transcribe --resume JOB_ID                # Resume a previously interrupted job
  transcribe --list-jobs                    # List all processing jobs

Supported models:
  Transcriber: whisper_openai (default) - Uses OpenAI's Whisper API
  Diarizer: pyannote (default) - Uses Pyannote for speaker diarization

For more information, visit: https://github.com/MDGrey33/PyHearingAI
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "audio_file", type=str, nargs="?", help="Path to the audio file to transcribe"
    )
    input_group.add_argument(
        "-s", "--source", type=str, help="Path to the audio file to transcribe"
    )
    input_group.add_argument(
        "--resume", type=str, help="Resume processing with the specified job ID"
    )
    input_group.add_argument("--list-jobs", action="store_true", help="List all processing jobs")

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

    # Processing options
    parser.add_argument("--show-chunks", action="store_true", help="Show chunk processing progress")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers for processing (default: auto)",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=10.0,
        help="Size of audio chunks in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory for caching intermediate results (default: system temp dir)",
    )
    parser.add_argument(
        "--use-legacy",
        action="store_true",
        help="Use legacy (non-idempotent) processing mode instead of the new resumable mode",
    )

    # Time range options
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Start time in seconds to process only a portion of the audio (optional)",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="End time in seconds to process only a portion of the audio (optional)",
    )
    parser.add_argument(
        "--sample-duration",
        type=float,
        default=None,
        help="Process only a sample of specified duration in seconds from the start of the file",
    )

    # Chunking options
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap between consecutive chunks in seconds (default: 0.0)",
    )

    # Batch processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=180,
        help="Size of each batch in seconds for reconciliation (default: 180, 3 minutes)",
    )
    parser.add_argument(
        "--batch-overlap",
        type=int,
        default=10,
        help="Overlap between batches in seconds (default: 10)",
    )
    parser.add_argument(
        "--use-batched-reconciliation",
        action="store_true",
        help="Use batched reconciliation for long files to avoid token limits",
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

    # Additional options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force new job creation even if a job already exists for this audio file",
    )

    args = parser.parse_args()

    # Handle list-jobs command
    if args.list_jobs:
        list_jobs()
        return 0

    # Handle resume command
    job_id = None
    if args.resume:
        job_id = args.resume
        # Validate that the job exists
        job_repo = JsonJobRepository()
        job = job_repo.get_by_id(job_id)
        if not job:
            print(f"Error: Job not found with ID: {job_id}", file=sys.stderr)
            return 1

        # Set the audio file from the job
        audio_path = job.original_audio_path
        print(f"Resuming job {job_id} for audio file: {audio_path}")

        # Check if job is already completed
        if job.status == ProcessingStatus.COMPLETED:
            print(f"Job {job_id} is already completed. No need to resume.", file=sys.stderr)
            return 0
    else:
        # Get the audio file path for a new job
        audio_file = args.audio_file if args.audio_file else args.source
        if not audio_file:
            parser.print_help()
            return 1

        audio_path = Path(audio_file)

        # Validate that the audio file exists
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_file}", file=sys.stderr)
            return 1

        # Check if there's an existing job for this audio file
        existing_job = find_job_by_audio_path(audio_path)
        if existing_job and existing_job.status != ProcessingStatus.COMPLETED and not args.force:
            print(f"Found existing job {existing_job.id} for this audio file.", file=sys.stderr)
            print(f"You can resume it with: transcribe --resume {existing_job.id}", file=sys.stderr)

            # Ask user if they want to resume
            response = input("Do you want to resume this job? (y/n): ")
            if response.lower() in ["y", "yes"]:
                job_id = existing_job.id
                print(f"Resuming job {job_id}")

    # Determine the output path if not specified
    output_path = None
    if args.output:
        output_path = Path(args.output)
    else:
        # Default output path: replace input extension with format
        output_path = Path(audio_path).with_suffix(f".{args.format}")

    # Prepare kwargs for API keys and additional options
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

    # Add additional processing options
    if args.max_workers:
        kwargs["max_workers"] = args.max_workers

    if args.chunk_size:
        kwargs["chunk_size"] = args.chunk_size

    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    # Set idempotent processing flags
    kwargs["use_idempotent_processing"] = not args.use_legacy
    kwargs["show_chunks"] = args.show_chunks

    # Handle time range parameters
    if args.start_time is not None:
        kwargs["start_time"] = args.start_time

    if args.end_time is not None:
        kwargs["end_time"] = args.end_time

    # Sample duration is a convenience parameter that sets end_time based on start_time + duration
    if args.sample_duration is not None:
        start = args.start_time or 0
        kwargs["start_time"] = start
        kwargs["end_time"] = start + args.sample_duration
        if args.verbose:
            print(f"Processing time range: {start}s to {start + args.sample_duration}s")

    # Pass job ID if resuming
    if job_id:
        kwargs["job_id"] = job_id

    # Call the transcribe function
    try:
        result = transcribe(audio_path=audio_path, verbose=args.verbose, **kwargs)

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
