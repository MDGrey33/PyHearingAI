# PyHearingAI Command Line Interface

This document explains how to use the PyHearingAI command line interface (CLI) for audio transcription and speaker diarization.

## Basic Usage

The basic command to transcribe an audio file with speaker diarization is:

```bash
python -m pyhearingai <audio_file>
```

For example:

```bash
python -m pyhearingai recording.mp3
```

This will transcribe the audio file and save the output to a text file with the same name (e.g., `recording.txt`).

## Output Formats

PyHearingAI supports multiple output formats:

```bash
# Default text format (speaker-labeled transcript)
python -m pyhearingai recording.mp3 -f txt

# Structured JSON with all metadata
python -m pyhearingai recording.mp3 -f json

# Subtitle formats
python -m pyhearingai recording.mp3 -f srt
python -m pyhearingai recording.mp3 -f vtt

# Markdown format
python -m pyhearingai recording.mp3 -f md
```

## Output Destination

You can specify a custom output file:

```bash
python -m pyhearingai recording.mp3 -o transcript.txt
```

## Processing Control

### Chunk Size

Control how the audio is divided for processing:

```bash
# Use smaller chunks (5 seconds each)
python -m pyhearingai recording.mp3 --chunk-size 5.0

# Use larger chunks (5 minutes each)
python -m pyhearingai recording.mp3 --chunk-size 300.0
```

Smaller chunks use less memory but may result in more processing overhead. Larger chunks are more efficient but require more memory.

### Parallel Processing

Control how many tasks run in parallel:

```bash
# Use a single worker (serial processing)
python -m pyhearingai recording.mp3 --max-workers 1

# Use 4 worker processes
python -m pyhearingai recording.mp3 --max-workers 4
```

More workers can speed up processing but use more system resources.

### Time Range Processing

Process only a specific portion of an audio file:

```bash
# Process only the first 3 minutes (180 seconds)
python -m pyhearingai recording.mp3 --end-time 180

# Process from 5 minutes to 10 minutes
python -m pyhearingai recording.mp3 --start-time 300 --end-time 600

# Process a 2-minute sample from the start
python -m pyhearingai recording.mp3 --sample-duration 120
```

This is useful for testing transcription on a small portion of a large file before committing to process the entire file.

## Memory Management

### Environment Variable

Set a global memory limit before running the CLI:

```bash
# Set a 4GB memory limit
export PYHEARINGAI_MEMORY_LIMIT=4096
python -m pyhearingai recording.mp3
```

### Indirect Memory Control

Use these options together to manage memory usage:

```bash
# Memory-efficient processing for large files
export PYHEARINGAI_MEMORY_LIMIT=2048
python -m pyhearingai large_recording.mp3 --chunk-size 30.0 --max-workers 2
```

## Progress Display

Show detailed progress during processing:

```bash
# Show chunk processing details
python -m pyhearingai recording.mp3 --show-chunks

# Show verbose output
python -m pyhearingai recording.mp3 --verbose
```

## Managing Jobs

PyHearingAI uses a job-based system for resumable processing:

```bash
# List all jobs
python -m pyhearingai --list-jobs

# Resume a previously interrupted job
python -m pyhearingai --resume JOB_ID

# Force a new job even if one exists for this file
python -m pyhearingai recording.mp3 --force
```

## API Keys

Provide API keys for cloud services:

```bash
# Set OpenAI API key
python -m pyhearingai recording.mp3 --openai-key YOUR_API_KEY

# Set Hugging Face API key
python -m pyhearingai recording.mp3 --huggingface-key YOUR_API_KEY
```

It's better to set these as environment variables in production:

```bash
export OPENAI_API_KEY=your_key_here
export HUGGINGFACE_API_KEY=your_key_here
```

## Advanced Options

### Cache Directory

Specify a custom directory for caching:

```bash
python -m pyhearingai recording.mp3 --cache-dir /path/to/cache
```

### Legacy Mode

Use the non-idempotent processing mode:

```bash
python -m pyhearingai recording.mp3 --use-legacy
```

This disables job persistence but may be faster for one-time processing.

## Complete Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `audio_file` | Path to the audio file to transcribe | (Required) |
| `-s, --source` | Alternative way to specify audio file | |
| `-o, --output` | Output file path | Same as input with format extension |
| `-f, --format` | Output format (txt, json, srt, vtt, md) | txt |
| `--show-chunks` | Show chunk processing progress | False |
| `--max-workers` | Maximum number of parallel workers | Auto |
| `--chunk-size` | Size of audio chunks in seconds | 10.0 |
| `--cache-dir` | Directory for caching results | System temp dir |
| `--use-legacy` | Use non-idempotent processing mode | False |
| `--start-time` | Start time in seconds to process a portion of audio | 0.0 |
| `--end-time` | End time in seconds to process a portion of audio | Full duration |
| `--sample-duration` | Process only a sample of specified duration | None |
| `--openai-key` | OpenAI API key | From env var |
| `--huggingface-key` | Hugging Face API key | From env var |
| `--verbose` | Enable verbose output | False |
| `--resume` | Resume processing with job ID | |
| `--list-jobs` | List all processing jobs | |
| `--force` | Force new job creation | False |
| `--version` | Show version number | |
| `-h, --help` | Show help message | |

## Examples

```bash
# Basic transcription with default settings
python -m pyhearingai recording.mp3

# Transcribe with specific format and output path
python -m pyhearingai recording.mp3 -f json -o analysis.json

# Memory-efficient processing with progress display
export PYHEARINGAI_MEMORY_LIMIT=1024
python -m pyhearingai long_recording.mp3 --chunk-size 30.0 --max-workers 2 --verbose --show-chunks

# Process only a 3-minute sample from a large file
python -m pyhearingai long_interview.mp3 --sample-duration 180 --verbose

# Resume a previously interrupted job
python -m pyhearingai --resume 550e8400-e29b-41d4-a716-446655440000
``` 