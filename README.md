# PyHearingAI

[![GitHub](https://img.shields.io/badge/GitHub-PyHearingAI-blue?logo=github)](https://github.com/MDGrey33/PyHearingAI)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The official library for transcribing audio conversations with accurate speaker identification.

## Current Status

PyHearingAI follows Clean Architecture principles with a well-organized code structure. The library provides a complete pipeline for audio transcription with speaker diarization and supports multiple output formats.

## Features

- Audio format conversion (supports mp3, wav, mp4, and more)
- Transcription pipeline powered by OpenAI Whisper
- Speaker diarization using Pyannote
- Speaker assignment using GPT-4o
- Support for multiple output formats:
  - TXT
  - JSON
  - SRT
  - VTT
  - Markdown
- Clean Architecture design for maintainability and extensibility
- End-to-end testing framework
- Progress tracking for long-running processes
- Comprehensive error handling
- Command-line interface

## Requirements

- Python 3.8+
- FFmpeg for audio conversion
- API keys:
  - OpenAI API key (for Whisper transcription and GPT-4o speaker assignment)
  - Hugging Face API key (for Pyannote speaker diarization)

## Installation

### System Dependencies

First, install FFmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg
```

### Using Poetry (Recommended)
```bash
poetry add pyhearingai
```

### Using pip
```bash
pip install pyhearingai
```

## API Key Setup

Set up your API keys as environment variables:

```bash
# In your terminal or .env file
export OPENAI_API_KEY=your_openai_api_key
export HUGGINGFACE_API_KEY=your_huggingface_api_key
```

Or in your Python code:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["HUGGINGFACE_API_KEY"] = "your_huggingface_api_key"
```

## Quick Start

```python
# Simple one-line usage
from pyhearingai import transcribe

# Process an audio file with default settings
result = transcribe("meeting.mp3")

# Print the full transcript with speaker labels
print(result.text)

# Save in different formats
result.save("transcript.txt")  # Plain text
result.save("transcript.json")  # JSON with segments, timestamps
result.save("transcript.srt")   # Subtitle format
result.save("transcript.md")    # Markdown format
```

## Advanced Usage

### Configuring the Transcription Process

```python
from pyhearingai import transcribe

# Configure transcription with specific options
result = transcribe(
    "interview.mp3",
    transcriber="whisper_openai",  # Specify transcriber
    diarizer="pyannote",           # Specify diarizer
    verbose=True                   # Enable verbose output
)
```

### Progress Tracking

```python
def progress_callback(progress_info):
    stage = progress_info.get('stage', 'unknown')
    percent = progress_info.get('progress', 0) * 100
    print(f"Processing {stage}: {percent:.1f}% complete")

result = transcribe(
    "long_recording.mp3",
    progress_callback=progress_callback
)
```

### Working with Results

```python
# Access the segments
for segment in result.segments:
    print(f"Speaker {segment.speaker_id}: {segment.text}")
    print(f"Time: {segment.start:.2f}s - {segment.end:.2f}s")

# Available output formats
from pyhearingai import list_output_formatters, get_output_formatter

# List available formatters
formatters = list_output_formatters()  # ['txt', 'json', 'srt', 'vtt', 'md']

# Get a specific formatter and format output
json_formatter = get_output_formatter('json')
json_content = json_formatter.format(result)
with open("transcript.json", "w") as f:
    f.write(json_content)
```

## Command Line Interface

PyHearingAI includes a command-line interface:

```bash
# Basic usage
transcribe meeting.mp3

# Specify output format
transcribe meeting.mp3 --output transcript.txt

# Configure models
transcribe meeting.mp3 --transcriber whisper-openai --diarizer pyannote --speaker-assigner gpt-4o

# Get help
transcribe --help
```

## Testing

The library includes an end-to-end test that validates the complete pipeline:

```bash
# Install test dependencies
pip install -r requirements_test.txt

# Run the end-to-end test
python -m pytest tests/test_end_to_end.py -v
```

## Repository

PyHearingAI is hosted on GitHub:
- [https://github.com/MDGrey33/PyHearingAI](https://github.com/MDGrey33/PyHearingAI)

## Architecture

PyHearingAI follows Clean Architecture principles, with clear separation of concerns:

- **Core (Domain Layer)**: Contains domain models and business rules
- **Application Layer**: Implements use cases like transcription and speaker assignment
- **Infrastructure Layer**: Provides concrete implementations of interfaces (OpenAI Whisper, Pyannote, GPT-4o)
- **Presentation Layer**: Offers user interfaces (CLI, future REST API)

For more details on the solution design and architecture, see the documentation:
- [Core Architecture](docs/architecture/01_core_architecture.md)
- [Solution Design](docs/architecture/02_solution_design.md)

## Extending PyHearingAI

The library is designed for extensibility:

### Custom Transcriber

```python
from pyhearingai.extensions import register_transcriber
from pyhearingai.models import Transcriber

@register_transcriber("my-transcriber")
class MyTranscriber(Transcriber):
    def transcribe(self, audio_path, **kwargs):
        # Custom transcription logic
        return segments
```

### Custom Diarizer

```python
from pyhearingai.extensions import register_diarizer
from pyhearingai.models import Diarizer

@register_diarizer("my-diarizer")
class MyDiarizer(Diarizer):
    def diarize(self, audio_path, **kwargs):
        # Custom diarization logic
        return speaker_segments
```

### Custom Speaker Assigner

```python
from pyhearingai.extensions import register_speaker_assigner
from pyhearingai.models import SpeakerAssigner

@register_speaker_assigner("my-assigner")
class MySpeakerAssigner(SpeakerAssigner):
    def assign_speakers(self, transcript_segments, diarization_segments, **kwargs):
        # Custom speaker assignment logic
        return labeled_segments
```

### Custom Output Format

```python
from pyhearingai.extensions import register_output_formatter
from pyhearingai.models import OutputFormatter

@register_output_formatter("my-format")
class MyOutputFormatter(OutputFormatter):
    def format(self, result):
        # Custom formatting logic
        return formatted_output
```

## Logging

Configure logging to control verbosity:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Set specific logger levels
logging.getLogger('pyhearingai.transcription').setLevel(logging.DEBUG)
logging.getLogger('pyhearingai.diarization').setLevel(logging.WARNING)
```

## Directory Structure

The library creates the following directory structure for outputs:

```
content/
├── audio_conversion/    # Converted audio files
├── transcription/       # Transcription results
├── diarization/         # Speaker diarization results
└── speaker_assignment/  # Final output with speaker labels
```

## Privacy and Data Handling

When using PyHearingAI, be aware that:

- Audio data is sent to third-party APIs (OpenAI and Hugging Face)
- OpenAI's data usage policies apply to audio sent for transcription
- Hugging Face's data usage policies apply to audio sent for diarization
- Consider data processing agreements when processing sensitive information

## API Rate Limits and Quotas

Users should be aware of:
- OpenAI has rate limits for the Whisper API (requests per minute)
- GPT-4o has token limits per request and rate limits
- Hugging Face API may have usage quotas

## Environment Variables

Required environment variables:
```
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

Optional environment variables:
```
PYHEARINGAI_DEFAULT_TRANSCRIBER=whisper-openai
PYHEARINGAI_DEFAULT_DIARIZER=pyannote
PYHEARINGAI_DEFAULT_SPEAKER_ASSIGNER=gpt-4o
PYHEARINGAI_OUTPUT_DIR=./content
PYHEARINGAI_LOG_LEVEL=INFO
```

## License

[Apache 2.0](LICENSE)

## Implemented Features

- Multiple output formats (TXT, JSON, SRT, VTT, Markdown)
- Transcription models:
  - OpenAI Whisper API (default)
- Diarization models:
  - Pyannote
- Speaker assignment models:
  - GPT-4o (using OpenAI API)

## Features Under Development

- 🎛️ **Extended Model Support**:
  - Local Whisper models
  - Faster Whisper
  - Additional diarization models

- 🚀 **Performance Features**:
  - GPU Acceleration
  - Batch processing
  - Memory optimization

## Contributing

We welcome contributions! Please check our [GitHub repository](https://github.com/MDGrey33/PyHearingAI) for guidelines.

## Acknowledgments

- OpenAI for the Whisper and GPT models
- Pyannote for the diarization technology
- The open-source community for various contributions
