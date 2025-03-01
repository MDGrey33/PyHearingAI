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

## Requirements

- Python 3.8+
- FFmpeg for audio conversion
- API keys:
  - OpenAI API key (for Whisper transcription and GPT-4o speaker assignment)
  - Hugging Face API key (for Pyannote speaker diarization)

## Installation

### Using Poetry (Recommended)
```bash
poetry add pyhearingai
```

### Using pip
```bash
pip install pyhearingai
```

## Quick Start

```python
# Simple one-line usage
from pyhearingai import transcribe

# Process an audio file with default settings
result = transcribe("meeting.mp3")

# Process with specific models
result = transcribe(
    audio_path="meeting.mp3",
    transcriber="whisper_openai",
    diarizer="pyannote",
    verbose=True
)

# Access the segments
for segment in result.segments:
    print(f"Speaker {segment.speaker_id}: {segment.text}")
    
# Available output formats
from pyhearingai import list_output_formatters, get_output_formatter

# List available formatters
formatters = list_output_formatters()  # ['txt', 'json', 'srt', 'vtt', 'md']

# Get a specific formatter and save output
json_formatter = get_output_formatter('json')
json_formatter.save(result, "transcript.json")
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

## Environment Variables

Required environment variables:
```
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
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

- OpenAI for the Whisper model
- Pyannote for the diarization technology
- The open-source community for various contributions
