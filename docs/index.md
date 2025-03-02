# PyHearingAI Documentation

Welcome to the PyHearingAI documentation. This library provides a simple yet powerful interface for audio transcription with speaker diarization and assignment.

## Documentation Sections

### Architecture
- [Core Architecture](architecture/01_core_architecture.md) - Overview of the system architecture and components
- [Solution Design](architecture/02_solution_design.md) - Detailed design of the solution
- [Migration Plan](MIGRATION_PLAN.md) - Plan for migrating to Clean Architecture

### Guides
- [Getting Started](#quick-start) - How to get started with PyHearingAI
- [Installation Guide](#installation) - How to install PyHearingAI
- [Basic Usage](#basic-usage) - Basic usage examples
- [Advanced Configuration](#advanced-usage) - Advanced configuration options

### API Reference
- [Core API](#core-api) - Core API documentation
- [Models Reference](#models) - Reference for domain models
- [Configuration Options](#configuration) - Available configuration options

### Examples
- [Basic Examples](#basic-examples) - Basic usage examples
- [Advanced Usage](#advanced-examples) - Advanced usage examples
- [Extension Examples](#extension-examples) - Examples of extending PyHearingAI

## Quick Start

Install the library:

```bash
pip install pyhearingai
```

Set up your API keys:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["HUGGINGFACE_API_KEY"] = "your_huggingface_api_key"
```

Transcribe an audio file:

```python
from pyhearingai import transcribe

result = transcribe("meeting.mp3")
print(result.text)  # Print the transcript with speaker labels
result.save("transcript.txt")  # Save as text file
```

## Installation

### Requirements

- Python 3.8+
- FFmpeg
- API Keys:
  - OpenAI API key (for Whisper transcription and GPT-4o speaker assignment)
  - Hugging Face API key (for Pyannote speaker diarization)

### Install with pip

```bash
pip install pyhearingai
```

### Install with Poetry

```bash
poetry add pyhearingai
```

## Basic Usage

### Transcribe an Audio File

```python
from pyhearingai import transcribe

# Simple transcription
result = transcribe("meeting.mp3")

# Print the transcript
print(result.text)

# Save in different formats
result.save("transcript.txt")   # Text format
result.save("transcript.json")  # JSON format
result.save("transcript.srt")   # SubRip subtitle format
```

## Advanced Usage

### Configure Models

```python
from pyhearingai import transcribe
from pyhearingai.models import TranscriptionConfig

config = TranscriptionConfig(
    transcriber="whisper-openai",
    diarizer="pyannote",
    speaker_assigner="gpt-4o",
    output_format="json",
    language="en"
)

result = transcribe("interview.mp3", config=config)
```

### Process Multiple Files

```python
from pyhearingai import pipeline_session

with pipeline_session(config) as session:
    result1 = session.transcribe("file1.mp3")
    result2 = session.transcribe("file2.mp3")
```

### Track Progress

```python
def on_progress(info):
    print(f"{info['stage']}: {info['progress']:.0%}")

result = transcribe("long_recording.mp3", progress_callback=on_progress)
```

## Models

PyHearingAI uses several types of models:

### Transcription Models

- `whisper-openai`: Uses OpenAI's Whisper API for transcription

### Diarization Models

- `pyannote`: Uses Pyannote.audio for speaker diarization

### Speaker Assignment Models

- `gpt-4o`: Uses OpenAI's GPT-4o for intelligent speaker assignment

## Extension Examples

### Custom Transcriber

```python
from pyhearingai.extensions import register_transcriber
from pyhearingai.models import Transcriber

@register_transcriber("my-transcriber")
class MyTranscriber(Transcriber):
    def transcribe(self, audio_path, **kwargs):
        # Custom implementation
        return segments
```

### Custom Speaker Assigner

```python
from pyhearingai.extensions import register_speaker_assigner
from pyhearingai.models import SpeakerAssigner

@register_speaker_assigner("my-assigner")
class MySpeakerAssigner(SpeakerAssigner):
    def assign_speakers(self, transcript_segments, diarization_segments, **kwargs):
        # Custom implementation
        return labeled_segments
```

## Quick Links

- [GitHub Repository](https://github.com/MDGrey33/PyHearingAI)
- [Issue Tracker](https://github.com/MDGrey33/PyHearingAI/issues)

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/MDGrey33/PyHearingAI/blob/main/CONTRIBUTING.md) for details.
