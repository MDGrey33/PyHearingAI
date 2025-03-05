# PyHearingAI Package Structure

The PyHearingAI package follows Clean Architecture principles to separate concerns and promote maintainability, testability, and extensibility. The package is organized into the following layers:

## Core (Domain Layer)

The `core` directory contains the domain model and business rules:

- `models.py`: Contains domain entities (`Segment`, `DiarizationSegment`, `TranscriptionResult`)
- `ports.py`: Defines interfaces (ports) that adapter implementations must satisfy

The domain layer has no dependencies on external frameworks or libraries.

## Application Layer

The `application` directory contains use cases and orchestration logic:

- `transcribe.py`: Primary use case for transcribing audio with speaker diarization
- `outputs.py`: Functions for creating different output formats

The application layer depends only on the domain layer.

## Infrastructure Layer

The `infrastructure` directory contains concrete implementations of ports defined in the domain layer:

- `registry.py`: Service locator and dependency injection mechanism
- `adapters.py`: Initializes all adapter implementations
- `audio_converter.py`: Audio conversion implementation using FFmpeg
- `speaker_assignment.py`: Default speaker assignment algorithm implementation
- `speaker_assignment_gpt.py`: GPT-4o based speaker assignment implementation
- `transcribers/`: Directory containing transcriber implementations
  - `whisper_openai.py`: OpenAI Whisper API transcriber implementation
- `diarizers/`: Directory containing diarizer implementations
  - `pyannote.py`: Pyannote.audio diarizer implementation
- `formatters/`: Directory containing output formatter implementations
  - `text.py`: Text output formatter implementation
  - `json.py`: JSON output formatter implementation
  - `srt.py`: SRT subtitle output formatter implementation
  - `vtt.py`: VTT subtitle output formatter implementation
  - `markdown.py`: Markdown output formatter implementation

## Presentation Layer

The `presentation` directory contains interfaces to the outside world:

- `cli.py`: Command-line interface

## Command-Line Interface

PyHearingAI provides a simple command-line interface for transcribing audio files:

### Basic Usage

```bash
# Basic usage
transcribe meeting.mp3

# With source flag
transcribe -s meeting.mp3

# Specifying output file
transcribe meeting.mp3 -o transcript.txt

# Output in different formats
transcribe meeting.mp3 -f json
```

### API Keys

API keys can be provided either as environment variables or command-line arguments:

```bash
# Using environment variables (recommended)
export OPENAI_API_KEY="your_openai_api_key"
export HUGGINGFACE_API_KEY="your_huggingface_api_key"
transcribe meeting.mp3

# Or using command-line arguments
transcribe meeting.mp3 --openai-key "your_openai_api_key" --huggingface-key "your_huggingface_api_key"
```

### Supported Output Formats

PyHearingAI can output transcriptions in multiple formats:

```bash
# Standard text output (default)
transcribe meeting.mp3 -f txt

# JSON output with detailed information
transcribe meeting.mp3 -f json

# SRT subtitle format
transcribe meeting.mp3 -f srt

# VTT subtitle format
transcribe meeting.mp3 -f vtt

# Markdown format
transcribe meeting.mp3 -f md
```

### Verbose Mode

Enable verbose mode for detailed logging:

```bash
transcribe meeting.mp3 --verbose
```

## Architecture Benefits

This architecture provides several benefits:

1. **Separation of Concerns**: Each layer has a specific responsibility
2. **Dependency Rule**: Dependencies point inward (infrastructure → application → domain)
3. **Testability**: Core business logic can be tested without external dependencies
4. **Extensibility**: New adapters can be added without changing core code
5. **Framework Independence**: The domain and application logic don't depend on external frameworks

## Adding New Components

To add new functionality:

1. **New Transcriber**: Create a new class in `infrastructure/transcribers/` that implements the `Transcriber` interface and decorates it with `@register_transcriber`
2. **New Diarizer**: Create a new class in `infrastructure/diarizers/` that implements the `Diarizer` interface and decorates it with `@register_diarizer`
3. **New Speaker Assigner**: Create a new class that implements the `SpeakerAssigner` interface and decorates it with `@register_speaker_assigner`
4. **New Output Format**: Create a new class in `infrastructure/formatters/` that implements the `OutputFormatter` interface and decorates it with `@register_output_formatter`

## Error Handling

The package implements comprehensive error handling:

- `AudioProcessingError`: Raised when audio conversion fails
- `TranscriptionError`: Raised when transcription fails
- `DiarizationError`: Raised when speaker diarization fails
- `SpeakerAssignmentError`: Raised when speaker assignment fails

## Performance Considerations

For large files, the package provides chunked processing capabilities:

- `transcribe_chunked`: Function for processing large audio files in smaller chunks
- Progress tracking via callbacks

## Usage Example

```python
from pyhearingai import transcribe

# Basic usage
result = transcribe("meeting.mp3")

# Advanced usage with specific models
from pyhearingai.models import TranscriptionConfig

config = TranscriptionConfig(
    transcriber="whisper-openai",
    diarizer="pyannote",
    speaker_assigner="gpt-4o",
    output_format="json",
    language="en"
)
result = transcribe("interview.mp3", config=config)
