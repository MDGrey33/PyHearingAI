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
- `speaker_assignment.py`: Speaker assignment algorithm implementation
- `transcribers/`: Directory containing transcriber implementations
  - `whisper_openai.py`: OpenAI Whisper API transcriber implementation
- `diarizers/`: Directory containing diarizer implementations
  - `pyannote.py`: Pyannote.audio diarizer implementation
- `formatters/`: Directory containing output formatter implementations
  - `text.py`: Text output formatter implementation

## Presentation Layer

The `presentation` directory contains interfaces to the outside world:

- `cli.py`: Command-line interface

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
3. **New Output Format**: Create a new class in `infrastructure/formatters/` that implements the `OutputFormatter` interface and decorates it with `@register_output_formatter` 