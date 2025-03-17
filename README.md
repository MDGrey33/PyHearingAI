# PyHearingAI

## Project Overview

PyHearingAI is a system for processing audio files with speaker diarization and transcription capabilities, designed specifically for handling long multilingual recordings with resilience to interruptions. The system employs an idempotent processing approach, enabling pause and resume functionality with no data loss.

## Documentation Guide

This guide provides navigation to all project documentation.

### Key Documents

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [README.md](README.md) (this file) | Central documentation index | Start here for all documentation needs |
| [DESIGN.md](docs/project/DESIGN.md) | Architectural design and implementation details | When you need to understand how the system works |
| [TEST_PLAN.md](docs/project/TEST_PLAN.md) | Testing strategy and coverage improvement plan | When working on testing or quality assurance |
| [TODO.md](docs/project/TODO.md) | Implementation plan and progress tracking | When checking project status or planning work |
| [COMPLIANCE_ROADMAP.md](docs/project/COMPLIANCE_ROADMAP.md) | Compliance information and roadmap | When addressing compliance requirements |

## Finding Specific Information

### Architecture & Design Information

| Topic | Location | Description |
|-------|----------|-------------|
| Architecture Overview | [DESIGN.md#architecture-overview](DESIGN.md#architecture-overview) | Layers and components of the system |
| Processing Workflow | [DESIGN.md#processing-workflow-design](DESIGN.md#processing-workflow-design) | The four-stage processing pipeline |
| Domain Entities | [DESIGN.md#domain-entities](DESIGN.md#domain-entities) | Core business entities and their structure |
| Design Patterns | [DESIGN.md#design-patterns-used](DESIGN.md#design-patterns-used) | Patterns used throughout the system |
| Idempotent Processing | [DESIGN.md#idempotent-processing-design](DESIGN.md#idempotent-processing-design) | Details on resumable processing implementation |
| Configuration System | [DESIGN.md#configuration-system](DESIGN.md#configuration-system) | How configuration is managed |
| Persistence Strategy | [DESIGN.md#persistence-strategy](DESIGN.md#persistence-strategy) | How data is stored and managed |
| CLI Enhancements | [DESIGN.md#cli-enhancements](DESIGN.md#cli-enhancements) | Command-line interface details |
| Future Extensions | [DESIGN.md#future-extensibility](DESIGN.md#future-extensibility) | Planned future enhancements |
| Design Lessons Learned | [DESIGN.md#lessons-learned](DESIGN.md#lessons-learned) | Key insights from development |

### Testing Information

| Topic | Location | Description |
|-------|----------|-------------|
| Current Test Coverage | [docs/project/TEST_PLAN.md#current-test-coverage-status](docs/project/TEST_PLAN.md#current-test-coverage-status) | Current test metrics and targets |
| Testing Strategy | [docs/project/TEST_PLAN.md#complete-testing-strategy](docs/project/TEST_PLAN.md#complete-testing-strategy) | Overall testing approach |
| Test Coverage Strategy | [docs/project/TEST_PLAN.md#test-coverage-strategy](docs/project/TEST_PLAN.md#test-coverage-strategy) | How to improve test coverage |
| Priority Components | [docs/project/TEST_PLAN.md#priority-components-for-testing](docs/project/TEST_PLAN.md#priority-components-for-testing) | Which components need testing most |
| Worker Component Tests | [docs/project/TEST_PLAN.md#1-worker-components-high-priority](docs/project/TEST_PLAN.md#1-worker-components-high-priority) | Testing the worker components |
| Audio Processing Tests | [docs/project/TEST_PLAN.md#2-audio-processing-components-high-priority](docs/project/TEST_PLAN.md#2-audio-processing-components-high-priority) | Testing audio processing functionality |
| Core Services Tests | [docs/project/TEST_PLAN.md#3-core-services-high-priority](docs/project/TEST_PLAN.md#3-core-services-high-priority) | Testing the core services |
| Mocking Strategy | [docs/project/TEST_PLAN.md#mocking-strategy](docs/project/TEST_PLAN.md#mocking-strategy) | How to mock external dependencies |
| Test Data Management | [docs/project/TEST_PLAN.md#test-data-management](docs/project/TEST_PLAN.md#test-data-management) | Managing test data and fixtures |
| Implementation Timeline | [docs/project/TEST_PLAN.md#test-implementation-timeline](docs/project/TEST_PLAN.md#test-implementation-timeline) | Timeline for improving test coverage |
| Test Quality Practices | [docs/project/TEST_PLAN.md#test-quality-best-practices](docs/project/TEST_PLAN.md#test-quality-best-practices) | Best practices for writing tests |
| Success Criteria | [docs/project/TEST_PLAN.md#success-criteria](docs/project/TEST_PLAN.md#success-criteria) | Criteria for successful testing |

### Project Status & Implementation

| Topic | Location | Description |
|-------|----------|-------------|
| Sprint 1 Status | [TODO.md#sprint-1-core-domain-and-audio-processing-1-2-weeks---completed](TODO.md#sprint-1-core-domain-and-audio-processing-1-2-weeks---completed) | Status of core domain implementation |
| Sprint 2 Status | [TODO.md#sprint-2-diarization-and-transcription-1-2-weeks---completed](TODO.md#sprint-2-diarization-and-transcription-1-2-weeks---completed) | Status of diarization & transcription |
| Sprint 3 Status | [TODO.md#sprint-3-transcription-and-reconciliation-1-2-weeks---completed](TODO.md#sprint-3-transcription-and-reconciliation-1-2-weeks---completed) | Status of reconciliation service |
| Sprint 4 Status | [TODO.md#sprint-4-cli-and-orchestration-1-2-weeks---completed](TODO.md#sprint-4-cli-and-orchestration-1-2-weeks---completed) | Status of CLI & orchestration |
| Sprint 5 Status | [TODO.md#sprint-5-finalization-1-2-weeks---in-progress](TODO.md#sprint-5-finalization-1-2-weeks---in-progress) | Current sprint status |
| Test Improvements | [TODO.md#testing-and-quality-improvements---high-priority](TODO.md#testing-and-quality-improvements---high-priority) | Test improvement tasks |
| Performance Tasks | [TODO.md#performance-optimization---high-priority](TODO.md#performance-optimization---high-priority) | Performance improvement tasks |
| Config System Tasks | [TODO.md#configuration-system-enhancement---medium-priority](TODO.md#configuration-system-enhancement---medium-priority) | Configuration system tasks |
| UX Improvement Tasks | [TODO.md#user-experience-improvements---medium-priority](TODO.md#user-experience-improvements---medium-priority) | User experience tasks |
| Documentation Tasks | [TODO.md#documentation---normal-priority](TODO.md#documentation---normal-priority) | Documentation tasks |
| Lessons Learned | [TODO.md#lessons-learned-and-design-updates](TODO.md#lessons-learned-and-design-updates) | Implementation lessons learned |
| Next Steps | [TODO.md#implementation-priority-and-next-steps](TODO.md#implementation-priority-and-next-steps) | Prioritized next implementation steps |
| Timeline | [TODO.md#estimated-timeline](TODO.md#estimated-timeline) | Timeline for remaining work |

## Key Implementation Status

- ✅ **COMPLETED**: Core domain entities, repositories, audio processing, diarization
- ✅ **COMPLETED**: Transcription service, reconciliation service, CLI enhancements
- ✅ **COMPLETED**: Orchestration layer, progress tracking, initial end-to-end testing
- 🔄 **IN PROGRESS**: Testing improvements, performance optimization
- ⏰ **PENDING**: Configuration system enhancement, UX improvements, documentation

## Getting Started

1. **For Developers**: Start with [DESIGN.md](DESIGN.md) for architecture and implementation details
2. **For Testers**: Begin with [TEST_PLAN.md](TEST_PLAN.md) for the testing strategy
3. **For Project Managers**: Check [TODO.md](TODO.md) for implementation status and next steps

## Current Focus Areas

Based on our [TODO.md](TODO.md), the current focus areas are:

1. 🔴 **HIGH PRIORITY**: Increasing test coverage to meet the 89.5% threshold
2. 🔴 **HIGH PRIORITY**: Optimizing performance for large audio files
3. 🟠 **MEDIUM PRIORITY**: Enhancing the configuration system
4. 🟠 **MEDIUM PRIORITY**: Improving user experience
5. 🟡 **NORMAL PRIORITY**: Completing project documentation

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
