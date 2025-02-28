# PyHearingAI

[![GitHub](https://img.shields.io/badge/GitHub-PyHearingAI-blue?logo=github)](https://github.com/MDGrey33/PyHearingAI)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The official library for transcribing audio conversations with accurate speaker identification.

## Current Status

PyHearingAI is under active development. Documentation is evolving as features are implemented.

## Current Features

- Audio format conversion for multiple input formats
- Basic transcription pipeline for audio files
- Integration with OpenAI Whisper and Pyannote models
- Speaker diarization and assignment
- End-to-end testing framework

## Requirements

- Python 3.8+
- FFmpeg for audio conversion
- API keys:
  - OpenAI API key (for Whisper model access)
  - Hugging Face API key (for Pyannote model access)

## Features Under Development

- Simple installation via pip
- One-line API usage
- Flexible model selection
- Performance optimizations
- Multiple output formats
- Additional model support

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

## Architecture Documentation

For details on the solution design and architecture, see the documentation:
- [Core Architecture](docs/architecture/01_core_architecture.md)
- [Solution Design](docs/architecture/02_solution_design.md)
- [Migration Plan](docs/MIGRATION_PLAN.md)

## Environment Variables

Required environment variables:
```
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

## License

[Apache 2.0](LICENSE)

## Features Under Development

### Installation (Coming Soon)
```bash
pip install pyhearingai
```

### Planned API
```python
# Simple one-line usage (in development)
from pyhearingai import transcribe
transcribe("meeting.mp3")

# CLI interface (planned)
pyhearingai transcribe meeting.mp3
```

### Planned Features

- 🎛️ **Flexible Model Selection**: 
  - Choice between cloud and local models
  - Support for multiple transcription and diarization models
  - Custom model integration

- 🚀 **Performance Features**:
  - GPU Acceleration
  - Batch processing
  - Memory optimization

- 📝 **Output Formats**:
  - Markdown
  - JSON
  - TXT
  - SRT subtitles

### Planned Model Support

#### Transcription
- OpenAI Whisper API (default)
- Local Whisper
- Faster Whisper

#### Diarization
- Pyannote (default)
- Additional models to be added

#### Speaker Detection
- Whisper-based detection
- Custom models support

### Documentation

Full documentation is under development. For now, please refer to:
- This README
- Source code at [GitHub](https://github.com/MDGrey33/PyHearingAI)

## Contributing

We welcome contributions! Please check our [GitHub repository](https://github.com/MDGrey33/PyHearingAI) for guidelines.

## Acknowledgments

- OpenAI for the Whisper model
- Pyannote for the diarization technology
- The open-source community for various contributions
