# PyHearingAI Utility Scripts

This directory contains utility scripts that support the PyHearingAI project but are not part of the core package. These scripts are intended for development, testing, and maintenance tasks.

## Available Scripts

### main.py
A utility script that provides command-line functionality for audio processing. This script offers direct access to core functionality without going through the full package structure.

### create_test_audio.py
Generates test audio files with specified characteristics for testing and development purposes. This is useful for creating controlled test data with known properties.

## Usage

Scripts can be run directly from the project root:

```bash
# Run the main utility script
python scripts/main.py

# Generate test audio
python scripts/create_test_audio.py
```

## Contributing

When adding new utility scripts:

1. Place them in this directory
2. Update this README with a brief description of the script
3. Include proper documentation within the script
4. Make sure the script follows the project's coding standards 