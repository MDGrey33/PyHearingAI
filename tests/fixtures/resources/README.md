# Test Fixtures Resources

This directory contains various resource files needed for testing the PyHearingAI library.

## Contents

- Audio files for testing transcription and diarization
- Pre-created test outputs for comparison
- Sample configurations for test scenarios

## Usage

These resources are used by the fixture functions defined in `tests/fixtures/` and should be accessed through those fixtures rather than directly.

For example:

```python
def test_something(test_audio_file):
    # test_audio_file is a fixture that may use resources from this directory
    pass
```
