# PyHearingAI Tests

This directory contains tests for the PyHearingAI library.

## Directory Structure

- `fixtures/` - Contains test data files
  - `example_audio.m4a` - Sample audio file for testing
  - `labeled_transcript.txt` - Reference transcript for validation
  - `diarization_segments.json` - Sample diarization output for testing
  - `transcription_segments.json` - Sample transcription output for testing

## Running Tests

To run the tests, use the following command from the project root:

```bash
python -m pytest tests/
```

For more verbose output:

```bash
python -m pytest tests/ -v
```

## End-to-End Test

The primary test is an end-to-end validation that:

1. Processes the example audio through the entire pipeline
2. Compares the output transcript to the reference transcript
3. Validates at least 80% similarity between the generated and reference transcripts
This ensures that changes to the codebase maintain the overall quality and accuracy of the transcription process.
