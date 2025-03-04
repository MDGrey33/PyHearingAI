# PyHearingAI Testing Infrastructure

This directory contains the testing infrastructure for PyHearingAI.

## Test Organization

We organize tests in multiple layers:

- **End-to-End Tests**: Located in the root `tests/` directory
  - `test_end_to_end.py` - Tests the complete pipeline from audio to labeled transcript
  - These tests are kept separate from other test infrastructure changes

- **Unit Tests**: Located in `tests/unit/`
  - Small, focused tests for individual components
  - Mock external dependencies
  - Fast execution

- **Integration Tests**: Located in `tests/integration/`
  - Test interaction between multiple components
  - May use real external dependencies for some tests
  - Validate component integration

- **Fixtures & Utilities**:
  - `fixtures/` - Contains test data files
  - `conftest.py` - Contains pytest fixtures and test utilities

## Directory Structure

- `fixtures/` - Contains test data files
  - `example_audio.m4a` - Sample audio file for testing
  - `labeled_transcript.txt` - Reference transcript for validation

- `unit/` - Unit tests for individual components
  - `test_audio_converter.py` - Tests for audio conversion
  - `test_transcriber.py` - Tests for transcription services
  - etc.

- `integration/` - Integration tests between components

## Running Tests

To run specific test categories:

```bash
# Run all tests
python -m pytest

# Run only unit tests
python -m pytest tests/unit/

# Run only integration tests
python -m pytest tests/integration/

# Run only the end-to-end test
python -m pytest tests/test_end_to_end.py
```

## Test Development Guidelines

1. **Separation of Concerns**:
   - E2E tests validate overall functionality
   - Unit tests focus on individual components
   - Keep these separate to allow independent development

2. **Fixtures**:
   - Use session-scoped fixtures for expensive resources
   - Use function-scoped fixtures for isolated test states

3. **Mocking**:
   - Mock external APIs in unit tests
   - Use dependency injection for easier mocking
   - Consider partial mocking for integration tests

4. **Test Independence**:
   - Tests should not depend on the state from other tests
   - Each test should set up its own state

5. **Compliance**:
   - We follow a phased approach to test compliance (see COMPLIANCE_ROADMAP.md)
   - Current coverage requirement: 89.5%
   - Target coverage: 89.5% (Achieved!)

## End-to-End Test

The primary test is an end-to-end validation that:

1. Processes the example audio through the entire pipeline
2. Compares the output transcript to the reference transcript
3. Validates at least 80% similarity between the generated and reference transcripts
This ensures that changes to the codebase maintain the overall quality and accuracy of the transcription process.
