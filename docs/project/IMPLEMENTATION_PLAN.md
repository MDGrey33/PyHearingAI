# PyHearingAI Implementation Plan

**Date:** 2023-03-19
**Updated:** 2023-03-21

## Overview

This document outlines the revised implementation plan for addressing issues identified during the testing phase of PyHearingAI. Instead of trying to fix failing tests, we are taking a more comprehensive approach by reimplementing the test suite from scratch, following clean architecture principles and proper testing patterns.

## Priority 1: Test Framework and Structure

### 1.1 Create New Test Directory Structure (Estimated effort: 1 day)

**Implementation:**

Create a new, more organized directory structure for tests:

```
tests/
  unit/          # Tests for individual components in isolation
    domain/      # Domain model tests
    application/ # Application service tests
    infrastructure/ # Infrastructure adapter tests
  integration/   # Tests for component interactions
  functional/    # End-to-end tests for user features
  fixtures/      # Shared test fixtures and helpers
  utils/         # Testing utilities
```

**Expected outcome:** Clear organization for all tests, making it easier to maintain and extend.

### 1.2 Implement Core Test Fixtures (Estimated effort: 2 days)

**Implementation:**

Create essential fixtures needed by multiple tests:

```python
# tests/fixtures/audio_fixtures.py
@pytest.fixture
def create_test_audio(tmp_path):
    """Create a test audio file with configurable parameters.

    Args:
        duration: Duration in seconds (default: 1.0)
        sample_rate: Sample rate in Hz (default: 16000)
        frequency: Tone frequency in Hz (default: 440)
        amplitude: Amplitude of the tone (default: 0.1)

    Returns:
        Path: Path to the created test audio file
    """
    def _create(duration=1.0, sample_rate=16000, frequency=440, amplitude=0.1,
                filename="test_audio.wav"):
        # Create audio with specified parameters
        samples = np.zeros(int(sample_rate * duration), dtype=np.float32)

        # Add a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        samples += amplitude * np.sin(2 * np.pi * frequency * t)

        audio_path = tmp_path / filename
        sf.write(audio_path, samples, sample_rate)

        return audio_path

    return _create

# tests/fixtures/mock_services.py
@pytest.fixture
def mock_transcription_service():
    """Create a mock transcription service."""
    service = MagicMock()
    service.transcribe.return_value = [
        {"start": 0.0, "end": 2.0, "text": "This is a test transcript."}
    ]
    return service

@pytest.fixture
def mock_diarization_service():
    """Create a mock diarization service."""
    service = MagicMock()
    service.diarize.return_value = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}
    ]
    return service
```

**Expected outcome:** Reusable fixtures that simplify test creation and maintenance.

## Priority 2: Domain and Core Tests

### 2.1 Implement Domain Model Tests (Estimated effort: 3 days)

**Implementation:**

Create tests for core domain models and business rules:

```python
# tests/unit/domain/test_models.py
def test_segment_creation():
    """Test creation and properties of Segment class."""
    segment = Segment(start=1.0, end=2.5, text="Hello world")

    assert segment.start == 1.0
    assert segment.end == 2.5
    assert segment.text == "Hello world"
    assert segment.duration == 1.5

def test_segment_validation():
    """Test validation rules for Segment class."""
    # Start must be less than end
    with pytest.raises(ValueError):
        Segment(start=2.0, end=1.0, text="Invalid")

    # Text cannot be empty
    with pytest.raises(ValueError):
        Segment(start=0.0, end=1.0, text="")
```

**Expected outcome:** Comprehensive tests for domain models with 90%+ coverage.

### 2.2 Implement Core Business Rules Tests (Estimated effort: 2 days)

**Implementation:**

Test business rules and core logic:

```python
# tests/unit/domain/test_validation.py
def test_audio_format_validation():
    """Test audio format validation rules."""
    validator = AudioValidator()

    # Valid file should pass
    assert validator.is_valid(valid_audio_path) == True

    # Invalid file should fail with specific error
    result = validator.is_valid(invalid_audio_path)
    assert result == False
    assert validator.last_error == "Unsupported audio format"
```

**Expected outcome:** Tests that verify business rules are correctly enforced.

## Priority 3: Application Service Tests

### 3.1 Implement Service Tests (Estimated effort: 5 days)

**Implementation:**

Create tests for application services:

```python
# tests/unit/application/test_transcription_service.py
def test_transcription_service_basic(create_test_audio, mock_transcriber):
    """Test basic transcription service functionality."""
    # Arrange
    audio_path = create_test_audio()
    service = TranscriptionService(transcriber=mock_transcriber)

    # Act
    result = service.transcribe(audio_path)

    # Assert
    assert len(result) > 0
    assert "text" in result[0]
    assert "start" in result[0]
    assert "end" in result[0]
    mock_transcriber.transcribe.assert_called_once()
```

**Expected outcome:** Comprehensive tests for all application services.

### 3.2 Implement Orchestration Tests (Estimated effort: 3 days)

**Implementation:**

Test service orchestration and coordination:

```python
# tests/unit/application/test_orchestrator.py
def test_orchestrator_end_to_end(
    create_test_audio,
    mock_transcription_service,
    mock_diarization_service
):
    """Test end-to-end orchestration process."""
    # Arrange
    audio_path = create_test_audio()
    orchestrator = Orchestrator(
        transcription_service=mock_transcription_service,
        diarization_service=mock_diarization_service
    )

    # Act
    result = orchestrator.process(audio_path)

    # Assert
    assert result.status == "completed"
    assert len(result.segments) > 0
    assert "speaker" in result.segments[0]
    assert "text" in result.segments[0]
```

**Expected outcome:** Tests that verify proper service coordination.

## Priority 4: Infrastructure Tests

### 4.1 Implement Repository Tests (Estimated effort: 3 days)

**Implementation:**

Test repository implementations:

```python
# tests/unit/infrastructure/test_repositories.py
def test_job_repository_crud(tmp_path):
    """Test CRUD operations on job repository."""
    # Arrange
    repo = JsonJobRepository(base_path=tmp_path)
    job = ProcessingJob(audio_path="test.wav")

    # Act - Create
    repo.save(job)

    # Assert - Read
    retrieved = repo.get(job.id)
    assert retrieved.id == job.id
    assert retrieved.audio_path == job.audio_path

    # Act - Update
    job.status = "completed"
    repo.save(job)

    # Assert
    updated = repo.get(job.id)
    assert updated.status == "completed"

    # Act - Delete
    repo.delete(job.id)

    # Assert
    assert repo.get(job.id) is None
```

**Expected outcome:** Thorough tests for all repository implementations.

### 4.2 Implement Adapter Tests (Estimated effort: 4 days)

**Implementation:**

Test external service adapters:

```python
# tests/unit/infrastructure/test_transcriber_adapters.py
@pytest.mark.vcr  # Use pytest-vcr to record/replay API interactions
def test_whisper_adapter(create_test_audio):
    """Test WhisperTranscriber adapter with controlled API responses."""
    # Arrange
    audio_path = create_test_audio()
    transcriber = WhisperTranscriber(api_key="test_key")

    # Act
    result = transcriber.transcribe(audio_path)

    # Assert
    assert len(result) > 0
    assert all(isinstance(item, dict) for item in result)
    assert all("text" in item for item in result)
```

**Expected outcome:** Tests for all external service adapters with proper mocking.

## Timeline and Execution Plan

| Week | Priority Tasks | Expected Outcomes |
|------|----------------|-------------------|
| Week 1 | 1.1, 1.2, 2.1 | Set up test structure and implement domain tests |
| Week 2 | 2.2, 3.1 | Complete domain and begin application service tests |
| Week 3 | 3.1, 3.2 | Complete application service and orchestration tests |
| Week 4 | 4.1, 4.2 | Implement infrastructure and adapter tests |

## Success Criteria

The implementation will be considered successful when:

1. New test suite achieves 90%+ coverage across all modules
2. All tests are properly documented with clear assertions
3. Tests follow consistent patterns and use fixtures effectively
4. No tests make actual API calls (everything properly mocked)
5. Tests run quickly and reliably

## Simultaneous Production Code Fixes

While reimplementing tests, we'll address these critical production code issues:

1. Add missing `overlap_duration` parameter to `ChunkingServiceImpl.create_audio_chunks`
2. Fix `ProcessingJob` class to handle additional parameters correctly
3. Complete missing implementations in `PyannoteDiarizer`
4. Implement job cancellation and deletion features

## Monitoring and Reporting

Progress will be tracked through:

1. Daily test runs to measure coverage improvement
2. Weekly status reports showing test implementation progress
3. Documentation updates for the new testing architecture

---

Last updated: 2023-03-21
