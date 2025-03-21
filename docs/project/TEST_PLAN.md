# PyHearingAI Test Plan

## Testing Strategy

This document outlines the testing strategy for the PyHearingAI project, following clean architecture principles and domain-driven design. The testing approach is organized by layer, ensuring that each component is tested in isolation with proper mocks, followed by integration tests to verify component interactions.

### Testing Pyramid

We follow a balanced testing pyramid:

1. **Unit Tests**: Largest number - test individual classes and functions in isolation
2. **Integration Tests**: Test interactions between components
3. **Functional Tests**: Test end-to-end user workflows

### Testing by Clean Architecture Layer

#### Domain Layer
- Focus on testing business rules and domain logic in isolation
- No dependencies on external systems or frameworks
- High test coverage (95%+)

**Key Components to Test:**
- Domain Entities (`ApiSizeLimit`, `AudioQualitySpecification`, etc.)
- Domain Services (`ApiSizeLimitPolicy`, `AudioValidationService`)
- Value Objects
- Domain Events

#### Application Layer
- Test service orchestration and use cases
- Mock external dependencies (infrastructure)
- Test all business flows and error cases
- Coverage target: 90%+

**Key Components to Test:**
- Service Implementations (`ChunkingServiceImpl`, etc.)
- Orchestrators
- Command/Query Handlers
- Event Handlers

#### Infrastructure Layer
- Test adapters and implementations of interfaces
- May require integration with external systems
- Mock external APIs where appropriate
- Coverage target: 85%+

**Key Components to Test:**
- Repository Implementations
- Adapters (`FFmpegAudioFormatService`, etc.)
- External API Clients
- Data Mappers/Transformers

#### Presentation Layer
- Test CLI functionality with mocked services
- Focus on user interaction patterns
- Coverage target: 80%+

**Key Components to Test:**
- CLI Commands
- Output Formatters
- User Input Validation

## Test Types and Tools

### Unit Tests
- Framework: pytest
- Mock Framework: pytest-mock
- Coverage Tool: pytest-cov
- Test Location: `tests/unit/`

### Integration Tests
- Framework: pytest
- Integration Points: Service interactions, repository data access
- Test Location: `tests/integration/`

### Functional Tests
- Framework: pytest
- Focus: End-to-end user workflows via CLI
- Test Location: `tests/functional/`

## Test Fixtures and Resources

- Audio Fixtures: Generate test audio files with configurable parameters
- Mock Providers: Simulate transcription and diarization services
- File Path Fixtures: Manage test file paths and cleanup

## Special Testing Considerations

### Audio Processing Tests
- Use small audio samples to keep tests fast
- Mock FFmpeg calls where appropriate
- Test various audio formats and qualities

### External API Integration
- Mock all external API calls in unit and most integration tests
- Provide configuration for optional "live" API tests

### Concurrency Testing
- Test worker pools and task processing
- Verify resource cleanup
- Test throttling mechanisms

## Test Implementation Phases

1. **Phase 1:** Core domain model unit tests (current)
2. **Phase 2:** Application service unit tests
3. **Phase 3:** Infrastructure adapter tests
4. **Phase 4:** Integration tests for key workflows
5. **Phase 5:** CLI functional tests

## Coverage Goals

- Overall project coverage: 89.5%
- Domain Layer: 95%+
- Application Layer: 90%+
- Infrastructure Layer: 85%+
- CLI/Presentation: 80%+

## Continuous Integration

- Run unit tests on every commit
- Run integration tests on PRs and main branch
- Run full test suite including functional tests nightly
- Generate coverage reports automatically

## Test Architecture

### Directory Structure

The new test structure will be organized as follows:

```
tests/
  unit/                  # Tests for components in isolation
    domain/              # Tests for domain models and business rules
    application/         # Tests for application services
    infrastructure/      # Tests for infrastructure adapters
  integration/           # Tests for component interactions
    repositories/        # Tests for repository implementations
    services/            # Tests for service integrations
  functional/            # End-to-end tests
  fixtures/              # Shared test fixtures
    audio/               # Audio file generation fixtures
    mocks/               # Mock service fixtures
  utils/                 # Test utilities and helpers
  conftest.py            # Global pytest configuration
```

### Naming Conventions

- Test files: `test_<component_name>.py`
- Test classes: `Test<ComponentName>`
- Test methods: `test_<functionality>_<condition>`
- Fixtures: Descriptive names that indicate what they provide

### Test Types

1. **Unit Tests**: Test individual components in isolation
   - Fast, focused, use mocks for dependencies
   - Test business rules, edge cases, error handling
   - Located in `tests/unit/`

2. **Integration Tests**: Test interactions between components
   - Test repository implementations with test databases
   - Test service dependencies and coordination
   - Located in `tests/integration/`

3. **Functional Tests**: Test end-to-end workflows
   - Test user-facing functionality
   - Minimal mocking, closer to real usage
   - Located in `tests/functional/`

## Test Implementation

### Phase 1: Test Framework Setup

#### 1.1 Directory Structure Creation

Create the directory structure as outlined above, with placeholder files and basic setup.

#### 1.2 Core Fixtures Implementation

Create essential fixtures that will be reused across tests:

1. **Audio Generation**:
   ```python
   @pytest.fixture
   def create_test_audio():
       """Create test audio files with configurable parameters."""
       def _create(duration=1.0, sample_rate=16000, channels=1, **kwargs):
           # Create and return audio file with specified parameters
           pass
       return _create
   ```

2. **Mock Services**:
   ```python
   @pytest.fixture
   def mock_transcription_service():
       """Create a mock transcription service."""
       service = MagicMock()
       service.transcribe.return_value = [
           {"text": "Test transcript", "start": 0.0, "end": 1.0}
       ]
       return service
   ```

3. **Test Repository**:
   ```python
   @pytest.fixture
   def test_repository(tmp_path):
       """Create a test repository with temp directory."""
       repo = JsonRepository(base_path=tmp_path)
       yield repo
       # Cleanup
   ```

#### 1.3 Test Utilities

Create helper utilities for common test operations:

```python
def assert_segments_equal(actual, expected, tolerance=0.1):
    """Assert that two segments are equal within a tolerance."""
    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        assert act["text"] == exp["text"]
        assert abs(act["start"] - exp["start"]) <= tolerance
        assert abs(act["end"] - exp["end"]) <= tolerance
```

### Phase 2: Domain Model Tests

#### 2.1 Domain Entity Tests

Test domain entities, value objects, and business rules:

```python
def test_segment_validation():
    """Test segment validation rules."""
    # Valid segment
    segment = Segment(start=0.0, end=1.0, text="Valid text")
    assert segment.start == 0.0
    assert segment.end == 1.0
    assert segment.text == "Valid text"

    # Invalid segment (start > end)
    with pytest.raises(ValueError):
        Segment(start=1.0, end=0.5, text="Invalid")
```

#### 2.2 Business Rule Tests

Test core business logic:

```python
def test_audio_quality_constraints():
    """Test audio quality constraints for different providers."""
    # Test constraints for OpenAI Whisper
    constraints = AudioQualityPolicy.get_constraints(Provider.OPENAI_WHISPER)
    assert constraints.max_file_size == 25 * 1024 * 1024  # 25 MB
    assert constraints.max_duration == 120  # 2 minutes
```

### Phase 3: Application Service Tests

#### 3.1 Service Operation Tests

Test application services with mocked dependencies:

```python
def test_transcription_service(create_test_audio, mock_provider):
    """Test basic transcription functionality."""
    # Arrange
    audio_path = create_test_audio()
    service = TranscriptionService(provider=mock_provider)

    # Act
    result = service.transcribe(audio_path)

    # Assert
    assert len(result) > 0
    assert "text" in result[0]
    assert mock_provider.transcribe.called_once_with(audio_path)
```

#### 3.2 Error Handling Tests

Test how services handle errors:

```python
def test_transcription_service_error_handling(create_test_audio, mock_provider):
    """Test error handling in transcription service."""
    # Arrange
    audio_path = create_test_audio()
    mock_provider.transcribe.side_effect = ProviderError("API error")
    service = TranscriptionService(provider=mock_provider)

    # Act & Assert
    with pytest.raises(TranscriptionError):
        service.transcribe(audio_path)
```

### Phase 4: Infrastructure Tests

#### 4.1 Repository Tests

Test repository implementations:

```python
def test_json_repository_crud(tmp_path):
    """Test CRUD operations on JSON repository."""
    # Arrange
    repo = JsonRepository(base_path=tmp_path)
    item = {"id": "test-1", "data": "test data"}

    # Act - Create
    repo.save(item)

    # Assert - Read
    retrieved = repo.get("test-1")
    assert retrieved["data"] == "test data"

    # Act - Update
    item["data"] = "updated data"
    repo.save(item)

    # Assert
    updated = repo.get("test-1")
    assert updated["data"] == "updated data"

    # Act - Delete
    repo.delete("test-1")

    # Assert
    assert repo.get("test-1") is None
```

#### 4.2 External Service Adapter Tests

Test adapters to external services using mocks:

```python
@pytest.mark.vcr  # Use pytest-vcr to record and replay API calls
def test_openai_transcriber(create_test_audio):
    """Test OpenAI transcriber adapter."""
    # Arrange
    audio_path = create_test_audio()
    transcriber = OpenAITranscriber(api_key="test-key")

    # Act
    result = transcriber.transcribe(audio_path)

    # Assert
    assert len(result) > 0
    for segment in result:
        assert "text" in segment
        assert "start" in segment
        assert "end" in segment
```

## Test Execution Strategy

### Test Execution Order

Tests will be developed and executed in this order:

1. Domain model tests (simplest, most fundamental)
2. Application service tests (builds on domain)
3. Infrastructure tests (depends on domain and application)
4. Integration tests (depends on all layers)
5. Functional tests (end-to-end verification)

### Continuous Integration

Tests will be integrated into the CI pipeline with these stages:

1. Fast tests (unit tests only)
2. Full tests (all tests)
3. Coverage reporting
4. Linting and static analysis

### Test Data Management

Test data will be managed using these approaches:

1. **Generated data**: Created during test execution
2. **Fixtures**: Reusable test setup
3. **Factory patterns**: Dynamic test data creation
4. **Recorded API responses**: For external service tests

## Test Coverage Goals

The test suite will aim to achieve:

- **90% overall code coverage**
- **95% coverage for domain models**
- **90% coverage for application services**
- **85% coverage for infrastructure**

Coverage will be measured using pytest-cov and reported in the CI pipeline.

## Documentation and Maintenance

### Test Documentation

All tests will be documented with:

1. Clear docstrings explaining test purpose
2. Arrange-Act-Assert (AAA) pattern
3. Comments for complex test logic

### Test Maintenance

To ensure maintainability:

1. Keep tests focused on a single aspect
2. Avoid test interdependencies
3. Use descriptive failure messages
4. Regularly refactor test code

## Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Framework Setup | Directory structure, core fixtures, test utilities |
| 2 | Domain Tests | Tests for entities, value objects, business rules |
| 3 | Application Services | Tests for orchestration, coordination, error handling |
| 4 | Infrastructure | Repository tests, adapter tests, integration tests |

## Success Criteria

The test reimplementation will be considered successful when:

1. Overall code coverage reaches the target of 90%
2. All tests are properly organized and documented
3. Tests run reliably and quickly
4. No tests make actual API calls (everything properly mocked)
5. The test suite provides valuable feedback for development

---

*Last updated: 2023-03-21*
