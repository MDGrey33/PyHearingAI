# PyHearingAI: Architectural Design

## Architecture Overview

PyHearingAI follows a Clean Architecture approach with a clear separation of concerns across multiple layers:

1. **Domain Layer (Core)**: Contains business entities and repository interfaces
   - Core business entities: `ProcessingJob`, `AudioChunk`, `SpeakerSegment`
   - Repository interfaces defining data access contracts
   - Value objects and enums for domain concepts

2. **Application Layer**: Contains core business logic and services
   - `AudioChunkingService`: Handles chunking of audio files
   - `DiarizationService`: Manages speaker diarization
   - `TranscriptionService`: Manages audio transcription
   - `ReconciliationService`: Reconciles results across chunks
   - `WorkflowOrchestrator`: Coordinates the end-to-end process

3. **Infrastructure Layer**: Provides implementations of repository interfaces and adapters
   - Repository implementations (JSON, in-memory)
   - Adapters for external services (Pyannote, Whisper)
   - Factory functions for service instantiation
   - Storage utilities for managing persistent data

4. **Presentation Layer**: Manages the CLI interface and progress visualization
   - Command-line interface
   - Progress visualization
   - Configuration handling
   - Output formatting

## Key Design Decisions

### Processing Workflow Design

The audio processing workflow is divided into four distinct stages:

1. **Chunking**: Divides large audio files into manageable chunks with overlap
2. **Diarization**: Identifies speaker segments within each chunk
3. **Transcription**: Transcribes audio content for each speaker segment
4. **Reconciliation**: Combines results from overlapping chunks for a coherent output

This staged approach allows for:
- Processing arbitrarily large audio files with constant memory usage
- Parallel processing of independent chunks
- Resumability after interruption at any stage
- Granular progress tracking and reporting

### Job and Chunk Management

The system uses a job-based model where:
- Each processing request is represented as a `ProcessingJob` entity
- Audio files are divided into `AudioChunk` entities
- Speaker segments are stored as `SpeakerSegment` entities
- All entities persist their state to disk during processing
- Repositories manage entity persistence with JSON file storage

This enables:
- Stateful processing with persistence
- Resumability after interruption
- Auditable processing history
- Flexible storage backend options

### Progress Tracking and Visualization

Progress is tracked at multiple levels:
- Overall job progress
- Per-chunk progress
- Per-stage progress (chunking, diarization, transcription, reconciliation)

The system provides:
- Real-time progress updates
- ETA calculation
- Chunk-level status visualization
- Stage-specific progress information

### Parallel Processing Model

The system employs a `ThreadPoolExecutor`-based parallel processing model:
- Configurable maximum worker count
- Automatic task distribution
- Resource optimization based on system capabilities
- Progress tracking across parallel tasks

### Error Handling and Recovery

Error handling follows a robust pattern:
- Graceful shutdown on interruption
- State saving before termination
- Detailed error logging and reporting
- Automatic recovery on resumption

## Design Patterns Used

### Repository Pattern
- Abstracts data storage and retrieval
- Enables persistence mechanism switching
- Separates domain model from storage concerns

### Adapter Pattern
- Wraps external services (Pyannote, Whisper)
- Isolates external dependencies
- Enables service implementation switching

### Factory Pattern
- Used for creating service instances
- Centralizes instantiation logic
- Facilitates dependency injection

### Strategy Pattern
- Applied for different processing strategies
- Used in transcription and diarization approaches
- Enables algorithm switching at runtime

### Observer Pattern
- Implemented for progress tracking
- Allows multiple observers of processing state
- Decouples progress tracking from processing logic

### Facade Pattern
- Implemented in the orchestrator
- Simplifies complex subsystem interactions
- Provides unified interface to the client

## Idempotent Processing Design

Idempotent processing is a core feature enabling:
- Resumability after interruption
- Efficient reprocessing of unchanged parts
- Reliable handling of long-running tasks

Key aspects:
1. **Stateful Processing**: All processing state is persisted
2. **Chunk-based Processing**: Audio is processed in chunks with overlap
3. **Incremental Progress**: Progress is tracked at fine granularity
4. **Work Skipping**: Already processed chunks are skipped on resumption
5. **Consistent Results**: Processing produces the same results regardless of interruptions

### Enhanced Silence Detection

Our implementation optimizes chunk boundaries through silence detection:

1. Uses librosa's RMS energy feature to identify silence regions
2. Optimizes chunk boundaries by aligning them with silence whenever possible
3. Ensures minimum overlap between chunks to preserve context
4. Makes boundary adjustments up to a configurable maximum to find suitable silence regions

### Domain Entities

The core domain entities include:

```python
class ProcessingJob:
    id: str
    original_audio_path: str
    status: ProcessingStatus
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    chunk_duration: float
    processing_options: Dict[str, Any]

class AudioChunk:
    id: str
    job_id: str
    index: int
    start_time: float
    end_time: float
    file_path: str
    status: ChunkStatus
    processing_metadata: Dict[str, Any]

class SpeakerSegment:
    id: str
    chunk_id: str
    speaker_id: str
    start_time: float
    end_time: float
    absolute_start_time: float
    absolute_end_time: float
    has_overlap: bool
    overlap_speakers: Optional[List[str]]
    language: Optional[str]
    status: SegmentStatus
```

### Value Objects

```python
class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ChunkStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SegmentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
```

### Entity ID Handling Strategy

Consistent ID handling is maintained throughout the system:

1. **Service Method Parameters**:
   - Accept both entity objects and string IDs as input
   - Implement helper methods to convert string IDs to entity objects when needed

2. **Repository Interfaces**:
   - Define clear methods for fetching by ID or object
   - Use consistent naming patterns (e.g., `get_by_id`, `save`, `exists`)

3. **Internal Processing**:
   - Always convert to entity objects early in processing flows
   - Use entity objects for all internal operations

4. **ID Generation**:
   - Generate UUIDs at entity creation time
   - Ensure IDs are immutable once created

## Configuration System

The configuration system is designed to be:
- Flexible: supports multiple configuration sources
- Hierarchical: organized in logical sections
- Type-safe: uses dataclasses for configuration
- Feature-flag aware: supports feature toggling
- Environment-sensitive: respects environment variables

```python
@dataclass
class IdempotentProcessingConfig:
    """Configuration settings for idempotent processing."""

    enabled: bool = False
    data_dir: Path = Path.home() / ".pyhearingai"
    jobs_dir: Path = None
    chunks_dir: Path = None
    chunk_duration: float = 300.0  # Default chunk size in seconds
    chunk_overlap: float = 5.0  # Default overlap between chunks in seconds
    use_json_persistence: bool = True

    def __post_init__(self):
        if self.jobs_dir is None:
            self.jobs_dir = self.data_dir / "jobs"
        if self.chunks_dir is None:
            self.chunks_dir = self.data_dir / "chunks"
```

## Persistence Strategy

For persistence, the system uses:

1. JSON files for job metadata and state
2. Audio files for chunked data
3. JSON files for transcription and diarization results

File structure:
```
~/.local/share/pyhearingai/
    /jobs/
        /<job_id>.json               # Job configuration and state
    /chunks/
        /<chunk_id>.json             # Chunk metadata
        /<chunk_id>.wav              # Audio chunk files
    /segments/
        /<segment_id>.json           # Segment metadata
    /transcriptions/
        /chunks/
            /<chunk_id>.json         # Whole chunk transcriptions
        /segments/
            /<segment_id>.json       # Segment transcriptions
    /reconciliation/
        /<job_id>.json               # Final reconciled result
```

## Reconciliation Service Enhancement

### OpenAI Responses API Integration

To address token limit challenges in reconciliation of long audio files, we've designed an enhanced architecture leveraging OpenAI's Responses API for more effective context management and token efficiency.

#### Architecture Components

1. **ResponsesReconciliationAdapter**: A new adapter implementing the reconciliation interface
   ```python
   class ResponsesReconciliationAdapter(BaseReconciliationAdapter):
       def __init__(self, model="gpt-4o"):
           self.client = OpenAI()
           self.model = model
           self.token_counter = TokenCounter(model)

       def reconcile(self, job, diarization_segments, transcription_segments,
                     segment_transcriptions, options=None):
           # Implementation using Responses API
           pass
   ```

2. **BatchProcessor**: Handles splitting reconciliation data into token-appropriate batches
   ```python
   class BatchProcessor:
       def __init__(self, token_counter, max_tokens=7000):
           self.token_counter = token_counter
           self.max_tokens = max_tokens

       def create_batches(self, diarization_segments, transcription_segments,
                          segment_transcriptions):
           # Split data into appropriately sized batches
           pass

       def format_batch(self, batch, batch_index, total_batches):
           # Format batch into message content
           pass
   ```

3. **TokenCounter**: Accurately counts tokens for OpenAI models
   ```python
   class TokenCounter:
       def __init__(self, model="gpt-4o"):
           self.encoding = tiktoken.encoding_for_model(model)

       def count_tokens(self, text):
           # Count tokens in text string
           return len(self.encoding.encode(text))

       def estimate_batch_size(self, segments, text_samples):
           # Estimate maximum segments per batch
           pass
   ```

4. **ResultAggregator**: Combines results from multiple batches into a coherent transcript
   ```python
   class ResultAggregator:
       def process_responses(self, responses, batches):
           # Combine and deduplicate responses
           pass

       def resolve_overlaps(self, segments):
           # Resolve overlapping segments
           pass
   ```

#### Processing Flow

The enhanced reconciliation process follows these steps:

1. **Initialization**:
   - Initialize the token counter for proper batch sizing
   - Set up the OpenAI client
   - Configure batch processor with appropriate token limits

2. **Batch Creation**:
   - Split diarization and transcription data into token-appropriate batches
   - Ensure proper overlap between batches for context continuity
   - Optimize batch boundaries to align with natural breaks in audio

3. **Message Processing**:
   - For each batch:
     - Format data into structured message
     - If first batch, send as new message
     - If subsequent batch, send with previous_response_id
     - Track response ID for next batch
     - Retrieve and parse response
     - Store partial results

4. **Result Aggregation**:
   - Combine results from all batches
   - Resolve overlapping segments
   - Generate final coherent transcript
   - Clean up resources

#### Integration Strategy

The enhanced reconciliation service integrates with existing architecture through:

1. **Adapter Pattern**: Implementation of the `BaseReconciliationAdapter` interface
2. **Feature Flagging**: Configuration option to enable/disable Responses API
3. **Graceful Fallback**: Automatic fallback to existing implementation if needed
4. **Incremental Deployment**: Phased rollout with monitoring and validation

#### Design Benefits

This architectural enhancement provides several advantages:

1. **Server-Side Context Management**: OpenAI maintains conversation state with minimal client-side complexity
2. **Token Efficiency**: Precise token counting and batching prevents limit errors
3. **Stateful Processing**: Previous response IDs maintain context between batches
4. **Arbitrary Length Support**: Process audio files of any length
5. **Improved Quality**: Maintained context between batches improves coherence
6. **Resilience**: Better error handling and recovery mechanisms

## CLI Enhancements

The CLI interface has been enhanced to support the idempotent processing workflow:

```python
@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('-o', '--output', help='Output file path')
@click.option('-f', '--format', 'output_format',
              type=click.Choice(['txt', 'json', 'srt', 'vtt', 'md']),
              default='txt', help='Output format')
@click.option('--resume', is_flag=True, help='Resume previous job if exists')
@click.option('--show-chunks', is_flag=True, help='Show processing status by chunk')
@click.option('--max-workers', type=int, default=4, help='Maximum number of concurrent workers')
def transcribe(audio_file, output, output_format, resume, show_chunks, max_workers):
    """Transcribe an audio file with speaker diarization."""
    # Implementation using the new architecture
```

## Future Extensibility

The architecture supports future extensions:
- Support for additional language models
- New output formats and visualization options
- Alternative storage backends
- UI options beyond command line
- Distributed processing capabilities

## Lessons Learned

During the design evolution, we identified areas for improvement:
1. **Interface Consistency**: Need for consistent parameter naming and typing
2. **Progress Tracking**: Challenges in accurate progress estimation
3. **Error Handling**: Need for comprehensive error classification
4. **Testing Strategies**: Complexity of testing asynchronous workflows

## Testing Architecture

The testing architecture follows the same clean design principles as the main codebase, with clear separation of concerns and layered testing strategies.

### Testing Structure

```
tests/
├── unit/               # Tests for individual components
│   ├── test_hardware_detection/
│   ├── test_responses_adapter/
│   └── test_progress_tracking/
├── integration/        # Tests for component interactions
├── fixtures/           # Test data and factory methods
└── conftest.py         # Shared pytest fixtures
```

### Testing Principles

1. **Clean Architecture Alignment**
   - Tests are organized to match the project's layered architecture
   - Each layer is tested independently with appropriate levels of mocking
   - Domain layer tests are isolated from infrastructure concerns

2. **Test Isolation**
   - Unit tests properly mock dependencies
   - Each test has clear boundaries and focuses on specific functionality
   - Tests don't rely on other tests' state

3. **Test-Driven Development**
   - Tests are written before or alongside implementation
   - Production code changes are separate from test changes
   - Tests verify behavior rather than implementation details

4. **Comprehensive Coverage**
   - Tests cover critical components across all layers:
     - Domain entities and value objects
     - Application services
     - Infrastructure adapters
     - Presentation components

### Mocking Strategy

The project employs a systematic mocking approach:

1. **External Services**
   - OpenAI API calls are mocked for reproducible tests
   - Hardware detection functions are patched for testing different configurations
   - File system operations are mocked to avoid test side effects

2. **Cross-Layer Dependencies**
   - Repository interfaces are mocked in service tests
   - Services are mocked in orchestrator tests
   - External utilities are mocked for isolation

3. **Time and Progress Testing**
   - Time-related functions are patched for deterministic progress testing
   - System utilities are mocked for hardware detection tests

### Test Data Management

1. **Fixtures**
   - Reusable test objects are defined in conftest.py
   - Test audio files are provided in the fixtures directory
   - Factory functions generate test entities

2. **Synthetic Data Generation**
   - Audio generators create test files with known characteristics
   - Entity factories produce domain objects with consistent properties

## Project Status and References

For implementation status and progress tracking, refer to [TODO.md](TODO.md).

For detailed testing strategy and coverage improvement plan, refer to [TEST_PLAN.md](TEST_PLAN.md).

## Job Management Features

### Overview

This document outlines the design for enhancing PyHearingAI's job management capabilities through the command-line interface (CLI). Specifically, it focuses on two new features:

1. Job cancellation (`--cancel`)
2. Job deletion (`--delete`)

These features will give users more control over the processing jobs they create, allowing them to cancel in-progress jobs or completely remove jobs from the system.

### Current Architecture

The PyHearingAI CLI is built on the following components:

- **Command-line Interface**: Using Python's `argparse` to define and parse command-line arguments.
- **Job Repository**: The `JsonJobRepository` class handles job persistence, storing jobs as JSON files.
- **Processing Job**: The `ProcessingJob` class represents a transcription job with status, audio path, and other properties.

The CLI currently supports listing jobs and resuming jobs but doesn't allow canceling or deleting them.

### Proposed Changes

#### 1. CLI Argument Additions

Add two new arguments to the command-line parser:

```python
input_group.add_argument("--cancel", type=str, help="Cancel the processing job with the specified ID")
input_group.add_argument("--delete", type=str, help="Delete the processing job with the specified ID")
```

These arguments should be part of the mutually exclusive input group to ensure users specify only one operation per command.

#### 2. Command Handlers

Implement handlers for these commands in the `main()` function:

```python
# Handle cancel command
if args.cancel:
    job_id = args.cancel
    job_repo = JsonJobRepository()
    job = job_repo.get_by_id(job_id)

    if not job:
        print(f"Error: Job not found with ID: {job_id}", file=sys.stderr)
        return 1

    # Update job status to FAILED to indicate cancellation
    job.status = ProcessingStatus.FAILED
    job_repo.save(job)
    print(f"Job canceled: {job_id}")
    return 0

# Handle delete command
if args.delete:
    job_id = args.delete
    job_repo = JsonJobRepository()
    job = job_repo.get_by_id(job_id)

    if not job:
        print(f"Error: Job not found with ID: {job_id}", file=sys.stderr)
        return 1

    # Delete the job
    if job_repo.delete(job_id):
        print(f"Job deleted: {job_id}")
        return 0
    else:
        print(f"Error: Failed to delete job with ID: {job_id}", file=sys.stderr)
        return 1
```

#### 3. Error Handling

Both commands will follow a consistent pattern for error handling:
- If the specified job ID doesn't exist, print an error message to stderr and return exit code 1
- For the delete command, handle the case where deletion fails

#### 4. Processing Impact

**Cancellation**:
- Updating a job's status to `FAILED` will prevent it from being processed further
- The job will remain in the system and can be viewed in job listings
- Users can still access any partial results that were generated before cancellation

**Deletion**:
- Completely removes the job from the system
- Any related files (chunks, results) should also be cleaned up
- This operation cannot be undone

### Technical Details

#### Repository Methods

The implementation relies on these existing methods in `JsonJobRepository`:

- `get_by_id(job_id)`: Retrieves a job by ID
- `save(job)`: Saves a modified job
- `delete(job_id)`: Deletes a job by ID

#### ProcessingJob Class

For cancellation, we modify the `status` attribute of the `ProcessingJob`:

```python
job.status = ProcessingStatus.FAILED
```

This status change is persisted by calling `job_repo.save(job)`.

### User Experience

#### Command Examples

**Canceling a job**:
```
pyhearingai --cancel 12345abc-def6-789g-hijk-lmnopqrstuv
```

Output on success:
```
Job canceled: 12345abc-def6-789g-hijk-lmnopqrstuv
```

Output when job isn't found:
```
Error: Job not found with ID: 12345abc-def6-789g-hijk-lmnopqrstuv
```

**Deleting a job**:
```
pyhearingai --delete 12345abc-def6-789g-hijk-lmnopqrstuv
```

Output on success:
```
Job deleted: 12345abc-def6-789g-hijk-lmnopqrstuv
```

Output when job isn't found:
```
Error: Job not found with ID: 12345abc-def6-789g-hijk-lmnopqrstuv
```

Output when deletion fails:
```
Error: Failed to delete job with ID: 12345abc-def6-789g-hijk-lmnopqrstuv
```

### Testing Strategy

Tests are already defined in `tests/test_cli.py` for both features. Once implemented, we can enable these tests by removing the `@pytest.mark.skip` decorators.

The tests verify:
- Successful cancellation
- Cancellation of nonexistent jobs
- Successful deletion
- Deletion of nonexistent jobs
- Failed deletion scenarios

### Future Considerations

1. **Clean-up on deletion**: Consider enhancing the `delete` method to clean up all associated files (chunks, audio, results)
2. **Confirmation prompts**: For destructive operations like deletion, add an optional confirmation prompt
3. **Wildcards for batch operations**: Support operations on multiple jobs matching criteria

## Testing Approach

Our testing approach follows several key principles:

### Test-Driven Development (TDD)

We adhere to TDD principles when developing new features:
1. Write failing tests first
2. Implement code to make tests pass
3. Refactor to improve code quality

### Test Organization

Tests are organized to match the codebase structure:
- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests across components
- `tests/test_*.py`: Module-level tests for larger features

### Testing Levels

1. **Unit Tests**: Test individual functions and classes in isolation
2. **Integration Tests**: Test interactions between components
3. **End-to-End Tests**: Test full workflows from CLI to output

### Testing with Immutable Production Code

To prevent unintended changes to production behavior, we follow these rules:

1. **Never Modify Production Code to Fix Tests**
   - Tests should be written or updated to match existing production behavior
   - Production code changes should be driven by feature requirements, not test requirements

2. **Skip Tests That Need Production Code Changes**
   - When tests expect behavior that doesn't exist in production code
   - Document why the test is skipped and what would need to change

3. **Document Required Production Code Changes**
   - Create detailed documentation of required changes in TODO.md
   - Use test plans to guide future development

### Test Fixtures

We use fixtures extensively to:
- Set up test environments
- Create test data
- Mock external dependencies
- Share common test utilities

### Mocking Strategy

When testing components with external dependencies:
1. Mock repositories and services at their interfaces
2. Use predefined test data instead of generating new data
3. Verify interactions with mocked components

### CI/CD Integration

Tests are integrated into the CI/CD pipeline:
- All tests run on every pull request
- Coverage reports are generated automatically
- Test failures block merges to main branch

### Code Coverage Goals

We target high test coverage:
- 89.5% line coverage across the codebase
- 100% coverage for critical components
- Coverage for all error handling paths

### Test Data Management

Test data is managed through:
- Fixture files in the `tests/fixtures` directory
- Helper functions to generate test data
- Cleanup to remove temporary files after tests

### Test Documentation

Each test includes detailed documentation:
- Purpose of the test
- Test setup and requirements
- Expected behavior
- Any special considerations

---

Last updated: 2025-03-19

## Conclusion

The PyHearingAI design follows clean architecture principles while addressing the unique challenges of audio processing at scale. The system prioritizes reliability, resumability, and user experience, all while maintaining a clear separation of concerns and adhering to solid software engineering principles.

The architecture enables processing of arbitrarily large audio files with constant memory usage, parallellization for performance, and robust error handling for reliability.
