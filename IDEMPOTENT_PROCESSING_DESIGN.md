# PyHearingAI Idempotent Processing Design

## Overview

This document outlines the architecture and implementation plan for enhancing PyHearingAI with idempotent processing capabilities, optimized for long multilingual recordings. The design follows Clean Architecture principles and focuses on robustness, resumability, and high-quality transcription results.

## Core Architecture

The implementation follows a hybrid chunking approach:

1. Divide long recordings into manageable time-based chunks
2. Process each chunk through a complete pipeline:
   - Diarize the chunk to identify speaker segments
   - Transcribe the whole chunk for context
   - Transcribe individual speaker segments for accuracy
3. Combine results from all chunks
4. Use GPT-4 to reconcile and produce the final output

## Domain Layer

### Entities

```python
class ProcessingJob:
    job_id: str
    source_file: str
    output_file: str
    output_format: str
    created_at: datetime
    updated_at: datetime
    status: JobStatus
    settings: Dict[str, Any]
    error: Optional[str]

class AudioChunk:
    chunk_id: str
    job_id: str
    index: int
    start_time: float
    end_time: float
    file_path: str
    status: ChunkStatus
    processing_metadata: Dict[str, Any]

class SpeakerSegment:
    segment_id: str
    chunk_id: str
    speaker_id: str
    start_time: float
    end_time: float
    absolute_start_time: float  # Relative to original recording
    absolute_end_time: float    # Relative to original recording
    has_overlap: bool
    overlap_speakers: Optional[List[str]]
    language: Optional[str]
    status: SegmentStatus

class ChunkTranscription:
    chunk_id: str
    text: str
    confidence: float
    language: Optional[str]

class SegmentTranscription:
    segment_id: str
    text: str
    confidence: float
    language: Optional[str]

class FinalSegment:
    id: str
    speaker: str
    start_time: float
    end_time: float
    text: str
    language: Optional[str]
    is_overlapping: bool
    overlapping_with: Optional[List[str]]
```

### Value Objects

```python
class JobStatus(Enum):
    CREATED = "created"
    CHUNKING = "chunking"
    PROCESSING = "processing"
    RECONCILING = "reconciling"
    COMPLETED = "completed"
    FAILED = "failed"

class ChunkStatus(Enum):
    CREATED = "created"
    DIARIZING = "diarizing"
    DIARIZED = "diarized"
    TRANSCRIBING_WHOLE = "transcribing_whole"
    WHOLE_TRANSCRIBED = "whole_transcribed"
    TRANSCRIBING_SEGMENTS = "transcribing_segments"
    SEGMENTS_TRANSCRIBED = "segments_transcribed"
    COMPLETED = "completed"
    FAILED = "failed"

class SegmentStatus(Enum):
    CREATED = "created"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    FAILED = "failed"
```

## Application Layer

### Use Cases

```python
class CreateProcessingJob:
    def execute(self, audio_file: str, output_file: str, output_format: str,
                settings: Dict[str, Any]) -> ProcessingJob:
        # Create a new processing job

class ResumeProcessingJob:
    def execute(self, job_id: str) -> ProcessingJob:
        # Resume an existing processing job

class ProcessNextChunk:
    def execute(self, job: ProcessingJob) -> Optional[AudioChunk]:
        # Process the next unprocessed chunk in the job

class ReconcileResults:
    def execute(self, job: ProcessingJob) -> str:
        # Reconcile and combine results from all chunks
```

### Services

```python
class AudioChunkingService:
    def chunk_audio(self, job: ProcessingJob) -> List[AudioChunk]:
        # Split audio into manageable chunks
        # Recommended chunk size: 5-10 minutes

class DiarizationService:
    def diarize_chunk(self, chunk: AudioChunk) -> List[SpeakerSegment]:
        # Identify speaker segments within a chunk
        # Track overlapping speech segments

class TranscriptionService:
    def transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscription:
        # Transcribe entire chunk for context

    def transcribe_segment(self, segment: SpeakerSegment) -> SegmentTranscription:
        # Transcribe individual speaker segment for accuracy

class ReconciliationService:
    def reconcile(self, job: ProcessingJob) -> List[FinalSegment]:
        # Combine all chunks and segments
        # Handle speaker continuation across chunk boundaries
        # Use GPT-4 to create final transcript
```

## Infrastructure Layer

### Repositories

```python
class JobRepository:
    def save(self, job: ProcessingJob) -> None:
        # Persist job state

    def find_by_id(self, job_id: str) -> Optional[ProcessingJob]:
        # Retrieve job by ID

    def find_by_source_file(self, source_file: str) -> List[ProcessingJob]:
        # Find jobs by source file

class ChunkRepository:
    def save(self, chunk: AudioChunk) -> None:
        # Persist chunk state

    def find_by_job_id(self, job_id: str) -> List[AudioChunk]:
        # Find all chunks for a job

    def find_next_unprocessed(self, job_id: str) -> Optional[AudioChunk]:
        # Find next chunk to process

class SegmentRepository:
    def save(self, segment: SpeakerSegment) -> None:
        # Persist segment state

    def find_by_chunk_id(self, chunk_id: str) -> List[SpeakerSegment]:
        # Find all segments for a chunk

class TranscriptionRepository:
    def save_chunk_transcription(self, transcription: ChunkTranscription) -> None:
        # Save whole chunk transcription

    def save_segment_transcription(self, transcription: SegmentTranscription) -> None:
        # Save segment transcription

    def find_by_job_id(self, job_id: str) -> List[ChunkTranscription]:
        # Find all chunk transcriptions for a job

    def find_segments_by_job_id(self, job_id: str) -> List[SegmentTranscription]:
        # Find all segment transcriptions for a job
```

### Adapters

```python
class PyannoteAdapter:
    # Interface with pyannote for diarization

class WhisperAdapter:
    # Interface with Whisper models for transcription

class OpenAIAdapter:
    # Interface with GPT-4 for reconciliation
```

## Persistence Strategy

For persistence, we'll use a combination of:

1. JSON files for job metadata and state
2. Audio files for chunked data
3. JSON files for transcription and diarization results

File structure:
```
/tmp/pyhearingai_jobs/
    /<job_id>/
        job.json                # Job configuration and state
        chunks/
            <chunk_id>.wav      # Audio chunk files
            <chunk_id>.json     # Chunk metadata
        segments/
            <segment_id>.wav    # Segment audio files (optional)
            <segment_id>.json   # Segment metadata
        transcriptions/
            chunks/
                <chunk_id>.json # Whole chunk transcriptions
            segments/
                <segment_id>.json # Segment transcriptions
        final/
            reconciled.json     # Final reconciled result
            output.<format>     # Final output in requested format
```

## Processing Pipeline

### 1. Job Creation & Chunking

```
START → Create Job → Chunk Audio File → Save Chunks → PROCESSING
```

### 2. Chunk Processing (for each chunk)

```
LOAD CHUNK → Diarize Chunk → Save Segments →
Transcribe Whole Chunk → Save Whole Transcription →
Transcribe Segments → Save Segment Transcriptions → MARK CHUNK COMPLETED
```

### 3. Reconciliation & Completion

```
ALL CHUNKS PROCESSED → Prepare Reconciliation Data →
Send to GPT-4 → Parse Response → Save Final Result →
Generate Output → MARK JOB COMPLETED
```

## GPT-4 Reconciliation

The GPT-4 prompt will include:

1. All chunk transcriptions in sequence
2. All segment transcriptions grouped by chunk
3. Metadata about speaker segments, including:
   - Speaker IDs
   - Timestamps
   - Chunk boundaries
   - Detected languages (if available)
   - Overlapping speech indicators

The prompt will instruct GPT-4 to:
1. Maintain speaker continuity across chunk boundaries
2. Resolve discrepancies between whole-chunk and segment transcriptions
3. Handle multilingual content appropriately
4. Properly attribute overlapping speech
5. Return a structured JSON response with the final transcript

## CLI Enhancements

```python
@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('-o', '--output', help='Output file path')
@click.option('-f', '--format', 'output_format',
              type=click.Choice(['txt', 'json', 'srt', 'vtt', 'md']),
              default='txt', help='Output format')
@click.option('--resume', is_flag=True, help='Resume previous job if exists')
@click.option('--show-chunks', is_flag=True, help='Show processing status by chunk')
def transcribe(audio_file, output, output_format, resume, show_chunks):
    """Transcribe an audio file with speaker diarization."""
    # Implementation using the new architecture
```

## Components to Preserve

The following components from the current PyHearingAI architecture will remain largely untouched and be reused in the new implementation:

### 1. Core Model Adapters

These components can be reused with minimal changes:

- **WhisperOpenAITranscriber**: The adapter for OpenAI's Whisper API can be reused as-is, since the underlying API interaction won't change
- **PyannoteAdapter**: The existing diarization implementation using Pyannote can be preserved
- **Model Authentication**: All code handling API keys and authentication with external services

### 2. Output Formatters

The existing output formatting logic can be reused:

- **Format Converters**: Code that converts transcription to various formats (TXT, JSON, SRT, VTT, MD)
- **Speaker Attribution Logic**: Existing code that formats speaker identifications
- **Timestamp Formatting**: Logic that handles formatting of time codes

### 3. Audio Processing Utilities

Low-level audio utilities should remain untouched:

- **Audio Loading**: Functions for loading and validating audio files
- **Audio Format Conversion**: Any utilities that handle audio format conversions
- **Sample Rate Handling**: Code that manages sample rates for different models

### 4. CLI Interface Structure

The basic CLI structure will remain, with enhancements:

- **Command Registration**: Basic command structure and Click framework usage
- **Parameter Definitions**: Most parameter definitions will be preserved
- **Help Text**: Core help text and documentation

### 5. Configuration Management

Configuration handling can be largely preserved:

- **Environment Variable Processing**: Code that handles environment variables
- **API Key Management**: Logic for retrieving and validating API keys
- **Model Configuration**: Settings for controlling model behavior

### Components Requiring Significant Changes

To implement the idempotent processing design, these areas will require significant changes:

1. **Process Flow**: The main execution flow will change to support chunking and resumability
2. **State Management**: New code for tracking and persisting processing state
3. **CLI Extensions**: Adding resume capabilities and progress reporting
4. **Storage Layer**: Adding the repository implementations for managing persisted data
5. **Chunking Logic**: New services for dividing audio and managing chunk processing
6. **Reconciliation**: New code for GPT-4 reconciliation of processed chunks

## Testing Strategy

1. **Unit Tests**:
   - Individual component testing with mocked dependencies
   - Focus on state transitions and error handling

2. **Integration Tests**:
   - End-to-end processing of short test audio files
   - Verify correct chunk/segment generation

3. **Manual Testing**:
   - Test with various languages, including Latvian/English mix
   - Validate resumability by intentionally interrupting processing

## Impact on Existing Tests

Following the proposed refactoring, we anticipate several categories of existing tests to fail initially. This assessment helps in planning the test adaptation strategy.

### Tests Expected to Fail

1. **End-to-End Tests**:
   - Tests that execute the full CLI pipeline will fail due to the new chunking approach
   - `test_end_to_end.py` - Complete workflow tests will need updates to accommodate the new pipeline stages
   - Any tests relying on immediate processing without chunking

2. **CLI Interface Tests**:
   - Tests verifying command-line arguments will fail due to new parameters (`--resume`, `--show-chunks`)
   - Tests expecting specific output formats may fail if the chunking affects output generation
   - Tests that verify help text content will need updating

3. **Processing Flow Tests**:
   - Tests that verify the direct flow from transcription to diarization will break because of the new order
   - Tests with expectations around memory usage and processing speed
   - Tests that don't account for the intermediate state persistence

4. **Diarization-Specific Tests**:
   - Tests expecting diarization to be applied to the entire audio file instead of chunks
   - Tests verifying speaker continuity across the entire recording

### Tests Expected to Pass

1. **Model Adapter Unit Tests**:
   - Tests for `WhisperOpenAITranscriber` and model-specific functionality should continue to pass
   - Tests for authentication and API key handling should remain valid

2. **Output Format Tests**:
   - Tests verifying the structure of each output format (JSON, TXT, SRT, etc.) should pass
   - Tests for timestamp formatting and speaker attribution in isolation

3. **Audio Utility Tests**:
   - Tests for audio loading, validation, and format conversion should pass
   - Sample rate management tests should remain valid

4. **Configuration Tests**:
   - Tests for environment variable processing and configuration loading

### Test Adaptation Strategy

1. **Update Test Fixtures**:
   - Create new fixtures that represent the chunked processing model
   - Add fixtures for various processing states to test resumability

2. **Mock New Components**:
   - Create mocks for the new repositories and services
   - Develop test doubles for the chunking and reconciliation services

3. **Add New Test Categories**:
   - State persistence and recovery tests
   - Chunk boundary handling tests
   - Reconciliation accuracy tests

4. **Modify Existing Tests**:
   - Update end-to-end tests to verify the complete chunked workflow
   - Adapt CLI tests to include the new parameters
   - Modify processing flow tests to account for the new pipeline stages

By anticipating these test failures, we can better plan the refactoring process and ensure test coverage remains high throughout the implementation.

## Module-Based Refactoring Plan

This plan outlines a module-by-module approach to gradually refactor PyHearingAI while maintaining system functionality throughout the process. Each module can be developed, tested, and integrated independently, allowing for incremental adoption.

### Module 1: Core Domain and State Management (`pyhearingai.core`)

**New modules to create:**
```
pyhearingai/
└── core/
    ├── __init__.py
    ├── domain/
    │   ├── __init__.py
    │   ├── job.py           # ProcessingJob entity
    │   ├── audio_chunk.py   # AudioChunk entity
    │   ├── segment.py       # SpeakerSegment entity
    │   └── enums.py         # Status enums
    └── repository/
        ├── __init__.py
        ├── job_repository.py
        ├── chunk_repository.py
        └── segment_repository.py
```

**Implementation:**
1. Create domain entities with proper data classes and value objects
2. Implement file-based repositories with JSON serialization
3. Add utility functions for working with job state

**Integration point:**
- Add feature flag in `config.py`: `USE_IDEMPOTENT_PROCESSING=False`
- No changes to existing code flow yet

### Module 2: Audio Processing (`pyhearingai.audio`)

**New modules to create:**
```
pyhearingai/
└── audio/
    ├── __init__.py
    ├── chunking.py      # Audio chunking service
    ├── utils.py         # Audio utilities
    └── timestamps.py    # Timestamp conversion utilities
```

**Implementation:**
1. Create `AudioChunkingService` with configurable chunk size
2. Implement silence detection for optimal chunk boundaries
3. Add utilities for converting between chunk and absolute timestamps
4. Reuse existing audio loading code through imports

**Integration point:**
- Audio chunking functions are called only when feature flag is enabled
- Can operate in single-chunk mode to mimic current behavior

### Module 3: Diarization Service (`pyhearingai.diarization`)

**New/modified modules:**
```
pyhearingai/
└── diarization/
    ├── __init__.py
    ├── service.py             # New: DiarizationService
    ├── adapters/
    │   ├── __init__.py
    │   ├── pyannote.py        # Modified: Add chunk awareness
    │   └── base.py            # Modified: Update interface
    └── repositories/
        ├── __init__.py
        └── diarization_repository.py  # New: Store diarization results
```

**Implementation:**
1. Create `DiarizationService` wrapper around existing adapters
2. Add chunk-aware methods to existing diarization adapters
3. Implement storage for diarization results
4. Maintain backward compatibility with existing interface

**Integration point:**
- Update the `transcribe` command to use the new service when feature flag is enabled
- Retain direct adapter usage when flag is disabled

### Module 4: Transcription Service (`pyhearingai.transcription`)

**New/modified modules:**
```
pyhearingai/
└── transcription/
    ├── __init__.py
    ├── service.py                # New: TranscriptionService
    ├── adapters/
    │   ├── __init__.py
    │   ├── whisper.py            # Modified: Add chunk awareness
    │   └── base.py               # Modified: Update interface
    └── repositories/
        ├── __init__.py
        └── transcription_repository.py  # New: Store results
```

**Implementation:**
1. Create `TranscriptionService` with methods for whole-chunk and segment transcription
2. Update adapters to handle both chunk and segment transcription
3. Implement storage for transcription results
4. Maintain backward compatibility with existing interfaces

**Integration point:**
- Update transcription code to use new service when feature flag is enabled
- Retain direct adapter usage when flag is disabled

### Module 5: Reconciliation Service (`pyhearingai.reconciliation`)

**New modules to create:**
```
pyhearingai/
└── reconciliation/
    ├── __init__.py
    ├── service.py        # ReconciliationService
    ├── gpt.py            # GPT-4 integration
    ├── prompt.py         # Prompt templates
    └── repository.py     # Final result storage
```

**Implementation:**
1. Create `ReconciliationService` to combine chunk results
2. Implement GPT-4 integration for advanced reconciliation
3. Create prompt templates for different scenarios
4. Add result storage and formatting functions

**Integration point:**
- Called after all chunks are processed when feature flag is enabled
- Reuses existing output formatting code

### Module 6: CLI Enhancements (`pyhearingai.cli`)

**Modified modules:**
```
pyhearingai/
└── cli.py    # Update with new options and flows
```

**Implementation:**
1. Add `--resume` and `--show-chunks` CLI options
2. Implement job lookup and resumption flow
3. Add progress reporting during processing
4. Create dual flow based on feature flag

**Integration point:**
- Enhance existing CLI but maintain backward compatibility
- Original command behavior preserved when new options aren't used

### Module 7: Orchestration Layer (`pyhearingai.orchestration`)

**New modules to create:**
```
pyhearingai/
└── orchestration/
    ├── __init__.py
    ├── workflow.py       # Processing workflow
    ├── job_manager.py    # Job state management
    └── progress.py       # Progress tracking
```

**Implementation:**
1. Create main workflow orchestrator
2. Implement job management functions
3. Add progress tracking and reporting
4. Connect all services into coherent workflow

**Integration point:**
- Main entry point when feature flag is enabled
- Fully encapsulates the new processing flow

### Module 8: Configuration and Integration (`pyhearingai.config`)

**Modified modules:**
```
pyhearingai/
└── config.py    # Update with new configuration options
```

**Implementation:**
1. Add configuration options for idempotent processing
2. Create settings for chunk size, storage location, etc.
3. Implement configuration migration utilities
4. Remove feature flags and finalize integration

**Integration point:**
- Final switch to new architecture
- Remove conditional code paths
- Maintain backward compatibility through configuration

### Refactoring Order and Dependency Graph

```
1. Core Domain     ────┐
                       │
2. Audio Processing ───┼─┐
                       │ │
3. Diarization ────────┘ │
       │                 │
       └────────┐        │
                ▼        │
4. Transcription         │
       │                 │
       └────────┐        │
                ▼        │
5. Reconciliation        │
       │                 │
       └────────┐        │
                ▼        │
6. CLI Enhancements      │
       │                 │
       └────────┐        │
                ▼        │
7. Orchestration ◄───────┘
       │
       └────────┐
                ▼
8. Configuration/Integration
```

### Parallel Development Strategy

To accelerate development while ensuring compatibility:

1. **Core Modules (1-2)**: Can be developed independently of the current system
2. **Adapter Modules (3-4)**: Wrapped around existing code with dual interface
3. **New Features (5-7)**: Developed in parallel once core modules are stable
4. **Integration (8)**: Final step after all modules are thoroughly tested

This approach allows refactoring one module at a time while maintaining a working system throughout the process. Each module can be activated gradually using feature flags, ensuring that the system can fall back to the original implementation if issues arise.

## Implementation Plan

The detailed implementation plan with specific tasks, timelines, and success metrics has been moved to the project's TODO.md file. This serves as the primary working document for tracking progress on the implementation of the idempotent processing features.

Please refer to TODO.md for:
- Sprint-by-sprint implementation tasks
- Assessment criteria for each milestone
- Success metrics for the overall implementation

## Conclusion

The idempotent processing design for PyHearingAI brings significant improvements to the handling of long multilingual recordings through a clean architecture approach. By implementing chunked processing with state persistence, we achieve:

1. **Resilience**: The system can recover from interruptions at any point in the processing pipeline, making it ideal for long-running tasks.

2. **Quality**: The hybrid approach of diarizing chunks and then transcribing both whole chunks and individual speaker segments maximizes accuracy, especially for multilingual content.

3. **Efficiency**: By processing audio in manageable chunks, memory usage is controlled while maintaining high-quality results.

4. **Maintainability**: The clean architecture with clear separation of concerns makes the system easier to extend and maintain over time.

This design specifically addresses the challenges of processing 1.5-hour Latvian/English recordings by handling overlapping speech, maintaining speaker continuity across chunk boundaries, and leveraging GPT-4's capabilities for reconciliation.

The implementation strategy follows a progressive, module-based approach that allows for continuous evaluation and adjustment, ensuring that each component builds upon existing functionality. By preserving key components of the existing system while enhancing its capabilities, we create a robust solution that meets the complex requirements of multilingual audio transcription with speaker diarization.
