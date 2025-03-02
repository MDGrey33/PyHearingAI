# PyHearingAI Implementation Todo List

This document lists features that are mentioned in the README but are not yet fully implemented in the codebase.

## Missing Features

### Processing Large Files
- [ ] Implement `transcribe_chunked` function for processing large audio files in chunks
```python
# For large files, process in chunks
from pyhearingai import transcribe_chunked

result = transcribe_chunked(
    "very_long_meeting.mp3",
    chunk_size_seconds=600,  # Process 10-minute chunks
    overlap_seconds=30       # Overlap chunks by 30 seconds
)
```

### Resource Management
- [ ] Implement `pipeline_session` context manager for efficient handling of multiple files
```python
from pyhearingai import pipeline_session

# Process multiple files with resource reuse
with pipeline_session(config) as session:
    result1 = session.transcribe("file1.mp3")
    result2 = session.transcribe("file2.mp3")
    # Resources are efficiently managed
```

### Memory Management
- [ ] Implement memory limit functionality
```python
# Set memory usage limits
from pyhearingai.config import set_memory_limit

# Limit total memory usage to 4GB
set_memory_limit(4096)  # In MB
```

- [ ] Implement resource cleanup functionality
```python
# Clean up resources when done
from pyhearingai import cleanup_resources

# After processing several files
cleanup_resources()
```

## Partial Implementations

### Configuration
- [ ] Create a formal `TranscriptionConfig` class as shown in the README
```python
from pyhearingai import transcribe
from pyhearingai.models import TranscriptionConfig

# Configure transcription with specific models
config = TranscriptionConfig(
    transcriber="whisper-openai",
    diarizer="pyannote",
    speaker_assigner="gpt-4o",
    output_format="json",
    language="en"
)

# Process with specific configuration
result = transcribe("interview.mp3", config=config)
```

### Exception Handling
- [ ] Implement specific exception classes for different error types
  - [ ] `TranscriptionError`
  - [ ] `DiarizationError`
  - [ ] `AudioProcessingError`
  - [ ] `SpeakerAssignmentError`

## Development Notes

- These features should be prioritized based on user needs
- Consider whether all features mentioned in the README are necessary, or if the README should be updated to match current implementation
- Update this list as features are implemented
