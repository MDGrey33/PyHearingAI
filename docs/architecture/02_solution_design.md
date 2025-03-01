# Solution Design

## Overview

PyHearingAI provides audio transcription and speaker identification with a simple, extensible API. This document outlines the detailed design of the solution.

## 1. Core Components

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Audio Input │──▶│Audio Process│──▶│Transcription│──▶│Speaker Diari│──▶│   Speaker   │──▶│ Output Form │
│             │   │    or       │   │   Engine    │   │  zation     │   │ Assignment  │   │    ats      │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

### Component Descriptions

1. **Audio Input**: Handles various audio file formats (.mp3, .wav, etc.)
2. **Audio Processor**: Converts audio to required format for transcription/diarization
3. **Transcription Engine**: Converts speech to text with timestamps
4. **Speaker Diarization**: Identifies when different speakers are talking
5. **Speaker Assignment**: Uses AI (GPT-4o) to identify speakers based on conversation content
6. **Result Combiner**: Merges transcription with speaker information
7. **Output Formats**: Provides results in various formats (TXT, JSON, SRT, Markdown)

## 2. Main API

```python
def transcribe(
    audio: Union[str, bytes, IO],
    transcriber: Union[str, dict] = "default",
    diarizer: Union[str, dict] = "default",
    speaker_assigner: Union[str, dict] = "default",
    output_format: str = "txt",
    progress_callback: Optional[Callable] = None
) -> TranscriptionResult:
    """
    Transcribe audio with speaker identification.
    
    Args:
        audio: Path to audio file, bytes, or file-like object
        transcriber: Transcription model to use (name or config dict)
        diarizer: Diarization model to use (name or config dict)
        speaker_assigner: Speaker assignment model to use (name or config dict)
        output_format: Output format (txt, json, srt, vtt, md)
        progress_callback: Function to call with progress updates
        
    Returns:
        TranscriptionResult object with the results
    """
```

## 3. Advanced API

```python
@contextmanager
def pipeline_session(
    transcriber: Union[str, dict] = "default",
    diarizer: Union[str, dict] = "default",
    **options
):
    """
    Create a reusable pipeline session for multiple transcriptions.
    
    Args:
        transcriber: Transcription model name or config dict
        diarizer: Diarization model name or config dict
        options: Additional options for the pipeline
        
    Yields:
        Session object with transcribe method
    """
    # Initialize models and resources
    try:
        # Create session object
        session = Session(transcriber, diarizer, **options)
        yield session
    finally:
        # Clean up resources
        session.close()
```

## 4. Result Object

```python
class TranscriptionResult:
    """Holds transcription results with speaker information."""
    
    text: str  # Complete transcript
    speakers: List[str]  # List of speaker names/ids
    segments: List[Segment]  # Detailed segment information
    
    def save(self, path: Union[str, Path], **options) -> None:
        """Save transcript to a file in the format specified by extension."""
        
    def to_dict(self) -> dict:
        """Convert result to a dictionary."""
        
    def to_str(self, format_name: str = "text") -> str:
        """Convert result to string using specified format."""
        
    @property
    def duration(self) -> float:
        """Total duration of the audio in seconds."""
```

## 5. Usage Examples

### 5.1 Basic Usage

```python
from pyhearingai import transcribe

# Simple transcription
result = transcribe("meeting.mp3")
print(result.text)  # Full transcript with speaker labels

# Save to different formats
result.save("transcript.txt")  # Text format
result.save("transcript.json") # JSON format with detailed information
result.save("transcript.srt")  # SubRip subtitle format
```

### 5.2 Model Selection

```python
# Choose specific models
result = transcribe(
    "interview.mp3",
    transcriber="whisper-openai",
    diarizer="pyannote",
    speaker_assigner="gpt-4o",
    output_format="markdown"
)
```

### 5.3 Progress Tracking

```python
def progress_handler(info):
    print(f"Stage: {info['stage']}, Progress: {info['progress']:.0%}")
    
result = transcribe("long_recording.mp3", 
                   progress_callback=progress_handler)
```

### 5.4 Reusing Resources

```python
# Process multiple files efficiently
with pipeline_session(transcriber="whisper-large") as session:
    result1 = session.transcribe("meeting1.mp3")
    result2 = session.transcribe("meeting2.mp3")
```

## 6. Extension Points

### 6.1 Registering Transcribers

```python
from pyhearingai.extensions import register_transcriber
from pyhearingai.models import Transcriber

@register_transcriber("my-transcriber")
class MyTranscriber(Transcriber):
    def transcribe(self, audio_path):
        # Implementation
        return segments
```

### 6.2 Registering Diarizers

```python
from pyhearingai.extensions import register_diarizer
from pyhearingai.models import Diarizer

@register_diarizer("my-diarizer")
class MyDiarizer(Diarizer):
    def diarize(self, audio_path):
        # Implementation
        return speaker_segments
```

### 6.3 Registering Speaker Assigners

```python
from pyhearingai.extensions import register_speaker_assigner
from pyhearingai.models import SpeakerAssigner

@register_speaker_assigner("my-assigner")
class MySpeakerAssigner(SpeakerAssigner):
    def assign_speakers(self, transcript_segments, diarization_segments):
        # Implementation
        return labeled_segments
```

### 6.4 Custom Audio Processing

```python
@register_processor("noise-reduction")
def reduce_noise(audio, config=None):
    """
    Custom audio preprocessing.
    
    Args:
        audio: Raw audio data
        config: Optional configuration dictionary
        
    Returns:
        Processed audio data
    """
    # Your implementation here
    return processed_audio
```

### 6.5 Custom Output Formats

```python
@register_format(".custom")
def custom_format(result, options=None):
    """
    Custom output format.
    
    Args:
        result: TranscriptionResult object
        options: Optional formatting options
        
    Returns:
        Formatted string
    """
    # Your implementation here
    return formatted_text
```

## 7. Configuration

### 7.1 API Keys

```python
# Using environment variables
OPENAI_API_KEY=xxx
HUGGINGFACE_API_KEY=xxx

# Programmatic configuration
import pyhearingai
pyhearingai.configure(
    openai_api_key="xxx",
    huggingface_api_key="xxx"
)
```

### 7.2 Default Models

```python
# Set default models for the library
pyhearingai.configure(
    default_transcriber="whisper-large",
    default_diarizer="pyannote"
)
```

## 8. Error Handling

### 8.1 Exception Hierarchy

```
BaseError
├── AudioProcessingError
├── TranscriptionError
│   └── ModelNotFoundError
├── DiarizationError
└── OutputFormatError
```

### 8.2 Partial Results

```python
try:
    result = transcribe("audio.mp3")
except TranscriptionError as e:
    if e.partial:
        # Use partial results
        print(f"Partial transcript: {e.partial.text}")
        e.partial.save("partial.txt")
```

## 9. File Organization

```
src/pyhearingai/
├── __init__.py         # Public API
├── pipeline.py         # Core pipeline implementation
├── models/
│   ├── __init__.py     # Model registry
│   ├── base.py         # Base classes
│   ├── transcribers/   # Transcription models
│   └── diarizers/      # Diarization models
├── audio/
│   ├── __init__.py     # Audio processing
│   └── processors.py   # Audio preprocessing
├── outputs/
│   ├── __init__.py     # Output format registry
│   └── formats.py      # Output format implementations
└── utils/
    ├── __init__.py
    ├── config.py       # Configuration management
    └── errors.py       # Error definitions
```

## 10. Testing Structure

```
tests/
├── test_pipeline.py    # End-to-end tests
├── test_transcribers.py # Transcription tests
├── test_diarizers.py   # Diarization tests
├── test_audio.py       # Audio processing tests
├── test_outputs.py     # Output format tests
└── fixtures/           # Test audio files
```

## 11. Performance Considerations

- **Lazy Loading**: Models are loaded only when needed
- **Resource Pooling**: Reuse model instances across calls when possible
- **Streaming**: Support for processing audio in chunks (future)
- **Caching**: Cache intermediate results for repeated operations
- **Memory Management**: Clean up resources after use