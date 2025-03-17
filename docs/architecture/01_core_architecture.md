# Core Architecture

## Overview

PyHearingAI converts audio to text with speaker identification. Simple to use, easy to extend.

## 1. Design Philosophy

```
"Make simple things simple and complex things possible."
```

- **Simple by default** - One line for common use cases
- **Extensible by design** - Easy to add new models and formats
- **Progressive complexity** - Start simple, add features as needed

## 2. Core Components

The library consists of these core components:

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Audio Input │──▶│  Processing │──▶│Transcription│──▶│   Speaker   │──▶│    Result   │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

1. **Audio Input**: Handles various audio formats (.mp3, .wav, .m4a, etc.)
2. **Processing**: Converts audio to the required format for the transcription and diarization models
3. **Transcription**: Converts audio to text using speech recognition models (e.g., OpenAI Whisper)
4. **Speaker Diarization**: Identifies who spoke when in the audio
   - **Speaker Assignment**: Uses GPT-4o to assign speaker identities based on transcript content
5. **Result**: Formats and returns the transcription with speaker information in various formats

## 3. User API

The library exposes a simple main API for basic usage and an advanced API for more control.

### Simple API

```python
from pyhearingai import transcribe

# Basic usage
result = transcribe("meeting_recording.mp3")
print(result.text)  # Prints the full transcript with speaker labels

# Save in different formats
result.save("transcript.txt")  # Plain text
result.save("transcript.json")  # JSON with segments, timestamps
result.save("transcript.srt")   # Subtitle format
result.save("transcript.md")    # Markdown format
```

### Advanced API

```python
from pyhearingai import transcribe, pipeline_session
from pyhearingai.models import TranscriptionConfig

# Configure transcription with specific models
config = TranscriptionConfig(
    transcriber="whisper-openai",
    diarizer="pyannote",
    speaker_assigner="gpt-4o",  # Added speaker assignment model
    output_format="json",
    audio_processor="ffmpeg",
    language="en"
)

# Process a single file with specific configuration
result = transcribe("interview.mp3", config=config)

# Or set up a reusable pipeline session
with pipeline_session(config) as session:
    result1 = session.transcribe("file1.mp3")
    result2 = session.transcribe("file2.mp3")
    # Resources are efficiently managed
```

## 4. Extension System

### 4.1 Extending Models
```python
# Add a custom transcription model
@register_transcriber("my-model")
def my_transcriber(audio, config=None):
    # Process audio
    return {"text": "...", "segments": [...]}

# Use your model
result = transcribe("audio.mp3", transcriber="my-model")
```

### 4.2 Custom Processing
```python
# Add custom audio preprocessing
@register_processor("noise-reduction")
def reduce_noise(audio, config=None):
    # Remove noise
    return processed_audio

# Add custom output format
@register_format(".myformat")
def my_format(result, options=None):
    # Format the result
    return formatted_text
```

## 5. Implementation Strategy

### 5.1 Core Function
```python
def transcribe(audio, **options):
    # 1. Load and validate the audio
    audio_data = load_audio(audio)

    # 2. Process audio (conversion, normalization)
    processed = process_audio(audio_data, options)

    # 3. Transcribe audio
    transcript = run_transcription(processed, options)

    # 4. Identify speakers
    speakers = run_diarization(processed, options)

    # 5. Combine results
    return create_result(transcript, speakers)
```

### 5.2 Extension Registry
```python
# Registry for extensions
_TRANSCRIBERS = {}
_DIARIZERS = {}
_PROCESSORS = {}
_FORMATS = {}

# Registration decorators
def register_transcriber(name):
    def decorator(func):
        _TRANSCRIBERS[name] = func
        return func
    return decorator
```

## 6. Project Structure
```
src/pyhearingai/
├── __init__.py     # Public API
├── pipeline.py     # Core pipeline
├── models.py       # Model integrations
├── processing.py   # Audio processing
├── outputs.py      # Output formats
└── extensions/     # Extension plugins
```

## 7. Configuration & Error Handling

### 7.1 Configuration
```python
# Environment variables
OPENAI_API_KEY=xxx
HUGGINGFACE_API_KEY=xxx

# Or programmatically
pyhearingai.configure(openai_api_key="xxx")
```

### 7.2 Error Handling
```python
try:
    result = transcribe("audio.mp3")
except TranscriptionError as e:
    if e.partial:
        # Use partial results
        print(e.partial.text)
```

## 8. Hardware Acceleration

PyHearingAI automatically detects and utilizes available hardware acceleration:

```python
# The system uses the optimal device without configuration
result = transcribe("interview.mp3")

# Under the hood, device prioritization happens:
# 1. Apple Silicon (MPS) on Mac
# 2. NVIDIA GPUs (CUDA)
# 3. CPU with optimized threading
```

### 8.1 Hardware Detection

```python
from pyhearingai.infrastructure.diarizers.pyannote import PyannoteDiarizer

# Instantiate the diarizer to see what hardware is being used
diarizer = PyannoteDiarizer()
print(f"Using device: {diarizer._device}")  # Will show "mps", "cuda", or "cpu"
```

### 8.2 Advanced Resource Configuration

```python
from pyhearingai import transcribe

# Configure specific resource usage
result = transcribe(
    "long_file.mp3",
    max_workers=4,  # Control thread pool size
    chunk_size=15.0  # Control audio chunk size in seconds
)
```

## 9. Progress Tracking

### 9.1 Simple Progress Callback

```python
def progress_callback(progress_info):
    stage = progress_info.get('stage', 'unknown')
    percent = progress_info.get('progress', 0) * 100
    print(f"Processing {stage}: {percent:.1f}% complete")

result = transcribe(
    "long_recording.mp3",
    progress_callback=progress_callback
)
```

### 9.2 Rich Progress Reporting

PyHearingAI provides rich progress reporting in the terminal:

```
Overall Progress: [████████████████████████████████] 100% | 42.8s
Batch 3/5: [█████████████████░░░░░░░░░░░░░░░░] 60% | ETA: 28s
Transcribing chunks: [██████████████████████████████] 100% | 18/18 complete
```

The progress reporting system includes:
- Overall job progress with time estimates
- Batch-level progress tracking
- Individual step progress (diarization, transcription, etc.)
- Estimated time to completion
- Performance metrics

## 10. Performance Optimization

### 10.1 Batch Processing

```python
from pyhearingai import transcribe

# Process long files efficiently with automatic batching
result = transcribe(
    "long_conference.mp3",
    chunk_size=10.0,  # 10-second chunks
    batch_size=10     # Process 10 chunks per batch
)
```

### 10.2 Memory Management

```python
from pyhearingai import transcribe

# Configure memory usage
result = transcribe(
    "large_file.mp3",
    keep_audio_in_memory=False,  # Store chunks on disk
    cleanup_after_processing=True  # Remove temporary files
)
```
