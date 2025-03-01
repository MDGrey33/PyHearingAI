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