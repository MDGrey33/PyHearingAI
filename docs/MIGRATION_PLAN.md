# Migration Plan: Current Implementation to Clean Architecture

This document outlines the plan for migrating the current PyHearingAI implementation to the new Clean Architecture design.

## Current Implementation

The current implementation follows a procedural approach with separate modules:

```
PyHearingAI/
├── main.py                     # Main entry point with pipeline stages
├── modules/
│   ├── __init__.py
│   ├── audio_conversion.py     # Convert audio to WAV
│   ├── transcription.py        # Transcribe audio using OpenAI Whisper
│   ├── diarization.py          # Speaker diarization using Pyannote
│   └── speaker_assignment.py   # Assign speakers to transcript segments
├── content/                    # Output directory for pipeline stages
└── example_audio.m4a           # Example input file
```

The current pipeline processes audio in four sequential steps:
1. Audio conversion: Convert input audio to WAV format
2. Transcription: Generate transcript from audio
3. Diarization: Identify speaker segments
4. Speaker assignment: Combine transcript and speaker information

## Target Architecture

The new architecture follows Clean Architecture principles with clear separation of concerns:

```
src/pyhearingai/
├── __init__.py                 # Public API
├── pipeline.py                 # Core pipeline
├── models.py                   # Model integrations
├── processing.py               # Audio processing
├── outputs.py                  # Output formats
└── extensions/                 # Extension plugins
```

Key principles:
- Single entry point (`transcribe` function)
- Extensible model registry
- Clear configuration management
- Proper error handling
- Resource cleanup

## Migration Steps

### 1. Create Project Structure

```bash
mkdir -p src/pyhearingai/extensions
touch src/pyhearingai/__init__.py
touch src/pyhearingai/pipeline.py
touch src/pyhearingai/models.py
touch src/pyhearingai/processing.py
touch src/pyhearingai/outputs.py
touch src/pyhearingai/extensions/__init__.py
```

### 2. Implement Core Components

1. **Processing Module**
   - Migrate audio conversion functionality
   - Add audio validation and normalization

2. **Models Module**
   - Create base classes for transcribers and diarizers
   - Implement OpenAI Whisper integration
   - Implement Pyannote integration
   - Create model registry

3. **Pipeline Module**
   - Implement main `transcribe` function
   - Add resource management
   - Implement progress tracking

4. **Outputs Module**
   - Create `TranscriptionResult` class
   - Implement output formatters (txt, srt, json)
   - Add format registry

5. **API Module**
   - Expose public API in `__init__.py`
   - Add configuration management
   - Create `pipeline_session` context manager

### 3. Test Migration

1. Create compatibility layer for existing code
2. Validate against end-to-end tests
3. Verify similarity between old and new output

### 4. Update Documentation

1. Update README with new API
2. Add usage examples
3. Document extension points

### 5. Final Release

1. Package for PyPI distribution
2. Update GitHub repository
3. Create release notes

## Implementation Timeline

1. **Week 1**: Create project structure and implement core components
2. **Week 2**: Implement model integrations and output formats
3. **Week 3**: Testing and refinement
4. **Week 4**: Documentation and release preparation

## Success Criteria

The migration will be considered successful when:
1. All current functionality is preserved
2. End-to-end tests pass with at least 80% similarity
3. Documentation is complete
4. The codebase adheres to Clean Architecture principles 