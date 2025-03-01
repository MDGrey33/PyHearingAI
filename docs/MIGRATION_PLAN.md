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

1. **[COMPLETED] Create Project Structure**
   - Set up the directory structure following Clean Architecture
   - Create empty modules for core entities and interfaces

2. **[COMPLETED] Implement Core Components**
   - Define domain models (`Segment`, `DiarizationSegment`, `TranscriptionResult`)
   - Create interface definitions for all components
   - Set up base extension registration system

3. **[COMPLETED] Implement Infrastructure Components**
   - Build concrete implementations of:
     - Audio conversion using FFmpeg
     - Transcription using OpenAI Whisper API
     - Diarization using Pyannote
     - Speaker assignment using GPT-4o
     - Output formatters for TXT, JSON, SRT, VTT, Markdown

4. **[COMPLETED] Implement Application Layer**
   - Develop the main transcription use case
   - Implement orchestration logic for the entire pipeline
   - Build the output generation functionality

5. **[COMPLETED] Build CLI Interface**
   - Create a command-line interface that mimics the functionality of the original implementation
   - Add configuration options for all components

6. **[COMPLETED] Test Migration**
   - Set up test infrastructure
   - Write tests for core components
   - Perform end-to-end testing against reference implementation

7. **[COMPLETED] Update Documentation**
   - Document the new architecture
   - Provide usage examples for the new API
   - Include extension development guidelines

8. **[IN PROGRESS] Prepare Final Release**
   - Package for distribution
   - Create installation guide
   - Finalize README and documentation

## Implementation Timeline

### Week 1 - Foundation and Core Domain
- [COMPLETED] Set up the project structure
- [COMPLETED] Define core entities and interfaces
- [COMPLETED] Create extension registration system

### Week 2 - Infrastructure Implementations
- [COMPLETED] Implement audio conversion adapter
- [COMPLETED] Implement transcription adapter for OpenAI Whisper
- [COMPLETED] Implement diarization adapter for Pyannote

### Week 3 - Application Layer and Additional Infrastructure
- [COMPLETED] Implement main application service
- [COMPLETED] Create output formatters
- [COMPLETED] Implement speaker assignment using GPT-4o

### Week 4 - Integration and Testing
- [COMPLETED] Connect all components
- [COMPLETED] Build CLI interface
- [COMPLETED] Perform end-to-end testing

### Week 5 - Documentation and Final Release
- [COMPLETED] Update all documentation with new architecture
- [IN PROGRESS] Prepare for distribution
- [PLANNED] Release final version

## Success Criteria

The migration will be considered successful when:
1. All current functionality is preserved
2. End-to-end tests pass with at least 80% similarity
3. Documentation is complete
4. The codebase adheres to Clean Architecture principles 