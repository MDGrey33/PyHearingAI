# Changelog

All notable changes to PyHearingAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-03-05

### Added

#### Core Functionality
- Audio transcription engine with high accuracy text-to-speech conversion
- Advanced speaker diarization for multi-speaker conversations
- GPT-based speaker assignment for improved speaker identification
- Multiple formats support for transcription output (JSON, TEXT, SRT)
- Segment-level timestamps and speaker identification
- Support for multiple audio formats (WAV, MP3, M4A, etc.)

#### User Interfaces
- Command-line interface (CLI) for easy access to transcription features
- Programmatic API for integration into other Python applications
- Simple example scripts demonstrating library usage

#### Model Integrations
- OpenAI API integration for transcription and speaker assignment
- SpeechBrain integration for speech recognition
- Hugging Face models support
- Pyannote.audio integration for speaker diarization

#### Developer Features
- Comprehensive configuration options via environment variables
- Debug logging capabilities
- Error handling and graceful degradation
- Structured output format with segments and speaker information
- Multiple output file formats supported

### Changed
- Initial release, no changes to document

### Fixed
- Initial release, no fixes to document

## [Unreleased]

### Added
- Test fixtures directory with synthetic test audio for integration tests
- Comprehensive unit test coverage for DiarizationService
- Integration tests for DiarizationService with real audio files

### Changed
- Improved DiarizationService to use lazy loading for diarizer initialization
- Enhanced error handling in audio chunk processing
- Improved logging with consistent level usage and more informative messages
- Simplified parallel processing using ThreadPoolExecutor
- Fixed handling of string chunk IDs in both sequential and parallel processing

### Fixed
- Fixed error in DiarizationService when processing string chunk IDs
- Improved reliability of parallel processing for audio chunks
- Fixed repository access with proper Path object handling
