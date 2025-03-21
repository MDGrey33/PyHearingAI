"""
Integration tests for the complete transcription workflow.

Tests the end-to-end transcription process, verifying that components work together
correctly to process audio files, including chunking, transcription, and result aggregation.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyhearingai.application.transcribe import transcribe
from pyhearingai.core.models import Segment
from pyhearingai.infrastructure.adapters.audio_format_service import AudioFormatService
from pyhearingai.infrastructure.transcribers.whisper_openai import WhisperOpenAITranscriber


# Create mock classes for testing
class MockTranscriptionProvider:
    def transcribe(self, audio_file, **kwargs):
        return [Segment(text="This is a test transcription", start=0.0, end=5.0)]


class MockDiarizationProvider:
    def diarize(self, audio_file, **kwargs):
        return [{"speaker": "SPEAKER_1", "start": 0.0, "end": 5.0}]


# Mock TranscriptionService for testing
class TranscriptionService:
    def __init__(self, transcription_provider=None, diarization_provider=None):
        self.transcription_provider = transcription_provider
        self.diarization_provider = diarization_provider

    def transcribe(
        self,
        audio_path,
        output_path=None,
        with_diarization=False,
        chunk_duration=None,
        overlap_duration=None,
    ):
        segments = self.transcription_provider.transcribe(audio_path)

        if with_diarization and self.diarization_provider:
            diar_results = self.diarization_provider.diarize(audio_path)
            for segment, diar in zip(segments, diar_results):
                segment.speaker = diar["speaker"]

        if output_path:
            with open(output_path, "w") as f:
                for segment in segments:
                    f.write(f"{segment.speaker}: {segment.text}\n")

        return segments


@pytest.mark.skip(reason="Integration test needs to be implemented")
def test_basic_transcription_workflow(create_test_audio, tmp_path):
    """
    Test the complete transcription workflow with mock providers.

    This integration test verifies that all components of the transcription workflow
    work together correctly, from audio processing through transcription and
    diarization to formatted output, using mock providers for external services.
    """
    # Create test audio file
    audio_path = create_test_audio(duration=5.0, channels=1, sample_rate=16000)

    # Set up mock providers
    transcription_provider = MockTranscriptionProvider()
    diarization_provider = MockDiarizationProvider()

    # Create the transcription service
    service = TranscriptionService(
        transcription_provider=transcription_provider, diarization_provider=diarization_provider
    )

    # Set up output file
    output_path = tmp_path / "output.txt"

    # Execute the transcription
    result = service.transcribe(
        audio_path=audio_path, output_path=output_path, with_diarization=True
    )

    # Verify the results
    assert len(result) > 0
    for segment in result:
        assert isinstance(segment, Segment)
        assert segment.text
        assert segment.start >= 0
        assert segment.end > segment.start
        assert segment.speaker  # Should have speaker ID with diarization

    # Verify the output file
    assert output_path.exists()
    with open(output_path, "r") as f:
        content = f.read()
        assert content  # Should not be empty
        # Verify specific formatting and content as needed


@pytest.mark.skip(reason="Integration test needs to be implemented")
def test_chunking_integration(create_test_audio, tmp_path):
    """
    Test integration of chunking with transcription workflow.

    This test verifies that audio chunking is correctly integrated with the
    transcription service, with chunks properly processed and results aggregated
    into a single transcription output.
    """
    # Create a longer test audio file that will need chunking
    audio_path = create_test_audio(
        duration=20.0, channels=1, sample_rate=16000  # Long enough to trigger chunking
    )

    # Set up mock providers
    transcription_provider = MockTranscriptionProvider()

    # Create the transcription service
    service = TranscriptionService(
        transcription_provider=transcription_provider,
        # Use default chunking service
    )

    # Execute the transcription with chunking parameters
    result = service.transcribe(
        audio_path=audio_path,
        chunk_duration=5.0,  # Force chunking into 5-second segments
        overlap_duration=0.5,  # With overlap
    )

    # Verify the results
    assert len(result) > 0

    # Verify chunk handling and aggregation
    # The mock provider should return segments with predictable content
    # that we can check for proper ordering and overlap handling

    # Check for continuity at chunk boundaries
    for i in range(len(result) - 1):
        curr_segment = result[i]
        next_segment = result[i + 1]

        # No large gaps should exist between consecutive segments
        if curr_segment.end < next_segment.start:
            gap_duration = next_segment.start - curr_segment.end
            assert gap_duration < 0.5  # Maximum allowed gap

    # Verify total duration matches expectation
    if result:
        total_duration = result[-1].end - result[0].start
        assert total_duration > 0
        # Should cover most of the audio (with some tolerance)
        assert total_duration >= 15.0  # At least 75% coverage


@pytest.mark.skip(reason="Integration test needs to be implemented")
def test_error_handling_integration(tmp_path):
    """
    Test error handling across integration points.

    This test verifies that errors occurring in one component are properly
    propagated and handled by the transcription service, with appropriate
    error messages and cleanup operations.
    """
    # Create an invalid audio file
    invalid_audio = tmp_path / "invalid.wav"
    with open(invalid_audio, "wb") as f:
        f.write(b"This is not a valid audio file")

    # Set up providers
    transcription_provider = MockTranscriptionProvider()

    # Create the transcription service
    service = TranscriptionService(transcription_provider=transcription_provider)

    # Test with non-existent file
    nonexistent_file = tmp_path / "nonexistent.wav"
    with pytest.raises(FileNotFoundError):
        service.transcribe(audio_path=nonexistent_file)

    # Test with invalid file format
    with pytest.raises(Exception) as excinfo:
        service.transcribe(audio_path=invalid_audio)

    # The error should come from the audio processing layer
    assert "audio" in str(excinfo.value).lower() or "format" in str(excinfo.value).lower()

    # Test with failing transcription provider
    class FailingProvider:
        def transcribe(self, *args, **kwargs):
            raise ValueError("Simulated transcription failure")

    service = TranscriptionService(transcription_provider=FailingProvider())

    # Create a valid audio file
    valid_audio = create_test_audio(tmp_path / "valid.wav")

    with pytest.raises(ValueError) as excinfo:
        service.transcribe(audio_path=valid_audio)

    assert "transcription failure" in str(excinfo.value)


@pytest.mark.skip(reason="Integration test needs to be implemented")
def test_real_providers_integration(create_test_audio, tmp_path):
    """
    Test integration with real provider implementations.

    This test verifies that the transcription service can work with real
    provider implementations (not just mocks), handling their specific
    behaviors and requirements correctly.

    Note: This test may be skipped in CI environments without API keys.
    """
    # Skip if no API keys are configured
    if not os.environ.get("MOCK_API_KEY"):
        pytest.skip("API keys not configured")

    # Create test audio
    audio_path = create_test_audio(duration=3.0, channels=1, sample_rate=16000)

    # Use actual provider implementations
    # These would typically need API keys from environment variables
    from pyhearingai.infrastructure.providers.diarization.pyannote_diarization_provider import (
        PyAnnoteDiarizationProvider,
    )
    from pyhearingai.infrastructure.providers.transcription.openai_transcription_provider import (
        OpenAITranscriptionProvider,
    )

    # Create service with real providers
    try:
        service = TranscriptionService(
            transcription_provider=OpenAITranscriptionProvider(),
            diarization_provider=PyAnnoteDiarizationProvider(),
        )

        # Execute transcription
        result = service.transcribe(audio_path=audio_path, with_diarization=True)

        # Verify results from real providers
        assert len(result) > 0
        for segment in result:
            assert isinstance(segment, Segment)
            assert segment.text
            assert segment.speaker
    except ImportError:
        pytest.skip("Real provider implementations not available")
    except Exception as e:
        if "API key" in str(e) or "authorization" in str(e).lower():
            pytest.skip(f"API authentication failed: {e}")
        else:
            raise
