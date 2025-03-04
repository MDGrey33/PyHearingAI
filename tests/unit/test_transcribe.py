"""
Unit tests for the transcribe module.

Tests the main transcription service including:
- Basic transcription flow
- Progress callback functionality
- Error handling
- Output format handling
- Logging configuration
"""

import logging
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

from pyhearingai.application.transcribe import transcribe
from pyhearingai.core.models import Segment, TranscriptionResult


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before and after each test."""
    # Store original logging level and handlers
    root = logging.getLogger()
    original_level = root.level
    original_handlers = root.handlers[:]

    # Remove all handlers and reset level
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.setLevel(logging.INFO)

    yield

    # Restore original state
    root.setLevel(original_level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in original_handlers:
        root.addHandler(handler)


@pytest.fixture
def mock_providers():
    """
    Mock all provider instances (converter, transcriber, diarizer, speaker_assigner).

    Returns:
        dict: Dictionary containing all mocked providers
    """
    with patch("pyhearingai.application.transcribe.get_converter") as mock_get_converter, patch(
        "pyhearingai.application.transcribe.get_transcriber"
    ) as mock_get_transcriber, patch(
        "pyhearingai.application.transcribe.get_diarizer"
    ) as mock_get_diarizer, patch(
        "pyhearingai.application.transcribe.get_speaker_assigner"
    ) as mock_get_assigner:
        # Create mock instances
        converter = Mock()
        converter.convert.return_value = Path("converted.wav")

        transcriber = Mock()
        transcriber.transcribe.return_value = [
            Segment(text="Hello", start=0.0, end=1.0),
            Segment(text="World", start=1.0, end=2.0),
        ]

        diarizer = Mock()
        diarizer.diarize.return_value = [
            Segment(text="", start=0.0, end=1.0, speaker_id="SPEAKER_01"),
            Segment(text="", start=1.0, end=2.0, speaker_id="SPEAKER_02"),
        ]

        assigner = Mock()
        assigner.assign_speakers.return_value = [
            Segment(text="Hello", start=0.0, end=1.0, speaker_id="SPEAKER_01"),
            Segment(text="World", start=1.0, end=2.0, speaker_id="SPEAKER_02"),
        ]

        # Set up the mocks to return our instances
        mock_get_converter.return_value = converter
        mock_get_transcriber.return_value = transcriber
        mock_get_diarizer.return_value = diarizer
        mock_get_assigner.return_value = assigner

        yield {
            "converter": converter,
            "transcriber": transcriber,
            "diarizer": diarizer,
            "assigner": assigner,
            "get_converter": mock_get_converter,
            "get_transcriber": mock_get_transcriber,
            "get_diarizer": mock_get_diarizer,
            "get_assigner": mock_get_assigner,
        }


def test_basic_transcription(mock_providers, tmp_path):
    """
    Test basic transcription flow.

    Given: An audio file
    When: Transcribing with default settings
    Then: All components are called in sequence and result is returned
    """
    # Arrange
    input_file = tmp_path / "test.wav"
    input_file.touch()

    # Act
    result = transcribe(input_file)

    # Assert
    assert isinstance(result, TranscriptionResult)
    assert len(result.segments) == 2
    assert result.segments[0].text == "Hello"
    assert result.segments[0].speaker_id == "SPEAKER_01"

    # Verify provider calls
    mock_providers["converter"].convert.assert_called_once()
    mock_providers["transcriber"].transcribe.assert_called_once()
    mock_providers["diarizer"].diarize.assert_called_once()
    mock_providers["assigner"].assign_speakers.assert_called_once()


def test_progress_callback(mock_providers, tmp_path):
    """
    Test progress callback functionality.

    Given: A progress callback function
    When: Transcribing an audio file
    Then: The callback is called with correct progress values
    """
    # Arrange
    input_file = tmp_path / "test.wav"
    input_file.touch()
    progress_callback = Mock()

    # Act
    transcribe(input_file, progress_callback=progress_callback)

    # Assert
    expected_calls = [
        call(0.0, "Starting transcription process"),
        call(0.1, "Audio conversion complete"),
        call(0.5, "Transcription complete"),
        call(0.8, "Diarization complete"),
        call(1.0, "Processing complete"),
    ]
    progress_callback.assert_has_calls(expected_calls)


def test_file_not_found():
    """
    Test error handling for non-existent files.

    Given: A non-existent audio file path
    When: Attempting to transcribe
    Then: FileNotFoundError is raised
    """
    # Act & Assert
    with pytest.raises(FileNotFoundError) as exc_info:
        transcribe("nonexistent.wav")
    assert "Audio file not found" in str(exc_info.value)


def test_output_format(mock_providers, tmp_path):
    """
    Test output format handling.

    Given: An output format specification
    When: Transcribing an audio file
    Then: The result is saved in the specified format
    """
    # Arrange
    input_file = tmp_path / "test.wav"
    input_file.touch()

    with patch("pyhearingai.application.outputs.save_transcript") as mock_save:
        # Act
        transcribe(input_file, output_format="txt")

        # Assert
        mock_save.assert_called_once()
        args = mock_save.call_args[0]
        assert isinstance(args[0], TranscriptionResult)
        assert args[1] == input_file.with_suffix(".txt")
        assert args[2] == "txt"


def test_verbose_logging(mock_providers, tmp_path):
    """
    Test verbose logging configuration.

    Given: Verbose mode enabled
    When: Transcribing an audio file
    Then: Logging is set to DEBUG level
    """
    # Arrange
    input_file = tmp_path / "test.wav"
    input_file.touch()

    # Act
    with patch("logging.basicConfig") as mock_basic_config:
        transcribe(input_file, verbose=True)

        # Assert
        mock_basic_config.assert_called_once_with(
            level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


def test_str_path_input(mock_providers, tmp_path):
    """
    Test string path input handling.

    Given: A string path instead of Path object
    When: Transcribing an audio file
    Then: The path is correctly converted and processed
    """
    # Arrange
    input_file = tmp_path / "test.wav"
    input_file.touch()

    # Act
    result = transcribe(str(input_file))

    # Assert
    assert isinstance(result.audio_path, Path)
    mock_providers["converter"].convert.assert_called_once()


def test_custom_providers(mock_providers, tmp_path):
    """
    Test custom provider selection.

    Given: Custom transcriber and diarizer names
    When: Transcribing an audio file
    Then: The correct providers are retrieved and used
    """
    # Arrange
    input_file = tmp_path / "test.wav"
    input_file.touch()

    # Act
    transcribe(input_file, transcriber="custom_transcriber", diarizer="custom_diarizer")

    # Assert
    mock_providers["get_transcriber"].assert_called_with("custom_transcriber")
    mock_providers["get_diarizer"].assert_called_with("custom_diarizer")


def test_kwargs_forwarding(mock_providers, tmp_path):
    """
    Test forwarding of additional kwargs to providers.

    Given: Additional keyword arguments
    When: Transcribing an audio file
    Then: The kwargs are forwarded to all providers
    """
    # Arrange
    input_file = tmp_path / "test.wav"
    input_file.touch()
    kwargs = {"option1": "value1", "option2": "value2"}

    # Act
    result = transcribe(input_file, **kwargs)

    # Assert
    mock_providers["converter"].convert.assert_called_with(input_file, **kwargs)
    mock_providers["transcriber"].transcribe.assert_called_with(Path("converted.wav"), **kwargs)
    mock_providers["diarizer"].diarize.assert_called_with(Path("converted.wav"), **kwargs)
    assert result.metadata["options"] == kwargs


def test_api_key_sanitization(mock_providers, tmp_path):
    """
    Test sanitization of API keys in metadata.

    Given: Kwargs containing API keys
    When: Transcribing an audio file
    Then: The API keys are not included in the result metadata
    """
    # Arrange
    input_file = tmp_path / "test.wav"
    input_file.touch()
    kwargs = {
        "api_key": "sk-1234567890abcdef",
        "huggingface_api_key": "hf_1234567890abcdef",
        "safe_option": "this_should_remain",
        "speaker_assigner_options": {"api_key": "sk-nested1234567890", "model": "gpt-4o"},
    }

    # Act
    result = transcribe(input_file, **kwargs)

    # Assert
    # Verify all kwargs are forwarded to providers (including API keys)
    mock_providers["converter"].convert.assert_called_with(input_file, **kwargs)
    mock_providers["transcriber"].transcribe.assert_called_with(Path("converted.wav"), **kwargs)
    mock_providers["diarizer"].diarize.assert_called_with(Path("converted.wav"), **kwargs)

    # Verify API keys are sanitized in metadata
    assert "options" in result.metadata
    metadata_options = result.metadata["options"]
    assert "api_key" not in metadata_options
    assert "huggingface_api_key" not in metadata_options
    assert metadata_options["safe_option"] == "this_should_remain"

    # Check nested options
    assert "speaker_assigner_options" in metadata_options
    assert "api_key" not in metadata_options["speaker_assigner_options"]
    assert metadata_options["speaker_assigner_options"]["model"] == "gpt-4o"
