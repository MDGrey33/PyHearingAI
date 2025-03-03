"""
Unit tests for PyHearingAI CLI functionality.

Tests command-line interface functionality including:
- Argument parsing
- Command execution
- Error handling
- Output formatting
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pyhearingai.cli import main
from pyhearingai.core.models import Segment, TranscriptionResult


@pytest.fixture
def mock_transcribe(formatter_segments_with_speakers):
    """
    Mock the transcribe function to return test data.

    Given: A set of test segments from our test infrastructure
    When: The transcribe function is called
    Then: Returns a TranscriptionResult with the test segments

    Args:
        formatter_segments_with_speakers: Test segments from our fixtures
    """
    with patch("pyhearingai.cli.transcribe") as mock:
        # Use our existing test data infrastructure
        result = TranscriptionResult(
            segments=formatter_segments_with_speakers,
            audio_path=Path("test.wav"),
            metadata={"test": True},
        )
        mock.return_value = result
        yield mock


@pytest.fixture
def cli_args(tmp_path):
    """
    Provide common CLI argument combinations for testing.

    Returns:
        dict: Different sets of CLI arguments for testing
    """
    input_file = tmp_path / "test.wav"
    input_file.touch()
    output_file = tmp_path / "output.txt"

    return {
        "basic": [str(input_file)],
        "with_output": [str(input_file), "-o", str(output_file)],
        "with_format": [str(input_file), "-f", "json"],
        "with_diarizer": [str(input_file), "-d", "pyannote"],
        "with_transcriber": [str(input_file), "-t", "whisper_openai"],
        "verbose": [str(input_file), "--verbose"],
        "full": [
            str(input_file),
            "-o",
            str(output_file),
            "-f",
            "json",
            "-d",
            "pyannote",
            "-t",
            "whisper_openai",
            "--verbose",
        ],
    }


def test_cli_basic(cli_args, mock_transcribe):
    """
    Test basic CLI functionality with minimal arguments.

    Given: An audio file path
    When: Running the CLI with basic arguments
    Then: The audio is transcribed and saved with default settings
    """
    # Arrange
    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["basic"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        mock_transcribe.assert_called_once()


def test_cli_with_output(cli_args, mock_transcribe, tmp_path):
    """
    Test CLI with custom output path.

    Given: An audio file and output path
    When: Running the CLI with output specification
    Then: The transcription is saved to the specified path
    """
    # Arrange
    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["with_output"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        output_file = tmp_path / "output.txt"
        assert output_file.exists()


@pytest.mark.parametrize(
    "format_name",
    [
        pytest.param("json", id="json_format"),
        pytest.param("srt", id="srt_format"),
        pytest.param("vtt", id="vtt_format"),
        pytest.param("md", id="markdown_format"),
        pytest.param("txt", id="text_format"),
    ],
)
def test_cli_output_formats(cli_args, mock_transcribe, tmp_path, format_name):
    """
    Test CLI with different output formats.

    Given: An audio file and output format
    When: Running the CLI with format specification
    Then: The transcription is saved in the correct format

    Args:
        format_name: The output format to test
    """
    # Arrange
    output_file = tmp_path / f"output.{format_name}"
    args = [cli_args["basic"][0], "-o", str(output_file), "-f", format_name]

    with patch.object(sys, "argv", ["pyhearingai"] + args):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        assert output_file.exists()


def test_cli_missing_file():
    """
    Test CLI with non-existent file.

    Given: A non-existent audio file path
    When: Running the CLI
    Then: The program exits with an error
    """
    # Arrange
    with patch.object(sys, "argv", ["pyhearingai", "nonexistent.wav"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 1


def test_cli_default_output(cli_args, mock_transcribe, tmp_path):
    """
    Test CLI with default output path.

    Given: An audio file without specifying output path
    When: Running the CLI
    Then: The transcription is saved with default path based on input file
    """
    # Arrange
    input_file = Path(cli_args["basic"][0])
    expected_output = input_file.with_suffix(".txt")

    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["basic"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        assert expected_output.exists()


def test_cli_error_verbose(cli_args, mock_transcribe, capsys):
    """
    Test CLI error handling in verbose mode.

    Given: A transcription error and verbose mode enabled
    When: Running the CLI
    Then: The error is printed with full traceback
    """
    # Arrange
    mock_transcribe.side_effect = Exception("Test error")

    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["verbose"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error: Test error" in captured.err
        assert "Traceback" in captured.err


def test_cli_transcriber_diarizer_combo(cli_args, mock_transcribe):
    """
    Test CLI with different transcriber and diarizer combinations.

    Given: Custom transcriber and diarizer options
    When: Running the CLI
    Then: The options are correctly passed to the transcribe function
    """
    # Arrange
    args = [cli_args["basic"][0], "-t", "custom_transcriber", "-d", "custom_diarizer"]

    with patch.object(sys, "argv", ["pyhearingai"] + args):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        mock_transcribe.assert_called_once_with(
            audio_path=cli_args["basic"][0],
            transcriber="custom_transcriber",
            diarizer="custom_diarizer",
            verbose=False,
        )
