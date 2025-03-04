"""
Unit tests for PyHearingAI CLI functionality.

Tests command-line interface functionality including:
- Argument parsing
- Command execution
- Error handling
- Output formatting
- API key handling
"""

import os
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
        "with_source": ["-s", str(input_file)],
        "with_output": [str(input_file), "-o", str(output_file)],
        "with_format": [str(input_file), "-f", "json"],
        "with_openai_key": [str(input_file), "--openai-key", "test_openai_key"],
        "with_huggingface_key": [str(input_file), "--huggingface-key", "test_hf_key"],
        "verbose": [str(input_file), "--verbose"],
        "full": [
            "-s",
            str(input_file),
            "-o",
            str(output_file),
            "-f",
            "json",
            "--openai-key",
            "test_openai_key",
            "--huggingface-key",
            "test_hf_key",
            "--verbose",
        ],
        "conflicting": [str(input_file), "-s", str(input_file)],  # Both positional and -s
        "invalid_format": [str(input_file), "-f", "invalid_format"],
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


def test_cli_with_source(cli_args, mock_transcribe):
    """
    Test CLI with source flag.

    Given: An audio file specified with the source flag
    When: Running the CLI
    Then: The audio is transcribed correctly
    """
    # Arrange
    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["with_source"]):
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


def test_cli_with_openai_key(cli_args, mock_transcribe, monkeypatch):
    """
    Test CLI with OpenAI API key.

    Given: An OpenAI API key provided via command line
    When: Running the CLI
    Then: The key is passed to the transcribe function and set as an environment variable
    """
    # Arrange
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["with_openai_key"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        # Check that the key was passed to the function
        call_kwargs = mock_transcribe.call_args[1]
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"] == "test_openai_key"
        # Check that it was also set as an environment variable
        assert os.environ.get("OPENAI_API_KEY") == "test_openai_key"


def test_cli_with_huggingface_key(cli_args, mock_transcribe, monkeypatch):
    """
    Test CLI with Hugging Face API key.

    Given: A Hugging Face API key provided via command line
    When: Running the CLI
    Then: The key is passed to the transcribe function and set as an environment variable
    """
    # Arrange
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["with_huggingface_key"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        # Check that the key was passed to the function
        call_kwargs = mock_transcribe.call_args[1]
        assert "huggingface_api_key" in call_kwargs
        assert call_kwargs["huggingface_api_key"] == "test_hf_key"
        # Check that it was also set as an environment variable
        assert os.environ.get("HUGGINGFACE_API_KEY") == "test_hf_key"


def test_cli_env_api_keys(cli_args, mock_transcribe, monkeypatch):
    """
    Test CLI with API keys from environment variables.

    Given: API keys set as environment variables
    When: Running the CLI without key arguments
    Then: The environment variable keys are used
    """
    # Arrange
    monkeypatch.setenv("OPENAI_API_KEY", "env_openai_key")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "env_hf_key")

    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["basic"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        # Check that the keys were passed to the function
        call_kwargs = mock_transcribe.call_args[1]
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"] == "env_openai_key"
        assert "huggingface_api_key" in call_kwargs
        assert call_kwargs["huggingface_api_key"] == "env_hf_key"


def test_cli_argument_precedence(cli_args, mock_transcribe, monkeypatch):
    """
    Test CLI argument precedence.

    Given: API keys set as both environment variables and command line arguments
    When: Running the CLI
    Then: The command line arguments take precedence
    """
    # Arrange
    monkeypatch.setenv("OPENAI_API_KEY", "env_openai_key")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "env_hf_key")

    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["with_openai_key"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        # Check that the command line key was used
        call_kwargs = mock_transcribe.call_args[1]
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"] == "test_openai_key"


def test_cli_missing_api_keys_warning(cli_args, mock_transcribe, monkeypatch, capsys):
    """
    Test CLI warning for missing API keys.

    Given: No API keys provided
    When: Running the CLI
    Then: Warnings are printed about missing keys
    """
    # Arrange
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["basic"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "OpenAI API key not found" in captured.err
        assert "Hugging Face API key not found" in captured.err


def test_cli_full_command(cli_args, mock_transcribe, monkeypatch, tmp_path):
    """
    Test CLI with all options.

    Given: A command with all available options
    When: Running the CLI
    Then: All options are processed correctly
    """
    # Arrange
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["full"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        output_file = tmp_path / "output.txt"
        assert output_file.exists()

        # Check API keys
        call_kwargs = mock_transcribe.call_args[1]
        assert call_kwargs["api_key"] == "test_openai_key"
        assert call_kwargs["huggingface_api_key"] == "test_hf_key"
        assert call_kwargs["verbose"] is True


def test_cli_conflicting_args(cli_args, capsys):
    """
    Test CLI with conflicting arguments.

    Given: CLI arguments with both positional argument and -s/--source
    When: Running the CLI
    Then: The parser shows an error and exits
    """
    # Arrange
    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["conflicting"]):
        with patch.object(sys, "exit") as mock_exit:
            # Act
            # Patch sys.exit to prevent actual exit in test
            mock_exit.side_effect = SystemExit
            with pytest.raises(SystemExit):
                main()

            # Assert
            captured = capsys.readouterr()
            assert "error" in captured.err.lower()
            assert "not allowed with argument" in captured.err.lower()


def test_cli_invalid_format(cli_args, capsys):
    """
    Test CLI with invalid format.

    Given: CLI arguments with an invalid output format
    When: Running the CLI
    Then: The parser shows an error and exits
    """
    # Arrange
    with patch.object(sys, "argv", ["pyhearingai"] + cli_args["invalid_format"]):
        with patch.object(sys, "exit") as mock_exit:
            # Act
            # Patch sys.exit to prevent actual exit in test
            mock_exit.side_effect = SystemExit
            with pytest.raises(SystemExit):
                main()

            # Assert
            captured = capsys.readouterr()
            assert "error" in captured.err.lower()
            assert "invalid choice" in captured.err.lower()
