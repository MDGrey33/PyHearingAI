"""
Functional tests for the CLI interface.

These tests verify the functionality of the command-line interface,
focusing on end-user experience and command behavior rather than
internal implementation details.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def cli_runner():
    """
    Fixture to run CLI commands and return their output.

    This creates a temporary environment for running CLI commands
    and provides a function to execute commands and capture results.
    """

    def run_command(args, input_str=None, expect_error=False):
        """
        Run a CLI command and return the result.

        Args:
            args: List of command arguments
            input_str: Optional input to provide to stdin
            expect_error: Whether to expect a non-zero exit code

        Returns:
            dict: Command result with stdout, stderr, and exit_code
        """
        # Ensure args is a list of strings
        if isinstance(args, str):
            args = args.split()

        # Prepare full command with python -m prefix
        full_command = [sys.executable, "-m", "pyhearingai"] + args

        # Run the command
        process = subprocess.run(
            full_command,
            input=input_str.encode("utf-8") if input_str else None,
            capture_output=True,
            text=True,
        )

        # Check for expected result
        if not expect_error and process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}")
            print(f"STDOUT: {process.stdout}")
            print(f"STDERR: {process.stderr}")

        return {"stdout": process.stdout, "stderr": process.stderr, "exit_code": process.returncode}

    return run_command


@pytest.mark.skip(reason="Functional test needs to be implemented")
def test_version_command(cli_runner):
    """
    Test the version command prints correct version information.

    Verifies that the CLI correctly reports its version number
    and that the format follows semantic versioning.
    """
    # Run the version command
    result = cli_runner(["--version"])

    # Verify the command succeeded
    assert result["exit_code"] == 0

    # Check that version string is present in output
    version_output = result["stdout"].strip()
    assert "version" in version_output.lower()

    # Verify version follows semantic versioning (e.g., 1.2.3)
    version_line = version_output.splitlines()[0]
    version_str = version_line.split()[-1].strip()

    # Basic check for semver pattern
    import re

    assert re.match(r"\d+\.\d+\.\d+", version_str)


@pytest.mark.skip(reason="Functional test needs to be implemented")
def test_help_command(cli_runner):
    """
    Test the help command shows usage information.

    Verifies that the help command produces formatted documentation
    for all supported commands and options.
    """
    # Run the help command
    result = cli_runner(["--help"])

    # Verify the command succeeded
    assert result["exit_code"] == 0

    # Check that expected sections are in the help output
    help_output = result["stdout"]
    assert "usage" in help_output.lower()
    assert "commands" in help_output.lower()
    assert "options" in help_output.lower()

    # Check for key commands in help output
    assert "transcribe" in help_output.lower()

    # Test help for specific command
    result = cli_runner(["transcribe", "--help"])
    assert result["exit_code"] == 0
    assert "transcribe" in result["stdout"].lower()


@pytest.mark.skip(reason="Functional test needs to be implemented")
def test_transcribe_command_basic(cli_runner, create_test_audio, tmp_path):
    """
    Test basic transcription command functionality.

    Verifies that the transcribe command correctly processes an audio
    file and generates a transcript with default settings.
    """
    # Create test audio file
    audio_path = create_test_audio(duration=3.0, output_dir=tmp_path)

    # Set up output path
    output_path = tmp_path / "output.txt"

    # Run transcribe command
    result = cli_runner(
        [
            "transcribe",
            str(audio_path),
            "--output",
            str(output_path),
            "--mock",  # Use mock provider for testing
        ]
    )

    # Verify command succeeded
    assert result["exit_code"] == 0

    # Check for progress indication in output
    assert "progress" in result["stdout"].lower() or "%" in result["stdout"]

    # Verify output file was created
    assert output_path.exists()

    # Check content of output file
    with open(output_path, "r") as f:
        content = f.read()
        assert content  # Should not be empty
        assert len(content) > 0


@pytest.mark.skip(reason="Functional test needs to be implemented")
def test_transcribe_with_diarization(cli_runner, create_multi_speaker_audio, tmp_path):
    """
    Test transcription with speaker diarization.

    Verifies that the transcribe command correctly identifies and labels
    different speakers when diarization is enabled.
    """
    # Create test audio with multiple speakers
    audio_path = create_multi_speaker_audio(duration=10.0, num_speakers=2, output_dir=tmp_path)

    # Set up output path
    output_path = tmp_path / "output.txt"

    # Run transcribe command with diarization
    result = cli_runner(
        [
            "transcribe",
            str(audio_path),
            "--output",
            str(output_path),
            "--diarize",
            "--mock",  # Use mock provider for testing
        ]
    )

    # Verify command succeeded
    assert result["exit_code"] == 0

    # Check that output mentions speakers
    assert "speaker" in result["stdout"].lower()

    # Verify output file has speaker labels
    with open(output_path, "r") as f:
        content = f.read()
        assert "SPEAKER" in content or "Speaker" in content


@pytest.mark.skip(reason="Functional test needs to be implemented")
def test_output_format_options(cli_runner, create_test_audio, tmp_path):
    """
    Test different output format options.

    Verifies that the transcribe command can generate output in different
    formats (txt, json, srt) as specified by the user.
    """
    # Create test audio file
    audio_path = create_test_audio(duration=3.0, output_dir=tmp_path)

    # Test different output formats
    formats = ["txt", "json", "srt"]

    for fmt in formats:
        # Set up output path
        output_path = tmp_path / f"output.{fmt}"

        # Run transcribe command with format
        result = cli_runner(
            [
                "transcribe",
                str(audio_path),
                "--output",
                str(output_path),
                "--format",
                fmt,
                "--mock",  # Use mock provider for testing
            ]
        )

        # Verify command succeeded
        assert result["exit_code"] == 0

        # Verify output file was created
        assert output_path.exists()

        # Validate format-specific content
        with open(output_path, "r") as f:
            content = f.read()

            if fmt == "json":
                # Should be valid JSON
                try:
                    json_data = json.loads(content)
                    assert isinstance(json_data, list)
                    assert len(json_data) > 0
                    assert "text" in json_data[0]
                except json.JSONDecodeError:
                    pytest.fail("JSON output is not valid")

            elif fmt == "srt":
                # Should have SRT format elements
                assert "-->" in content
                assert "00:" in content

            elif fmt == "txt":
                # Should be plain text
                assert content.strip()


@pytest.mark.skip(reason="Functional test needs to be implemented")
def test_error_handling(cli_runner, tmp_path):
    """
    Test CLI error handling.

    Verifies that the CLI appropriately handles error conditions
    such as missing files, invalid arguments, and processing errors,
    providing clear error messages to the user.
    """
    # Test with non-existent file
    nonexistent_file = tmp_path / "nonexistent.wav"

    result = cli_runner(["transcribe", str(nonexistent_file)], expect_error=True)

    # Verify command failed with appropriate error
    assert result["exit_code"] != 0
    assert (
        "file not found" in result["stderr"].lower() or "no such file" in result["stderr"].lower()
    )

    # Test with invalid format
    audio_path = create_test_audio(duration=1.0, output_dir=tmp_path)

    result = cli_runner(
        ["transcribe", str(audio_path), "--format", "invalid_format"], expect_error=True
    )

    # Verify command failed with appropriate error
    assert result["exit_code"] != 0
    assert "format" in result["stderr"].lower()

    # Test with invalid provider
    result = cli_runner(
        ["transcribe", str(audio_path), "--provider", "invalid_provider"], expect_error=True
    )

    # Verify command failed with appropriate error
    assert result["exit_code"] != 0
    assert "provider" in result["stderr"].lower()


@pytest.mark.skip(reason="Functional test needs to be implemented")
def test_verbose_mode(cli_runner, create_test_audio, tmp_path):
    """
    Test verbose mode output.

    Verifies that the verbose mode provides additional detailed
    information about the transcription process.
    """
    # Create test audio file
    audio_path = create_test_audio(duration=3.0, output_dir=tmp_path)

    # Run transcribe command with verbose flag
    result = cli_runner(
        ["transcribe", str(audio_path), "--verbose", "--mock"]  # Use mock provider for testing
    )

    # Verify command succeeded
    assert result["exit_code"] == 0

    # Check for detailed logging in verbose output
    verbose_output = result["stdout"]
    assert "debug" in verbose_output.lower() or "info" in verbose_output.lower()

    # Should contain more detailed information than non-verbose mode
    normal_result = cli_runner(["transcribe", str(audio_path), "--mock"])

    # Verbose output should be more detailed
    assert len(verbose_output) > len(normal_result["stdout"])

    # Check for specific verbose details
    assert "configuration" in verbose_output.lower() or "settings" in verbose_output.lower()
    assert "duration" in verbose_output.lower() or "time" in verbose_output.lower()
