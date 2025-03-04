"""
Integration tests for the CLI functionality.

This module tests the CLI in a more realistic environment,
ensuring that it correctly interfaces with the transcription pipeline.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def test_audio_file():
    """Return a path to a test audio file."""
    return Path("example_audio.m4a")  # Use the example file in the repository


def test_cli_basic_integration(test_audio_file, tmp_path):
    """
    Test basic CLI functionality with a real audio file.

    Given: A valid audio file
    When: Running the CLI command
    Then: A transcript file is produced
    """
    # Skip if the file doesn't exist
    if not test_audio_file.exists():
        pytest.skip(f"Test audio file not found: {test_audio_file}")

    # Check if API keys are set
    openai_key = os.environ.get("OPENAI_API_KEY")
    huggingface_key = os.environ.get("HUGGINGFACE_API_KEY")

    if not openai_key or not huggingface_key:
        pytest.skip("API keys not set - skipping integration test")

    # Create temporary output file
    output_file = tmp_path / "output.txt"

    # Run the CLI command as a subprocess
    cmd = [
        sys.executable,
        "-m",
        "pyhearingai.cli",
        str(test_audio_file),
        "-o",
        str(output_file),
        "--verbose",
    ]

    try:
        # Use a timeout to prevent tests from hanging
        subprocess.run(cmd, check=True, timeout=180)

        # Verify that the output file exists and has content
        assert output_file.exists()
        content = output_file.read_text()
        assert len(content) > 0

    except subprocess.TimeoutExpired:
        pytest.fail("CLI command timed out")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"CLI command failed with exit code {e.returncode}")


def test_cli_source_option_integration(test_audio_file, tmp_path):
    """
    Test CLI source option.

    Given: A valid audio file
    When: Running the CLI command with the source option
    Then: A transcript file is produced
    """
    # Skip if the file doesn't exist
    if not test_audio_file.exists():
        pytest.skip(f"Test audio file not found: {test_audio_file}")

    # Check if API keys are set
    openai_key = os.environ.get("OPENAI_API_KEY")
    huggingface_key = os.environ.get("HUGGINGFACE_API_KEY")

    if not openai_key or not huggingface_key:
        pytest.skip("API keys not set - skipping integration test")

    # Create temporary output file
    output_file = tmp_path / "output.json"

    # Run the CLI command as a subprocess
    cmd = [
        sys.executable,
        "-m",
        "pyhearingai.cli",
        "-s",
        str(test_audio_file),
        "-o",
        str(output_file),
        "-f",
        "json",
    ]

    try:
        # Use a timeout to prevent tests from hanging
        subprocess.run(cmd, check=True, timeout=180)

        # Verify that the output file exists and has content
        assert output_file.exists()
        content = output_file.read_text()
        assert len(content) > 0
        assert content.startswith("{")  # Simple check for JSON format

    except subprocess.TimeoutExpired:
        pytest.fail("CLI command timed out")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"CLI command failed with exit code {e.returncode}")
