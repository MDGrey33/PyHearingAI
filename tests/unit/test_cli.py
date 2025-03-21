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
