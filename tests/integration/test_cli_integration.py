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
