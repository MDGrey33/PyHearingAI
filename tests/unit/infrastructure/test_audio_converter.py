"""
Tests for the FFmpegAudioConverter class.

These tests verify the functionality of the audio conversion capabilities.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import ffmpeg
import pytest

from pyhearingai.infrastructure.audio_converter import FFmpegAudioConverter
