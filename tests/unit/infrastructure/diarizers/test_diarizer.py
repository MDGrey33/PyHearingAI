"""
Tests for the PyannoteDiarizer implementation.

These tests verify the behavior of the speaker diarization functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook

from pyhearingai.core.models import DiarizationSegment
from pyhearingai.infrastructure.diarizers.pyannote import PYANNOTE_AVAILABLE, PyannoteDiarizer
from tests.helpers import create_segment, patch_pyannote_pipeline
