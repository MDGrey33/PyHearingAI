"""
Tests for the feature flag system in PyHearingAI.

This module tests the feature flag system, particularly the USE_IDEMPOTENT_PROCESSING flag
that controls whether to use the new idempotent processing or the legacy approach.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# Import the feature flag from config
from pyhearingai.config import USE_IDEMPOTENT_PROCESSING
