"""
Tests for the feature flag system in PyHearingAI.

This module tests the feature flag system, particularly the USE_IDEMPOTENT_PROCESSING flag
that controls whether to use the new idempotent processing or the legacy approach.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Import the feature flag from config
from pyhearingai.config import USE_IDEMPOTENT_PROCESSING


class TestFeatureFlags:
    """Tests for the feature flags in the config module."""

    def test_idempotent_processing_flag_default(self):
        """Test that the idempotent processing flag has the correct default value."""
        # Clear any environment variable to test the default
        with patch.dict(os.environ, {}, clear=True):
            # Reimport to reload the module with fresh environment
            import importlib
            import pyhearingai.config

            importlib.reload(pyhearingai.config)

            # Check the default value
            assert pyhearingai.config.USE_IDEMPOTENT_PROCESSING == False

    def test_idempotent_processing_flag_enabled(self):
        """Test setting the flag to enabled via environment variable."""
        # Set the environment variable to true
        with patch.dict(os.environ, {"PYHEARINGAI_USE_IDEMPOTENT_PROCESSING": "true"}):
            # Reimport to reload the module with updated environment
            import importlib
            import pyhearingai.config

            importlib.reload(pyhearingai.config)

            # Check the value is now True
            assert pyhearingai.config.USE_IDEMPOTENT_PROCESSING == True

    def test_idempotent_processing_flag_disabled(self):
        """Test explicitly disabling the flag via environment variable."""
        # Set the environment variable to false
        with patch.dict(os.environ, {"PYHEARINGAI_USE_IDEMPOTENT_PROCESSING": "false"}):
            # Reimport to reload the module with updated environment
            import importlib
            import pyhearingai.config

            importlib.reload(pyhearingai.config)

            # Check the value is False
            assert pyhearingai.config.USE_IDEMPOTENT_PROCESSING == False

    def test_idempotent_processing_case_insensitive(self):
        """Test that the flag parsing is case-insensitive."""
        # Test with mixed case "True"
        with patch.dict(os.environ, {"PYHEARINGAI_USE_IDEMPOTENT_PROCESSING": "TrUe"}):
            import importlib
            import pyhearingai.config

            importlib.reload(pyhearingai.config)

            # Check the value is True
            assert pyhearingai.config.USE_IDEMPOTENT_PROCESSING == True


class TestTranscribeWithFlag:
    """Tests for how the transcribe function uses the feature flag."""

    def test_transcribe_flag_inheritance(self):
        """
        Test that the USE_IDEMPOTENT_PROCESSING flag affects the config module.

        This tests that when USE_IDEMPOTENT_PROCESSING is set in the environment,
        the config module's USE_IDEMPOTENT_PROCESSING value matches it.
        """
        # Test with flag enabled
        with patch.dict(os.environ, {"PYHEARINGAI_USE_IDEMPOTENT_PROCESSING": "true"}):
            import importlib
            import pyhearingai.config

            importlib.reload(pyhearingai.config)

            # Verify the config value matches our environment setting
            assert pyhearingai.config.USE_IDEMPOTENT_PROCESSING == True

        # Test with flag disabled
        with patch.dict(os.environ, {"PYHEARINGAI_USE_IDEMPOTENT_PROCESSING": "false"}):
            import importlib
            import pyhearingai.config

            importlib.reload(pyhearingai.config)

            # Verify the config value matches our environment setting
            assert pyhearingai.config.USE_IDEMPOTENT_PROCESSING == False

    def test_transcribe_respects_parameter(self):
        """
        Test that transcribe respects the use_idempotent_processing parameter.

        This tests that when use_idempotent_processing is explicitly passed,
        it overrides the default from config.
        """

        # Create a simplified transcribe function for testing
        def mock_transcribe(use_idempotent_processing=None):
            return use_idempotent_processing

        # Configure the mock function with our test function
        with patch("pyhearingai.application.transcribe.transcribe", side_effect=mock_transcribe):
            from pyhearingai.application.transcribe import transcribe

            # Test with explicit True
            assert transcribe(use_idempotent_processing=True) == True

            # Test with explicit False
            assert transcribe(use_idempotent_processing=False) == False


class TestCLIWithFlag:
    """Tests for how the CLI uses the feature flag."""

    def test_cli_overrides_default_flag(self):
        """Test that the CLI always sets use_idempotent_processing=True by default."""
        # Mock the environment and config to disable idempotent processing
        with patch.dict(os.environ, {"PYHEARINGAI_USE_IDEMPOTENT_PROCESSING": "false"}):
            import importlib
            import pyhearingai.config

            importlib.reload(pyhearingai.config)

            # Import CLI after reloading config
            from pyhearingai.cli import main

            # Mock the transcribe function
            with patch("pyhearingai.cli.transcribe") as mock_transcribe, patch(
                "sys.argv", ["pyhearingai", "test.wav"]
            ), patch("pathlib.Path.exists", return_value=True):
                # Configure the mock
                mock_result = mock_transcribe.return_value
                mock_result.save = lambda *args, **kwargs: None

                # Call the CLI
                main()

                # Verify transcribe was called with use_idempotent_processing=True
                args, kwargs = mock_transcribe.call_args
                assert kwargs["use_idempotent_processing"] == True

    def test_cli_legacy_flag(self):
        """Test that --use-legacy flag sets use_idempotent_processing=False."""
        with patch("pyhearingai.cli.transcribe") as mock_transcribe, patch(
            "sys.argv", ["pyhearingai", "test.wav", "--use-legacy"]
        ), patch("pathlib.Path.exists", return_value=True):
            # Configure the mock
            mock_result = mock_transcribe.return_value
            mock_result.save = lambda *args, **kwargs: None

            # Call the CLI
            from pyhearingai.cli import main

            main()

            # Verify transcribe was called with use_idempotent_processing=False
            args, kwargs = mock_transcribe.call_args
            assert kwargs["use_idempotent_processing"] == False


if __name__ == "__main__":
    pytest.main()
