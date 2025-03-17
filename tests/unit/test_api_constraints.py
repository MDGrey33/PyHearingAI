"""
Tests for the ApiSizeLimitPolicy and related domain models.
"""

import os
import tempfile
from pathlib import Path

import pytest

from pyhearingai.core.domain.api_constraints import ApiProvider, ApiSizeLimit, ApiSizeLimitPolicy


class TestApiSizeLimit:
    """Tests for the ApiSizeLimit class."""

    def test_default_values(self):
        """Test default values for ApiSizeLimit."""
        limit = ApiSizeLimit(
            provider=ApiProvider.OPENAI_WHISPER, max_file_size_bytes=25 * 1024 * 1024  # 25MB
        )

        assert limit.provider == ApiProvider.OPENAI_WHISPER
        assert limit.max_file_size_bytes == 25 * 1024 * 1024
        assert limit.max_duration_seconds is None
        assert "wav" in limit.supported_formats
        assert "mp3" in limit.supported_formats

    def test_check_file_size_nonexistent_file(self):
        """Test check_file_size with a nonexistent file."""
        limit = ApiSizeLimit(
            provider=ApiProvider.OPENAI_WHISPER, max_file_size_bytes=25 * 1024 * 1024
        )

        is_valid, error_message = limit.check_file_size(Path("/nonexistent/file.wav"))
        assert not is_valid
        assert "does not exist" in error_message

    def test_check_file_size_valid_file(self, tmp_path):
        """Test check_file_size with a valid file."""
        # Create a small test file
        test_file = tmp_path / "test.wav"
        with open(test_file, "wb") as f:
            f.write(b"x" * 1024)  # 1KB file

        limit = ApiSizeLimit(
            provider=ApiProvider.OPENAI_WHISPER, max_file_size_bytes=25 * 1024 * 1024
        )

        is_valid, error_message = limit.check_file_size(test_file)
        assert is_valid
        assert error_message is None

    def test_check_file_size_oversized_file(self, tmp_path):
        """Test check_file_size with an oversized file."""
        # Create a test file slightly larger than limit
        limit_bytes = 1024  # 1KB limit for testing
        test_file = tmp_path / "test.wav"
        with open(test_file, "wb") as f:
            f.write(b"x" * (limit_bytes + 1))  # 1KB + 1 byte

        limit = ApiSizeLimit(provider=ApiProvider.OPENAI_WHISPER, max_file_size_bytes=limit_bytes)

        is_valid, error_message = limit.check_file_size(test_file)
        assert not is_valid
        assert "exceeds maximum allowed size" in error_message

    def test_check_file_size_unsupported_format(self, tmp_path):
        """Test check_file_size with an unsupported format."""
        # Create a test file with unsupported extension
        test_file = tmp_path / "test.xyz"
        with open(test_file, "wb") as f:
            f.write(b"x" * 1024)

        limit = ApiSizeLimit(
            provider=ApiProvider.OPENAI_WHISPER,
            max_file_size_bytes=25 * 1024 * 1024,
            supported_formats=["wav", "mp3"],
        )

        is_valid, error_message = limit.check_file_size(test_file)
        assert not is_valid
        assert "not supported" in error_message

    def test_bytes_under_limit(self):
        """Test bytes_under_limit calculation."""
        limit = ApiSizeLimit(
            provider=ApiProvider.OPENAI_WHISPER, max_file_size_bytes=10 * 1024 * 1024  # 10MB
        )

        # File size under limit
        assert limit.bytes_under_limit(8 * 1024 * 1024) == 2 * 1024 * 1024  # 2MB under

        # File size equal to limit
        assert limit.bytes_under_limit(10 * 1024 * 1024) == 0

        # File size over limit
        assert limit.bytes_under_limit(12 * 1024 * 1024) == -2 * 1024 * 1024  # 2MB over


class TestApiSizeLimitPolicy:
    """Tests for the ApiSizeLimitPolicy class."""

    def test_get_limit_for_known_provider(self):
        """Test getting limits for known providers."""
        # OpenAI Whisper
        whisper_limit = ApiSizeLimitPolicy.get_limit_for_provider(ApiProvider.OPENAI_WHISPER)
        assert whisper_limit.provider == ApiProvider.OPENAI_WHISPER
        assert whisper_limit.max_file_size_bytes == 25 * 1024 * 1024  # 25MB

        # Google Speech
        google_limit = ApiSizeLimitPolicy.get_limit_for_provider(ApiProvider.GOOGLE_SPEECH)
        assert google_limit.provider == ApiProvider.GOOGLE_SPEECH
        assert google_limit.max_file_size_bytes == 10 * 1024 * 1024  # 10MB

    def test_get_limit_for_other_provider(self):
        """Test getting limits for OTHER provider."""
        other_limit = ApiSizeLimitPolicy.get_limit_for_provider(ApiProvider.OTHER)
        assert other_limit.provider == ApiProvider.OTHER
        assert other_limit.max_file_size_bytes == 10 * 1024 * 1024  # Default 10MB

    def test_get_limit_for_unknown_provider(self):
        """Test getting limits for unknown provider raises ValueError."""
        # Create a custom ApiProvider value not in the policy
        with pytest.raises(ValueError):
            ApiSizeLimitPolicy.get_limit_for_provider("unknown_provider")

    def test_validate_file_for_provider(self, tmp_path):
        """Test validating a file against provider constraints."""
        # Create a small test file
        test_file = tmp_path / "test.wav"
        with open(test_file, "wb") as f:
            f.write(b"x" * 1024)  # 1KB file

        # Valid file
        is_valid, error_message = ApiSizeLimitPolicy.validate_file_for_provider(
            test_file, ApiProvider.OPENAI_WHISPER
        )
        assert is_valid
        assert error_message is None

        # Unsupported format
        test_file_unsupported = tmp_path / "test.xyz"
        with open(test_file_unsupported, "wb") as f:
            f.write(b"x" * 1024)

        is_valid, error_message = ApiSizeLimitPolicy.validate_file_for_provider(
            test_file_unsupported, ApiProvider.OPENAI_WHISPER
        )
        assert not is_valid
        assert "not supported" in error_message

    def test_register_custom_limit(self):
        """Test registering a custom size limit."""
        # Create a custom limit
        custom_limit = ApiSizeLimit(
            provider=ApiProvider.OTHER,
            max_file_size_bytes=5 * 1024 * 1024,  # 5MB
            supported_formats=["wav"],
        )

        # Register the custom limit
        ApiSizeLimitPolicy.register_custom_limit(ApiProvider.OTHER, custom_limit)

        # Get the registered limit
        retrieved_limit = ApiSizeLimitPolicy.get_limit_for_provider(ApiProvider.OTHER)
        assert retrieved_limit.max_file_size_bytes == 5 * 1024 * 1024
        assert retrieved_limit.supported_formats == ["wav"]
