"""
Test for API size limit policy domain services.

Tests the implementation of ApiSizeLimit and ApiSizeLimitPolicy.
"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from pyhearingai.core.domain.api_constraints import ApiProvider, ApiSizeLimit, ApiSizeLimitPolicy


class TestApiSizeLimit:
    """Test suite for the ApiSizeLimit value object."""

    def test_init_with_default_values(self):
        """Test that ApiSizeLimit can be created with default values."""
        limit = ApiSizeLimit(
            provider=ApiProvider.OPENAI_WHISPER, max_file_size_bytes=25 * 1024 * 1024
        )

        assert limit.provider == ApiProvider.OPENAI_WHISPER
        assert limit.max_file_size_bytes == 25 * 1024 * 1024
        assert limit.max_duration_seconds == 0  # Default value

    def test_init_with_custom_values(self):
        """Test that ApiSizeLimit can be created with custom values."""
        limit = ApiSizeLimit(
            provider=ApiProvider.OPENAI_WHISPER,
            max_file_size_bytes=50 * 1024 * 1024,
            max_duration_seconds=900,
        )

        assert limit.provider == ApiProvider.OPENAI_WHISPER
        assert limit.max_file_size_bytes == 50 * 1024 * 1024
        assert limit.max_duration_seconds == 900


class TestApiSizeLimitPolicy:
    """Test suite for the ApiSizeLimitPolicy domain service."""

    def test_get_limit_for_known_provider(self):
        """Test retrieving limits for a known provider."""
        policy = ApiSizeLimitPolicy()

        # Test with OpenAI Whisper provider
        limit = policy.get_limit_for_provider(ApiProvider.OPENAI_WHISPER)

        assert limit.provider == ApiProvider.OPENAI_WHISPER
        assert limit.max_file_size_bytes == 25 * 1024 * 1024  # 25MB
        assert limit.max_duration_seconds == 600  # 10 minutes

    def test_get_limit_for_unknown_provider(self):
        """Test that getting limits for an unknown provider raises ValueError."""
        policy = ApiSizeLimitPolicy()

        # Create a non-existent provider value
        # We can't extend Enum, so we'll test with a non-existent provider
        with pytest.raises(ValueError, match="Unsupported API provider"):
            # Use a provider value that doesn't exist
            class FakeProvider:
                value = "non_existent_provider"

            # This should raise ValueError
            policy.get_limit_for_provider(FakeProvider())

    def test_validate_file_for_provider(self):
        """Test file validation against provider constraints."""
        policy = ApiSizeLimitPolicy()

        # Create a temporary test file
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file_path = Path(temp_file.name)

            # Write 1MB of data
            temp_file.write(b"0" * (1 * 1024 * 1024))

        try:
            # Test with a file under the limit
            is_valid, error_message = policy.validate_file_for_provider(
                temp_file_path, ApiProvider.OPENAI_WHISPER
            )

            assert is_valid is True
            assert error_message is None
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    def test_validate_file_over_size_limit(self):
        """Test validation of a file exceeding size limits."""
        policy = ApiSizeLimitPolicy()

        # Test with a non-existent file
        non_existent_path = Path("/non/existent/file.wav")
        is_valid, error_message = policy.validate_file_for_provider(
            non_existent_path, ApiProvider.OPENAI_WHISPER
        )

        assert is_valid is False
        assert "File not found" in error_message

    def test_validate_file_nonexistent(self):
        """Test validation with a non-existent file."""
        policy = ApiSizeLimitPolicy()

        # Create a patched version of get_limit_for_provider
        original_get_limit = ApiSizeLimitPolicy.get_limit_for_provider

        try:
            # Create a test file that will exceed our mocked size limit
            with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file_path = Path(temp_file.name)

                # Write data exceeding our mock limit
                temp_file.write(b"0" * 1000)  # 1000 bytes > 100 bytes limit

            # Patch the method with mock
            def mock_get_tiny_limit(provider):
                """Mock method that returns a tiny size limit."""
                if provider == ApiProvider.OPENAI_WHISPER:
                    return ApiSizeLimit(
                        provider=ApiProvider.OPENAI_WHISPER,
                        max_file_size_bytes=100,  # Very small limit
                        max_duration_seconds=600,
                    )
                return original_get_limit(provider)

            # Apply the patch
            ApiSizeLimitPolicy.get_limit_for_provider = staticmethod(mock_get_tiny_limit)

            try:
                # Test with a file over the limit
                is_valid, error_message = policy.validate_file_for_provider(
                    temp_file_path, ApiProvider.OPENAI_WHISPER
                )

                assert is_valid is False
                assert "exceeds" in error_message
                assert "limit" in error_message
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)

        finally:
            # Restore the original method
            ApiSizeLimitPolicy.get_limit_for_provider = original_get_limit
