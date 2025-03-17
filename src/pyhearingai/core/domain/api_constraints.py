"""
API constraints domain model.

This module defines domain entities and value objects that encapsulate
constraints imposed by various API providers, such as size limits.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple


class ApiProvider(Enum):
    """Enumeration of supported API providers."""

    OPENAI_WHISPER = "openai_whisper"
    WHISPER_LOCAL = "whisper_local"
    HUGGINGFACE = "huggingface"
    ASSEMBLY_AI = "assembly_ai"
    GOOGLE_SPEECH = "google_speech"


@dataclass(frozen=True)
class ApiSizeLimit:
    """
    Immutable value object representing size limits for an API provider.

    Attributes:
        provider: The API provider this limit applies to
        max_file_size_bytes: Maximum file size in bytes
        max_duration_seconds: Maximum audio duration in seconds
    """

    provider: ApiProvider
    max_file_size_bytes: int
    max_duration_seconds: int = 0  # 0 means no duration limit


class ApiSizeLimitPolicy:
    """
    Domain service for managing API size limits across different providers.

    This service encapsulates the logic for validating files against
    provider-specific size constraints.
    """

    # Default size limits for each provider
    _DEFAULT_LIMITS: Dict[ApiProvider, ApiSizeLimit] = {
        ApiProvider.OPENAI_WHISPER: ApiSizeLimit(
            provider=ApiProvider.OPENAI_WHISPER,
            max_file_size_bytes=25 * 1024 * 1024,  # 25MB
            max_duration_seconds=600,  # 10 minutes
        ),
        ApiProvider.WHISPER_LOCAL: ApiSizeLimit(
            provider=ApiProvider.WHISPER_LOCAL,
            max_file_size_bytes=0,  # No size limit for local processing
            max_duration_seconds=0,  # No duration limit for local processing
        ),
        ApiProvider.HUGGINGFACE: ApiSizeLimit(
            provider=ApiProvider.HUGGINGFACE,
            max_file_size_bytes=150 * 1024 * 1024,  # 150MB
            max_duration_seconds=0,  # Varies by model
        ),
        ApiProvider.ASSEMBLY_AI: ApiSizeLimit(
            provider=ApiProvider.ASSEMBLY_AI,
            max_file_size_bytes=1024 * 1024 * 1024,  # 1GB
            max_duration_seconds=0,  # No strict limit
        ),
        ApiProvider.GOOGLE_SPEECH: ApiSizeLimit(
            provider=ApiProvider.GOOGLE_SPEECH,
            max_file_size_bytes=500 * 1024 * 1024,  # 500MB
            max_duration_seconds=480 * 60,  # 480 minutes
        ),
    }

    @staticmethod
    def get_limit_for_provider(provider: ApiProvider) -> ApiSizeLimit:
        """
        Get the size limit for a specific provider.

        Args:
            provider: The API provider to get limits for

        Returns:
            ApiSizeLimit: The size limit constraints for the provider

        Raises:
            ValueError: If the provider is not supported
        """
        if provider not in ApiSizeLimitPolicy._DEFAULT_LIMITS:
            raise ValueError(f"Unsupported API provider: {provider}")

        return ApiSizeLimitPolicy._DEFAULT_LIMITS[provider]

    @staticmethod
    def validate_file_for_provider(
        file_path: Path, provider: ApiProvider
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a file against provider-specific constraints.

        Args:
            file_path: Path to the file to validate
            provider: The API provider to validate against

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, f"File not found: {file_path}"

        # Get provider limit
        try:
            limit = ApiSizeLimitPolicy.get_limit_for_provider(provider)
        except ValueError as e:
            return False, str(e)

        # Check file size
        if limit.max_file_size_bytes > 0:
            file_size = os.path.getsize(file_path)
            if file_size > limit.max_file_size_bytes:
                return (
                    False,
                    f"File size {file_size} bytes exceeds {provider.value} limit of "
                    f"{limit.max_file_size_bytes} bytes",
                )

        # For duration checks, we would need to analyze the audio file
        # This could be implemented with a service that extracts duration

        return True, None
