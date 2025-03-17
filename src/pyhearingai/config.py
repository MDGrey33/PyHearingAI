"""
Configuration module for PyHearingAI.

This module provides centralized configuration management for the PyHearingAI system,
including feature flags, global settings, and environment variable handling.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Feature flags
USE_IDEMPOTENT_PROCESSING = (
    os.environ.get("PYHEARINGAI_USE_IDEMPOTENT_PROCESSING", "false").lower() == "true"
)


# Default base directory for storing persistence data
def _get_default_data_dir() -> Path:
    """
    Get the default data directory for storing persistent data.

    Uses the XDG Base Directory specification on Unix systems and
    appropriate locations on other platforms.
    """
    if os.name == "posix":  # Unix-like systems
        base_dir = os.environ.get(
            "XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share")
        )
        return Path(base_dir) / "pyhearingai"
    else:  # Windows and others
        return Path(os.path.expanduser("~")) / ".pyhearingai"


# Global configuration settings
DATA_DIR = Path(os.environ.get("PYHEARINGAI_DATA_DIR", str(_get_default_data_dir())))
JOBS_DIR = DATA_DIR / "jobs"
CHUNKS_DIR = DATA_DIR / "chunks"

# Memory management settings
# Default to 0 (no limit), value is in MB
MEMORY_LIMIT = int(os.environ.get("PYHEARINGAI_MEMORY_LIMIT", "0"))


@dataclass
class IdempotentProcessingConfig:
    """Configuration settings for idempotent processing."""

    # Whether to use idempotent processing
    enabled: bool = USE_IDEMPOTENT_PROCESSING

    # Directories for storing persistent data
    data_dir: Path = DATA_DIR
    jobs_dir: Path = JOBS_DIR
    chunks_dir: Path = CHUNKS_DIR

    # Processing parameters
    chunk_duration: float = 300.0  # Default chunk size in seconds (5 minutes)
    chunk_overlap: float = 5.0  # Default overlap between chunks in seconds

    # Repository persistence
    use_json_persistence: bool = True  # Whether to use JSON files for persistence


@dataclass
class ResourceManagementConfig:
    """Configuration settings for resource management."""

    # Memory limits in MB (0 = no limit)
    memory_limit: int = MEMORY_LIMIT

    # CPU usage thresholds
    cpu_high_threshold: float = 80.0  # 80% CPU usage
    cpu_low_threshold: float = 60.0  # 60% CPU usage

    # Memory usage thresholds
    memory_high_threshold: float = 80.0  # 80% memory usage
    memory_low_threshold: float = 60.0  # 60% memory usage

    # Monitoring settings
    enable_monitoring: bool = True
    poll_interval: float = 5.0  # Check resources every 5 seconds


@dataclass
class TranscriptionConfig:
    """Configuration settings for audio transcription."""

    # Model selections
    transcriber: str = "openai"  # Default transcriber (openai, whisper, etc.)
    diarizer: str = "pyannote"  # Default diarizer (pyannote, none, etc.)

    # API keys
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None

    # Output formatting
    output_format: str = "txt"  # Default output format (txt, json, srt, vtt, md)

    # Processing options
    verbose: bool = False  # Whether to show verbose output

    # Idempotent processing settings
    idempotent: IdempotentProcessingConfig = field(default_factory=IdempotentProcessingConfig)

    # Resource management settings
    resources: ResourceManagementConfig = field(default_factory=ResourceManagementConfig)

    # Additional options passed to models
    model_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize API keys from environment variables if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")

        if self.huggingface_api_key is None:
            self.huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")


def setup():
    """
    Set up the configuration system, ensuring necessary directories exist.
    """
    # Create data directories if they don't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def set_memory_limit(limit_mb: int):
    """
    Set the global memory usage limit for PyHearingAI.

    This function configures how much memory the application is allowed to use.
    When the limit is approached, the system will take action to reduce memory usage
    by cleaning up resources or throttling processing.

    Args:
        limit_mb: Maximum memory usage in megabytes (0 = no limit)

    Returns:
        None

    Examples:
        >>> from pyhearingai.config import set_memory_limit
        >>> set_memory_limit(4096)  # Limit memory usage to 4GB
    """
    global MEMORY_LIMIT
    MEMORY_LIMIT = limit_mb


# Run setup when module is imported
setup()
