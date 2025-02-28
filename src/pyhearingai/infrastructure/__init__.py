"""
Infrastructure layer package initialization.

This module imports all adapter implementations to ensure they are registered
with the registry when the package is imported.
"""

# Import the registry to make it available via the infrastructure package
from pyhearingai.infrastructure.registry import (
    list_output_formatters,
    get_output_formatter,
    list_transcribers,
    get_transcriber,
    list_diarizers,
    get_diarizer,
    get_converter,
    get_speaker_assigner
)

# Import the adapters module to register all implementations
from pyhearingai.infrastructure import adapters

__all__ = [
    "list_output_formatters",
    "get_output_formatter",
    "list_transcribers",
    "get_transcriber",
    "list_diarizers",
    "get_diarizer",
    "get_converter",
    "get_speaker_assigner"
]
