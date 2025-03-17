"""
Diarizers package initialization.

This module imports all diarizer implementations to ensure they are registered
with the registry when the package is imported.
"""

# Import all diarizers so they are registered
from pyhearingai.infrastructure.diarizers.pyannote import PyannoteDiarizer

__all__ = ["PyannoteDiarizer"]
