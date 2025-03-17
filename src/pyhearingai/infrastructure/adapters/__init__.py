"""
Adapters module for infrastructure implementations.

This package contains concrete implementations of the infrastructure adapters
for external services like audio conversion, format services, etc.
"""

# Export the audio format service
from pyhearingai.infrastructure.adapters.audio_format_service import FFmpegAudioFormatService

# Export the size-aware audio converter
from pyhearingai.infrastructure.adapters.size_aware_audio_converter import SizeAwareFFmpegConverter 