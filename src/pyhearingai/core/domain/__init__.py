"""
Domain layer for PyHearingAI.

This package contains the core domain models, entities, value objects,
and domain services for the audio processing domain.
"""

from pyhearingai.core.domain.api_constraints import ApiProvider, ApiSizeLimit, ApiSizeLimitPolicy
from pyhearingai.core.domain.audio_quality import AudioCodec, AudioFormat, AudioQualitySpecification
from pyhearingai.core.domain.audio_validation import AudioValidationService
from pyhearingai.core.domain.events import (
    AudioConversionEvent,
    AudioSizeExceededEvent,
    AudioValidationEvent,
    ChunkingEvent,
    DomainEvent,
    EventPublisher,
    EventSeverity,
)

__all__ = [
    # Audio quality
    "AudioCodec",
    "AudioFormat",
    "AudioQualitySpecification",
    # API constraints
    "ApiProvider",
    "ApiSizeLimit",
    "ApiSizeLimitPolicy",
    # Audio validation
    "AudioValidationService",
    # Events
    "DomainEvent",
    "EventSeverity",
    "AudioValidationEvent",
    "AudioConversionEvent",
    "AudioSizeExceededEvent",
    "ChunkingEvent",
    "EventPublisher",
]
