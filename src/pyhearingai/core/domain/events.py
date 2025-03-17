"""
Domain events for the audio processing workflow.

This module defines domain events that capture important occurrences
during audio processing, such as validation failures or conversion issues.
These events facilitate communication between domain components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from pyhearingai.core.domain.audio_quality import AudioQualitySpecification
from pyhearingai.core.domain.api_constraints import ApiProvider


class EventSeverity(Enum):
    """Severity levels for domain events."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DomainEvent:
    """Base class for domain events."""
    
    # Required fields without defaults must come first
    # Adding defaults for all fields to avoid inheritance issues with derived classes
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: __import__('uuid').uuid4().__str__())
    severity: EventSeverity = EventSeverity.INFO
    
    # No longer need __post_init__ as we use default_factory instead
    # which is called at object initialization time


@dataclass
class AudioValidationEvent(DomainEvent):
    """Event indicating an audio validation result."""
    
    # Required fields
    file_path: Path = None  # Add default to avoid errors
    is_valid: bool = False  # Add default to avoid errors
    
    # Optional fields
    error_message: Optional[str] = None
    api_provider: Optional[ApiProvider] = None
    
    def __post_init__(self):
        """Set severity based on validation result."""
        # No need to call super().__post_init__() as it's been removed
        
        if not self.is_valid:
            self.severity = EventSeverity.ERROR


@dataclass
class AudioConversionEvent(DomainEvent):
    """Event indicating an audio conversion result."""
    
    # Required fields with defaults to avoid parameter order issues
    source_path: Path = None
    target_path: Path = None
    is_successful: bool = False
    quality_spec: AudioQualitySpecification = None
    original_size: int = 0
    converted_size: int = 0
    
    # Optional fields
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set severity based on conversion result and calculate metadata."""
        # Initialize metadata if None
        if self.metadata is None:
            self.metadata = {}
            
        # Set severity based on success
        if not self.is_successful:
            self.severity = EventSeverity.ERROR
            
        # Calculate compression ratio if both sizes are valid
        if self.original_size > 0 and self.converted_size > 0:
            self.metadata["compression_ratio"] = self.original_size / self.converted_size


@dataclass
class AudioSizeExceededEvent(DomainEvent):
    """Event indicating that an audio file exceeds size constraints."""
    
    # Required fields with defaults to avoid parameter order issues
    source_path: Path = None
    best_achieved_size: int = 0
    target_size_bytes: int = 0
    adjustments_tried: List[Dict[str, Any]] = None
    api_provider: Optional[ApiProvider] = None
    
    # Optional fields
    auto_adjustment_attempted: bool = False
    auto_adjustment_successful: bool = False
    adjusted_file_path: Optional[Path] = None
    
    def __post_init__(self):
        """Set severity based on auto-adjustment result."""
        # Initialize lists if None
        if self.adjustments_tried is None:
            self.adjustments_tried = []
            
        # Set severity based on adjustment result
        if not self.auto_adjustment_attempted or not self.auto_adjustment_successful:
            self.severity = EventSeverity.ERROR
        else:
            self.severity = EventSeverity.WARNING


@dataclass
class ChunkingEvent(DomainEvent):
    """Event related to audio chunking process."""
    
    # Required fields with defaults to avoid parameter order issues
    source_path: Path = None
    job_id: str = ""
    chunk_count: int = 0
    chunk_duration: float = 0.0
    overlap_duration: float = 0.0
    
    # Optional fields
    chunk_paths: List[Path] = None
    has_oversized_chunks: bool = False
    oversized_chunk_indices: List[int] = None
    
    def __post_init__(self):
        """Initialize lists and set severity based on chunking result."""
        # Initialize lists if None
        if self.chunk_paths is None:
            self.chunk_paths = []
            
        if self.oversized_chunk_indices is None:
            self.oversized_chunk_indices = []
            
        # Set warning severity if any chunks are oversized
        if self.has_oversized_chunks:
            self.severity = EventSeverity.WARNING


class EventPublisher:
    """
    Simple event publisher for domain events.
    
    This implementation uses a simple in-memory callback system for
    subscribing to and publishing events.
    """
    
    # Dict mapping event types to lists of subscribers
    _subscribers: Dict[type, List[callable]] = {}
    
    @classmethod
    def subscribe(cls, event_type: type, callback: callable) -> None:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Callback function to receive events
        """
        if event_type not in cls._subscribers:
            cls._subscribers[event_type] = []
            
        cls._subscribers[event_type].append(callback)
    
    @classmethod
    def publish(cls, event: DomainEvent) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        # Get subscribers for this event type
        event_type = type(event)
        subscribers = cls._subscribers.get(event_type, [])
        
        # Also check for subscribers of parent classes
        for base_type in event_type.__mro__[1:]:
            if base_type is object:
                break
                
            if base_type in cls._subscribers:
                subscribers.extend(cls._subscribers[base_type])
        
        # Notify all subscribers
        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    f"Error notifying subscriber for {event_type.__name__}: {str(e)}"
                )
    
    @classmethod
    def unsubscribe(cls, event_type: type, callback: callable) -> None:
        """
        Unsubscribe from a specific event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if event_type in cls._subscribers and callback in cls._subscribers[event_type]:
            cls._subscribers[event_type].remove(callback)
            
    @classmethod
    def clear_subscribers(cls) -> None:
        """Clear all subscribers."""
        cls._subscribers.clear() 