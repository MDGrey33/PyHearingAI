"""
Tests for the domain events system.
"""

from datetime import datetime
from pathlib import Path

import pytest

from pyhearingai.core.domain.api_constraints import ApiProvider
from pyhearingai.core.domain.audio_quality import AudioQualitySpecification
from pyhearingai.core.domain.events import (
    AudioConversionEvent,
    AudioSizeExceededEvent,
    AudioValidationEvent,
    ChunkingEvent,
    DomainEvent,
    EventPublisher,
    EventSeverity,
)


class TestDomainEvent:
    """Tests for the base DomainEvent class."""

    def test_default_values(self):
        """Test default values for DomainEvent."""
        event = DomainEvent()

        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)
        assert event.event_id is not None
        assert event.severity == EventSeverity.INFO


class TestAudioValidationEvent:
    """Tests for the AudioValidationEvent class."""

    def test_event_creation(self):
        """Test creating an AudioValidationEvent."""
        file_path = Path("/test/file.wav")
        event = AudioValidationEvent(file_path=file_path, is_valid=True)

        assert event.file_path == file_path
        assert event.is_valid is True
        assert event.error_message is None
        assert event.api_provider is None
        assert event.severity == EventSeverity.INFO

    def test_invalid_event_severity(self):
        """Test that invalid events get ERROR severity."""
        event = AudioValidationEvent(
            file_path=Path("/test/file.wav"), is_valid=False, error_message="Test error"
        )

        assert event.severity == EventSeverity.ERROR
        assert event.error_message == "Test error"


class TestAudioConversionEvent:
    """Tests for the AudioConversionEvent class."""

    def test_event_creation(self):
        """Test creating an AudioConversionEvent."""
        source_path = Path("/test/source.m4a")
        target_path = Path("/test/target.wav")
        quality_spec = AudioQualitySpecification()

        event = AudioConversionEvent(
            source_path=source_path,
            target_path=target_path,
            is_successful=True,
            quality_spec=quality_spec,
            original_size=1000,
            converted_size=500,
        )

        assert event.source_path == source_path
        assert event.target_path == target_path
        assert event.is_successful is True
        assert event.quality_spec == quality_spec
        assert event.original_size == 1000
        assert event.converted_size == 500
        assert event.error_message is None
        assert event.severity == EventSeverity.INFO

    def test_failed_conversion_event(self):
        """Test creating a failed conversion event."""
        event = AudioConversionEvent(
            source_path=Path("/test/source.m4a"),
            target_path=Path("/test/target.wav"),
            is_successful=False,
            quality_spec=AudioQualitySpecification(),
            original_size=1000,
            converted_size=0,
            error_message="Conversion failed",
        )

        assert event.is_successful is False
        assert event.error_message == "Conversion failed"
        assert event.severity == EventSeverity.ERROR

    def test_compression_ratio_calculation(self):
        """Test that compression ratio is calculated correctly."""
        event = AudioConversionEvent(
            source_path=Path("/test/source.m4a"),
            target_path=Path("/test/target.wav"),
            is_successful=True,
            quality_spec=AudioQualitySpecification(),
            original_size=1000,
            converted_size=500,
        )

        assert "compression_ratio" in event.metadata
        assert event.metadata["compression_ratio"] == 2.0  # 1000/500


class TestAudioSizeExceededEvent:
    """Tests for the AudioSizeExceededEvent class."""

    def test_event_creation(self):
        """Test creating an AudioSizeExceededEvent."""
        file_path = Path("/test/file.wav")
        event = AudioSizeExceededEvent(
            file_path=file_path,
            actual_size=30 * 1024 * 1024,  # 30MB
            max_allowed_size=25 * 1024 * 1024,  # 25MB
            api_provider=ApiProvider.OPENAI_WHISPER,
        )

        assert event.file_path == file_path
        assert event.actual_size == 30 * 1024 * 1024
        assert event.max_allowed_size == 25 * 1024 * 1024
        assert event.api_provider == ApiProvider.OPENAI_WHISPER
        assert event.auto_adjustment_attempted is False
        assert event.auto_adjustment_successful is False
        assert event.adjusted_file_path is None
        assert event.severity == EventSeverity.ERROR

    def test_successful_adjustment(self):
        """Test creating an event with successful adjustment."""
        file_path = Path("/test/file.wav")
        adjusted_path = Path("/test/file_adjusted.wav")
        event = AudioSizeExceededEvent(
            file_path=file_path,
            actual_size=30 * 1024 * 1024,  # 30MB
            max_allowed_size=25 * 1024 * 1024,  # 25MB
            api_provider=ApiProvider.OPENAI_WHISPER,
            auto_adjustment_attempted=True,
            auto_adjustment_successful=True,
            adjusted_file_path=adjusted_path,
        )

        assert event.auto_adjustment_attempted is True
        assert event.auto_adjustment_successful is True
        assert event.adjusted_file_path == adjusted_path
        assert event.severity == EventSeverity.WARNING


class TestChunkingEvent:
    """Tests for the ChunkingEvent class."""

    def test_event_creation(self):
        """Test creating a ChunkingEvent."""
        source_path = Path("/test/source.wav")
        chunk_paths = [Path("/test/chunks/chunk_0000.wav"), Path("/test/chunks/chunk_0001.wav")]

        event = ChunkingEvent(
            source_path=source_path,
            job_id="test_job",
            chunk_count=2,
            chunk_duration=30.0,
            overlap_duration=5.0,
            chunk_paths=chunk_paths,
        )

        assert event.source_path == source_path
        assert event.job_id == "test_job"
        assert event.chunk_count == 2
        assert event.chunk_duration == 30.0
        assert event.overlap_duration == 5.0
        assert event.chunk_paths == chunk_paths
        assert event.has_oversized_chunks is False
        assert event.oversized_chunk_indices == []
        assert event.severity == EventSeverity.INFO

    def test_oversized_chunks(self):
        """Test creating an event with oversized chunks."""
        event = ChunkingEvent(
            source_path=Path("/test/source.wav"),
            job_id="test_job",
            chunk_count=3,
            chunk_duration=30.0,
            overlap_duration=5.0,
            has_oversized_chunks=True,
            oversized_chunk_indices=[1],
        )

        assert event.has_oversized_chunks is True
        assert event.oversized_chunk_indices == [1]
        assert event.severity == EventSeverity.WARNING


class TestEventPublisher:
    """Tests for the EventPublisher class."""

    def setup_method(self):
        """Set up the test by clearing subscribers."""
        EventPublisher.clear_subscribers()
        self.received_events = []

    def event_callback(self, event):
        """Callback for receiving events."""
        self.received_events.append(event)

    def test_subscribe_and_publish(self):
        """Test subscribing to and publishing events."""
        # Subscribe to DomainEvent (will receive all events)
        EventPublisher.subscribe(DomainEvent, self.event_callback)

        # Create and publish an event
        event = DomainEvent()
        EventPublisher.publish(event)

        # Check that the event was received
        assert len(self.received_events) == 1
        assert self.received_events[0] == event

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        # Subscribe and then unsubscribe
        EventPublisher.subscribe(DomainEvent, self.event_callback)
        EventPublisher.unsubscribe(DomainEvent, self.event_callback)

        # Publish an event
        EventPublisher.publish(DomainEvent())

        # Check that no events were received
        assert len(self.received_events) == 0

    def test_inheritance_based_subscription(self):
        """Test that subscribing to a parent class receives child class events."""
        # Subscribe to DomainEvent
        EventPublisher.subscribe(DomainEvent, self.event_callback)

        # Publish a child class event
        child_event = AudioValidationEvent(file_path=Path("/test/file.wav"), is_valid=True)
        EventPublisher.publish(child_event)

        # Check that the event was received
        assert len(self.received_events) == 1
        assert self.received_events[0] == child_event

    def test_specific_subscription(self):
        """Test subscribing to a specific event type."""
        # Subscribe to AudioValidationEvent only
        EventPublisher.subscribe(AudioValidationEvent, self.event_callback)

        # Publish a different event type
        EventPublisher.publish(DomainEvent())

        # Check that no events were received
        assert len(self.received_events) == 0

        # Publish an AudioValidationEvent
        validation_event = AudioValidationEvent(file_path=Path("/test/file.wav"), is_valid=True)
        EventPublisher.publish(validation_event)

        # Check that the event was received
        assert len(self.received_events) == 1
        assert self.received_events[0] == validation_event
