"""
Unit tests for the ResponsesReconciliationAdapter.

This module tests the functionality of the ResponsesReconciliationAdapter which uses
OpenAI's Responses API for efficient reconciliation of diarization and transcription results.
"""

import json
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from pyhearingai.core.idempotent import ProcessingJob, ProcessingStatus
from pyhearingai.core.models import DiarizationSegment, Segment
from pyhearingai.reconciliation.adapters.responses import (
    BatchProcessor,
    ResponsesReconciliationAdapter,
    ResultAggregator,
    TokenCounter,
)


class TestTokenCounter:
    """Tests for the TokenCounter class."""

    def test_init(self):
        """Test initialization with default model."""
        counter = TokenCounter()
        assert counter.model == "gpt-4o"
        # Encoding should be initialized

    def test_count_tokens(self):
        """Test counting tokens in a string."""
        counter = TokenCounter("gpt-4o")
        test_text = "This is a test string for counting tokens."
        token_count = counter.count_tokens(test_text)
        assert token_count > 0, "Token count should be greater than zero"

    def test_estimate_tokens_for_segments(self):
        """Test estimating tokens for diarization and transcription segments."""
        counter = TokenCounter("gpt-4o")

        # Create test segments
        diarization_segments = [
            DiarizationSegment(start=0.0, end=5.0, speaker_id="SPEAKER_01", confidence=0.9),
            DiarizationSegment(start=5.0, end=10.0, speaker_id="SPEAKER_02", confidence=0.8),
        ]

        transcription_segments = [
            Segment(start=0.0, end=5.0, text="Hello, how are you today?", speaker_id=None),
            Segment(start=5.0, end=10.0, text="I'm doing well, thank you.", speaker_id=None),
        ]

        segment_transcriptions = {
            "segment_0": "Hello, how are you today?",
            "segment_1": "I'm doing well, thank you.",
        }

        # Estimate tokens
        token_count = counter.estimate_tokens_for_batch(
            diarization_segments, transcription_segments, segment_transcriptions
        )

        assert token_count > 0, "Token count should be greater than zero"
        # The exact count may vary based on the tokenizer, but should be reasonable
        assert token_count > 20, "Token count seems too low for the given segments"


class TestBatchProcessor:
    """Tests for the BatchProcessor class."""

    def test_init(self):
        """Test initialization with default parameters."""
        token_counter = TokenCounter()
        processor = BatchProcessor(token_counter)
        assert processor.token_counter == token_counter
        assert processor.max_tokens == 7000  # Default max tokens

    def test_create_batches_single_batch(self):
        """Test creating a single batch when data is small enough."""
        # Mock token counter that always returns a small token count
        token_counter = MagicMock()
        token_counter.estimate_tokens_for_batch.return_value = 1000

        processor = BatchProcessor(token_counter, max_tokens=7000)

        # Create test data
        diarization_segments = {"chunk_1": [MagicMock(spec=DiarizationSegment)]}
        transcription_segments = {"chunk_1": [MagicMock(spec=Segment)]}
        segment_transcriptions = {"segment_1": "Test transcript"}

        # Create batches
        batches = processor.create_batches(
            diarization_segments, transcription_segments, segment_transcriptions
        )

        assert len(batches) == 1, "Should create a single batch"
        assert "diarization_segments" in batches[0]
        assert "transcription_segments" in batches[0]
        assert "segment_transcriptions" in batches[0]

    def test_create_batches_multiple_batches(self):
        """Test creating multiple batches when data exceeds token limit."""
        # Mock token counter
        token_counter = MagicMock()
        # First call returns a value exceeding max_tokens, second call returns a small value
        token_counter.estimate_tokens_for_batch.side_effect = [8000, 4000, 3000]

        processor = BatchProcessor(token_counter, max_tokens=7000)

        # Create test data with multiple chunks
        diarization_segments = {
            "chunk_1": [MagicMock(spec=DiarizationSegment)],
            "chunk_2": [MagicMock(spec=DiarizationSegment)],
        }
        transcription_segments = {
            "chunk_1": [MagicMock(spec=Segment)],
            "chunk_2": [MagicMock(spec=Segment)],
        }
        segment_transcriptions = {
            "segment_1": "First transcript",
            "segment_2": "Second transcript",
        }

        # Create batches
        batches = processor.create_batches(
            diarization_segments, transcription_segments, segment_transcriptions
        )

        assert len(batches) > 1, "Should create multiple batches"

    def test_format_batch_prompt(self):
        """Test formatting a batch into a prompt string."""
        token_counter = TokenCounter()
        processor = BatchProcessor(token_counter)

        # Create a simple test batch
        batch = {
            "diarization_segments": {
                "chunk_1": [
                    DiarizationSegment(start=0.0, end=5.0, speaker_id="SPEAKER_01", confidence=0.9)
                ]
            },
            "transcription_segments": {
                "chunk_1": [
                    Segment(start=0.0, end=5.0, text="Hello, how are you today?", speaker_id=None)
                ]
            },
            "segment_transcriptions": {"segment_1": "Hello, how are you today?"},
        }

        # Create a mock job
        job = MagicMock(spec=ProcessingJob)
        job.id = "test-job-123"

        # Format the prompt
        prompt = processor.format_batch_prompt(batch, 0, 1, job)

        # Verify prompt contains key elements
        assert "diarization segments" in prompt.lower()
        assert "transcription segments" in prompt.lower()
        assert "Hello, how are you today?" in prompt


class TestResultAggregator:
    """Tests for the ResultAggregator class."""

    def test_process_responses(self):
        """Test processing a list of responses and extracting segments."""
        aggregator = ResultAggregator()

        # Mock responses with segment information
        responses = [
            """Here are the reconciled segments:

SPEAKER_01 (0.0s - 5.0s): Hello, how are you today?
SPEAKER_02 (5.0s - 10.0s): I'm doing well, thank you.
""",
            """Here are the reconciled segments:

SPEAKER_02 (10.0s - 15.0s): Let's continue our conversation.
SPEAKER_01 (15.0s - 20.0s): That sounds good to me.
""",
        ]

        # Process responses
        segments = aggregator.process_responses(responses)

        assert len(segments) == 4, "Should extract 4 segments from the responses"
        assert segments[0].speaker_id == "SPEAKER_01"
        assert segments[0].text == "Hello, how are you today?"
        assert segments[0].start == 0.0
        assert segments[0].end == 5.0


class TestResponsesReconciliationAdapter:
    """Tests for the ResponsesReconciliationAdapter class."""

    @patch("pyhearingai.reconciliation.adapters.responses.OpenAI")
    def test_init(self, mock_openai):
        """Test initialization with default parameters."""
        adapter = ResponsesReconciliationAdapter()
        assert adapter.model == "gpt-4o"
        assert isinstance(adapter.token_counter, TokenCounter)
        assert isinstance(adapter.batch_processor, BatchProcessor)
        assert isinstance(adapter.result_aggregator, ResultAggregator)
        mock_openai.assert_called_once()

    @patch("pyhearingai.reconciliation.adapters.responses.OpenAI")
    def test_reconcile(self, mock_openai):
        """Test the reconcile method for processing diarization and transcription data."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "resp_123"
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = """Here are the reconciled segments:

SPEAKER_01 (0.0s - 5.0s): Hello, how are you today?
SPEAKER_02 (5.0s - 10.0s): I'm doing well, thank you.
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Create adapter with mocked components
        adapter = ResponsesReconciliationAdapter()

        # Mock batch processor to return a single batch
        mock_batch = {
            "diarization_segments": {
                "chunk_1": [
                    DiarizationSegment(start=0.0, end=5.0, speaker_id="SPEAKER_01", confidence=0.9)
                ]
            },
            "transcription_segments": {
                "chunk_1": [
                    Segment(start=0.0, end=5.0, text="Hello, how are you today?", speaker_id=None)
                ]
            },
            "segment_transcriptions": {"segment_1": "Hello, how are you today?"},
        }
        adapter.batch_processor.create_batches = MagicMock(return_value=[mock_batch])
        adapter.batch_processor.format_batch_prompt = MagicMock(return_value="Test prompt")

        # Create test data
        job = MagicMock(spec=ProcessingJob)
        job.id = "test-job-123"
        job.status = ProcessingStatus.PROCESSING

        diarization_segments = {"chunk_1": [MagicMock(spec=DiarizationSegment)]}
        transcription_segments = {"chunk_1": [MagicMock(spec=Segment)]}
        segment_transcriptions = {"segment_1": "Test transcript"}

        # Call reconcile
        segments = adapter.reconcile(
            job, diarization_segments, transcription_segments, segment_transcriptions
        )

        # Verify API was called
        mock_client.chat.completions.create.assert_called_once()

        # Verify results
        assert segments is not None
        assert len(segments) == 2, "Should extract 2 segments from the response"
        assert segments[0].speaker_id == "SPEAKER_01"
        assert segments[0].text == "Hello, how are you today?"
        assert segments[0].start == 0.0
        assert segments[0].end == 5.0

    @patch("pyhearingai.reconciliation.adapters.responses.OpenAI")
    def test_api_error_handling(self, mock_openai):
        """Test handling of API errors."""
        # Mock OpenAI client to raise an exception
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        # Create adapter with mocked components
        adapter = ResponsesReconciliationAdapter()

        # Mock batch processor
        adapter.batch_processor.create_batches = MagicMock(return_value=[{"mock": "batch"}])
        adapter.batch_processor.format_batch_prompt = MagicMock(return_value="Test prompt")

        # Create test data
        job = MagicMock(spec=ProcessingJob)
        job.id = "test-job-123"
        job.status = ProcessingStatus.PROCESSING

        diarization_segments = {"chunk_1": [MagicMock(spec=DiarizationSegment)]}
        transcription_segments = {"chunk_1": [MagicMock(spec=Segment)]}
        segment_transcriptions = {"segment_1": "Test transcript"}

        # Call reconcile and expect exception
        with pytest.raises(Exception):
            adapter.reconcile(
                job, diarization_segments, transcription_segments, segment_transcriptions
            )

        # Verify API was called
        mock_client.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
