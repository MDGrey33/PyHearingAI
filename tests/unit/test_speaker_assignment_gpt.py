"""
Unit tests for the GPTSpeakerAssigner class.

Tests the GPT-based speaker assignment including:
- Basic functionality
- Response parsing
- Error handling
"""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from pyhearingai.core.models import Segment
from pyhearingai.infrastructure.speaker_assignment_gpt import GPTSpeakerAssigner


@pytest.fixture
def transcript_segments():
    """
    Sample transcript segments for testing.

    Returns:
        List[Segment]: A list of transcript segments
    """
    return [
        Segment(text="Hello, how are you today?", start=0.0, end=2.0),
        Segment(text="I'm doing well, thank you. How about you?", start=2.5, end=5.0),
        Segment(text="Pretty good, thanks for asking.", start=5.5, end=7.0),
    ]


@pytest.fixture
def diarization_segments():
    """
    Sample diarization segments for testing.

    Returns:
        List[Segment]: A list of diarization segments with speaker IDs
    """
    return [
        Segment(text="", start=0.0, end=2.0, speaker_id="01"),
        Segment(text="", start=2.5, end=5.0, speaker_id="02"),
        Segment(text="", start=5.5, end=7.0, speaker_id="01"),
    ]


@pytest.fixture
def mock_openai_response():
    """
    Mock response from OpenAI API for testing.

    Returns:
        dict: A mock OpenAI API response
    """
    return {
        "choices": [
            {
                "message": {
                    "content": """
**Speaker 1:** Hello, how are you today?

**Speaker 2:** I'm doing well, thank you. How about you?

**Speaker 1:** Pretty good, thanks for asking.
"""
                }
            }
        ],
        "model": "gpt-4o",
        "usage": {"total_tokens": 100},
    }


def test_basic_speaker_assignment(transcript_segments, diarization_segments, tmp_path):
    """
    Test basic speaker assignment functionality.

    Given: Transcript segments, diarization segments, and a mock OpenAI response
    When: Calling assign_speakers
    Then: The segments have the correct speaker IDs assigned
    """
    # Arrange
    assigner = GPTSpeakerAssigner()

    # Create a mock response object
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": """
**Speaker 1:** Hello, how are you today?

**Speaker 2:** I'm doing well, thank you. How about you?

**Speaker 1:** Pretty good, thanks for asking.
"""
                }
            }
        ],
        "model": "gpt-4o",
        "usage": {"total_tokens": 100},
    }

    with patch("requests.post") as mock_post, patch("os.makedirs") as mock_makedirs, patch(
        "builtins.open", create=True
    ), patch("builtins.print"):
        mock_post.return_value = mock_response

        # Act
        result = assigner.assign_speakers(
            transcript_segments,
            diarization_segments,
            output_dir=str(tmp_path),
            api_key="test_api_key",  # Add API key
        )

        # Assert
        assert len(result) == 3
        assert result[0].speaker_id == "Speaker 1"
        assert result[1].speaker_id == "Speaker 2"
        assert result[2].speaker_id == "Speaker 1"

        # Verify text is preserved
        assert result[0].text == "Hello, how are you today?"
        assert result[1].text == "I'm doing well, thank you. How about you?"
        assert result[2].text == "Pretty good, thanks for asking."

        # Verify timing is preserved
        assert result[0].start == 0.0
        assert result[0].end == 2.0


def test_error_handling_api_failure(transcript_segments, diarization_segments, tmp_path):
    """
    Test error handling when the OpenAI API call fails.

    Given: A failing OpenAI API
    When: Calling assign_speakers
    Then: An appropriate error is raised
    """
    # Arrange
    assigner = GPTSpeakerAssigner()

    # Create a mock response object with an error
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "API Error"

    with patch("requests.post") as mock_post, patch("os.makedirs") as mock_makedirs, patch(
        "builtins.open", create=True
    ), patch("builtins.print"):
        mock_post.return_value = mock_response

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            assigner.assign_speakers(
                transcript_segments,
                diarization_segments,
                output_dir=str(tmp_path),
                api_key="test_api_key",  # Add API key
            )

        assert "API Error" in str(exc_info.value)


def test_json_speaker_mapping(transcript_segments, diarization_segments, tmp_path):
    """
    Test parsing JSON speaker mapping from GPT response.

    Given: A GPT response with JSON mapping
    When: Calling assign_speakers
    Then: The segments have the correct speaker IDs assigned from the JSON mapping
    """
    # Arrange
    assigner = GPTSpeakerAssigner()

    # Create a mock response object with JSON mapping
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": """
Some text before the JSON.

```json
[
  {"segment_index": 0, "speaker": "Speaker 1"},
  {"segment_index": 1, "speaker": "Speaker 2"},
  {"segment_index": 2, "speaker": "Speaker 1"}
]
```

Some text after the JSON.
"""
                }
            }
        ],
        "model": "gpt-4o",
        "usage": {"total_tokens": 100},
    }

    with patch("requests.post") as mock_post, patch("os.makedirs") as mock_makedirs, patch(
        "builtins.open", create=True
    ), patch("builtins.print"):
        mock_post.return_value = mock_response

        # Act
        result = assigner.assign_speakers(
            transcript_segments,
            diarization_segments,
            output_dir=str(tmp_path),
            api_key="test_api_key",  # Add API key
        )

        # Assert
        assert len(result) == 3
        assert result[0].speaker_id == "Speaker 1"
        assert result[1].speaker_id == "Speaker 2"
        assert result[2].speaker_id == "Speaker 1"
