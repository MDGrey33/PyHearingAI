import io
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Import helper functions
from tests.helpers import patch_openai


def test_whisper_openai_transcriber_basic():
    """Test basic functionality of the WhisperOpenAITranscriber."""
    import openai

    from pyhearingai.infrastructure.transcribers.whisper_openai import WhisperOpenAITranscriber

    # Skip the test if no API key is available
    api_key = os.environ.get("OPENAI_API_KEY", "dummy_key")

    transcriber = WhisperOpenAITranscriber()

    # Create segment objects with proper attributes using SimpleNamespace
    segment1 = SimpleNamespace(id=0, start=0.0, end=2.0, text="This is a test.")
    segment2 = SimpleNamespace(id=1, start=2.5, end=4.5, text="Testing the transcriber.")

    # Configure the mock response with the appropriate segment objects
    mock_response = MagicMock()
    mock_response.model = "whisper-1"
    mock_response.segments = [segment1, segment2]

    # Create a mock Path object
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.__str__.return_value = "dummy_path.wav"

    # Mock file data
    mock_file_data = b"dummy audio data"

    # Patch both the open function and the OpenAI API call
    with patch("builtins.open", mock_open(read_data=mock_file_data)) as mock_open_func, patch(
        "openai.audio.transcriptions.create"
    ) as mock_create:
        # Set up our mock response
        mock_create.return_value = mock_response

        # Call the transcriber with our mock path
        segments = transcriber.transcribe(audio_path=mock_path, api_key=api_key)

        # Verify the mocks were called
        mock_open_func.assert_called_once_with(mock_path, "rb")
        mock_create.assert_called_once()

        # Verify structure of returned segments
        assert len(segments) == 2

        # Verify first segment
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0
        assert segments[0].text == "This is a test."

        # Verify second segment
        assert segments[1].start == 2.5
        assert segments[1].end == 4.5
        assert segments[1].text == "Testing the transcriber."
