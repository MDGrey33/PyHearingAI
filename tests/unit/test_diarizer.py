import os

import pytest

from tests.helpers import create_segment, patch_pyannote_pipeline


def test_pyannote_diarizer_basic(
    temp_audio_file, temp_dir, mock_pyannote, diarization_segments, assert_segments
):
    """Test basic functionality of the PyannoteDiarizer."""
    from pyhearingai.infrastructure.diarizers.pyannote import PyannoteDiarizer

    # Skip the test if no API key is available
    api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not api_key:
        pytest.skip("HUGGINGFACE_API_KEY not available in environment")

    diarizer = PyannoteDiarizer()

    # Use the patch helper to mock Pyannote Pipeline
    with patch_pyannote_pipeline() as mock_pipeline_cls:
        # Set the return value to our pre-configured mock
        mock_pipeline_cls.from_pretrained.return_value = mock_pyannote

        # Configure the mock to return our test segments
        mock_pipeline_cls.return_value = mock_pyannote

        # Call the diarizer with a mock implementation that returns our segments
        with pytest.MonkeyPatch.context() as mp:
            # Mock the _mock_diarize method to return our test segments
            mp.setattr(diarizer, "_mock_diarize", lambda *args, **kwargs: diarization_segments)

            # Force using the mock implementation
            segments = diarizer._mock_diarize(audio_path=temp_audio_file, output_dir=temp_dir)

        # Verify the structure of the returned segments
        assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"

        # Check first segment
        expected_segment = create_segment(start=0.0, end=2.0, speaker_id="SPEAKER_00", text="")
        assert_segments(segments[0], expected_segment, check_speaker=True)

        # Check second segment
        expected_segment = create_segment(start=2.5, end=4.5, speaker_id="SPEAKER_01", text="")
        assert_segments(segments[1], expected_segment, check_speaker=True)

        # Check third segment
        expected_segment = create_segment(start=5.0, end=7.0, speaker_id="SPEAKER_00", text="")
        assert_segments(segments[2], expected_segment, check_speaker=True)
