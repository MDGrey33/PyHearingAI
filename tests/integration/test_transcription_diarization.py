import os
from unittest.mock import patch

import pytest

# Import helper functions
from tests.helpers import create_segment


@pytest.mark.integration
def test_transcription_diarization_integration(
    example_audio_path,
    temp_dir,
    transcript_segments,
    diarization_segments,
    audio_converter,
    transcriber,
    diarizer,
    speaker_assigner,
    assert_segment_lists,
):
    """Test integration between transcription and diarization components."""
    # Skip the test if API keys are not available
    openai_key = os.environ.get("OPENAI_API_KEY")
    hf_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not openai_key or not hf_key:
        pytest.skip("Required API keys not available in environment")

    # Set up temporary directories
    conv_dir = temp_dir / "audio_conversion"
    diar_dir = temp_dir / "diarization"
    conv_dir.mkdir(exist_ok=True)
    diar_dir.mkdir(exist_ok=True)

    # Step 1: Convert audio
    converted_path = audio_converter.convert(example_audio_path, output_dir=conv_dir)

    # Verify conversion happened
    assert converted_path.exists(), "Converted audio file does not exist"

    # Step 2: Mock transcription
    with patch.object(transcriber.__class__, "transcribe", return_value=transcript_segments):
        transcript_result = transcriber.transcribe(audio_path=converted_path, api_key=openai_key)

        # Step 3: Mock diarization
        with patch.object(diarizer.__class__, "diarize", return_value=diarization_segments):
            diarization_result = diarizer.diarize(
                audio_path=converted_path, api_key=hf_key, output_dir=diar_dir
            )

            # Step 4: Apply speaker assignment
            labeled_segments = speaker_assigner.assign_speakers(
                transcript_segments=transcript_result,
                diarization_segments=diarization_result,
                output_dir=temp_dir,
            )

            # Verify speaker assignment
            assert len(labeled_segments) == 3, f"Expected 3 segments, got {len(labeled_segments)}"

            # Create expected segments
            expected_segments = [
                create_segment(start=0.0, end=2.0, text="This is a test.", speaker_id="Speaker 0"),
                create_segment(
                    start=2.5, end=4.5, text="Testing the transcriber.", speaker_id="Speaker 1"
                ),
                create_segment(
                    start=5.0, end=7.0, text="Final test segment.", speaker_id="Speaker 0"
                ),
            ]

            # Verify the labeled segments match the expected result
            assert_segment_lists(labeled_segments, expected_segments, check_speaker=True)
