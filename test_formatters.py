"""
Test script for PyHearingAI formatters.

This script demonstrates using all available output formatters.
"""

import os
from pathlib import Path

from pyhearingai import (
    Segment,
    TranscriptionResult,
    get_output_formatter,
    list_output_formatters,
    transcribe,
)


def test_formatters():
    """Test all available output formatters with a sample transcript."""

    # Create a sample transcript result
    sample_segments = [
        Segment(text="Hello, this is speaker one.", start=0.0, end=2.5, speaker_id="Speaker 1"),
        Segment(
            text="Hi there, this is speaker two responding.",
            start=3.0,
            end=5.5,
            speaker_id="Speaker 2",
        ),
        Segment(
            text="Let me continue with some additional context.",
            start=6.0,
            end=9.0,
            speaker_id="Speaker 1",
        ),
    ]

    sample_result = TranscriptionResult(
        segments=sample_segments,
        audio_path=Path("sample_audio.wav"),
        metadata={"transcriber": "test", "diarizer": "test", "duration": 9.0},
    )

    # Create output directory
    output_dir = Path("formatter_test_output")
    output_dir.mkdir(exist_ok=True)

    # Get all available formatters
    formatters = list_output_formatters()
    print(f"Available formatters: {formatters}")

    # Test each formatter
    for format_name in formatters:
        formatter = get_output_formatter(format_name)
        output_path = output_dir / f"sample.{format_name}"

        # Format and save
        formatter.save(sample_result, output_path)

        # Read back and print
        content = output_path.read_text(encoding="utf-8")
        print(f"\n--- {format_name.upper()} Format ---")
        print(f"Saved to: {output_path}")
        print("Content preview:")
        print(content[:200] + "..." if len(content) > 200 else content)
        print()


if __name__ == "__main__":
    test_formatters()
