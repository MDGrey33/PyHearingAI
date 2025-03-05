#!/usr/bin/env python
"""
Simplest possible example of using PyHearingAI.

IMPORTANT: Before using the PyHearingAI library with the default diarizer:
1. Make sure you have set up API keys for OpenAI and Hugging Face
2. You MUST accept the Hugging Face model terms at:
   - https://huggingface.co/pyannote/speaker-diarization
   - https://huggingface.co/pyannote/segmentation

This example produces the same result as the CLI command:
python -m pyhearingai.cli tests/fixtures/example_audio.m4a -f json -o transcript.json
"""

import os

from pyhearingai import transcribe

# Get API keys from environment
openai_key = os.environ.get("OPENAI_API_KEY")
huggingface_key = os.environ.get("HUGGINGFACE_API_KEY")

# Prepare arguments
kwargs = {}
if openai_key:
    kwargs["api_key"] = openai_key
if huggingface_key:
    kwargs["huggingface_api_key"] = huggingface_key

# Transcribe an audio file
result = transcribe("tests/fixtures/example_audio.m4a", **kwargs)

# Print results
for segment in result.segments:
    print(f"{segment.speaker_id}: {segment.text}")

# Save as JSON file - explicitly specify format like the CLI does
result.save("transcript.json", format="json")
