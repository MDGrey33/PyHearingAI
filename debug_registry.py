#!/usr/bin/env python
"""
Debug script to verify component registration in PyHearingAI.
This will print all registered components to help diagnose issues.
"""

import sys
from pathlib import Path

# Add the project root to the path to import pyhearingai
sys.path.insert(0, str(Path(__file__).parent))

print("Attempting to import registry...")
try:
    from pyhearingai.infrastructure.registry import (
        list_transcribers,
        list_diarizers,
        list_output_formatters,
    )
    print("Registry imported successfully")
except ImportError as e:
    print(f"Error importing registry: {e}")
    sys.exit(1)

print("\nChecking registered components:")
print(f"Transcribers: {list_transcribers()}")
print(f"Diarizers: {list_diarizers()}")
print(f"Formatters: {list_output_formatters()}")

print("\nAttempting to import components directly:")
try:
    print("\nImporting diarizers...")
    from pyhearingai.infrastructure.diarizers.pyannote import PyannoteDiarizer
    print("Pyannote diarizer imported successfully")
except ImportError as e:
    print(f"Error importing Pyannote diarizer: {e}")

try:
    print("\nImporting transcribers...")
    from pyhearingai.infrastructure.transcribers.whisper_openai import WhisperOpenAITranscriber
    print("Whisper OpenAI transcriber imported successfully")
except ImportError as e:
    print(f"Error importing Whisper OpenAI transcriber: {e}")

try:
    print("\nImporting formatters...")
    from pyhearingai.infrastructure.formatters.text import TextFormatter
    print("Text formatter imported successfully")
except ImportError as e:
    print(f"Error importing Text formatter: {e}")

print("\nRechecking registered components:")
print(f"Transcribers: {list_transcribers()}")
print(f"Diarizers: {list_diarizers()}")
print(f"Formatters: {list_output_formatters()}") 