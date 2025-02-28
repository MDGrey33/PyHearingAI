#!/usr/bin/env python
"""
Simple example of using PyHearingAI to transcribe an audio file.

This example shows how to use the library to transcribe an audio file
with speaker diarization and save the result in various formats.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from pyhearingai import transcribe


def main():
    """Run the transcription example."""
    # Load environment variables (API keys)
    load_dotenv()
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in a .env file or export it in your shell.")
        return 1
    
    if not os.getenv("HUGGINGFACE_API_KEY"):
        print("Error: HUGGINGFACE_API_KEY environment variable not set.")
        print("Please set it in a .env file or export it in your shell.")
        return 1
    
    # Determine the path to the example audio file
    audio_path = Path(__file__).parent / "example.mp3"
    
    # If the example audio file doesn't exist, print an error
    if not audio_path.exists():
        print(f"Error: Example audio file not found: {audio_path}")
        print("Please download an example audio file and save it as 'example.mp3'")
        print("in the examples directory.")
        return 1
    
    # Transcribe the audio file
    print(f"Transcribing {audio_path}...")
    
    try:
        # Simple transcription with default options
        result = transcribe(audio_path, verbose=True)
        
        # Save the result in different formats
        formats = ["txt", "json", "srt", "md"]
        
        for format in formats:
            output_path = audio_path.with_suffix(f".{format}")
            print(f"Saving transcription to {output_path}...")
            result.save(output_path, format=format)
        
        # Print the transcription
        print("\nTranscription:")
        print("-" * 80)
        for segment in result.segments:
            print(f"{segment.speaker_id} [{segment.start_timecode}-{segment.end_timecode}]: {segment.text}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 