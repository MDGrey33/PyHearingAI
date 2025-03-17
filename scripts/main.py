import json
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

# Import from pyhearingai clean architecture
from pyhearingai.infrastructure.audio_converter import FFmpegAudioConverter
from pyhearingai.infrastructure.diarizers.pyannote import PyannoteDiarizer
from pyhearingai.infrastructure.speaker_assignment import DefaultSpeakerAssigner
from pyhearingai.infrastructure.speaker_assignment_gpt import GPTSpeakerAssigner
from pyhearingai.infrastructure.transcribers.whisper_openai import WhisperOpenAITranscriber


def main():
    """
    Main pipeline that processes an audio file through conversion, transcription, diarization, and speaker assignment.
    Testing each step individually for better debugging.
    """
    # Load environment variables
    load_dotenv()

    try:
        # 1. Audio Conversion - Test this step first
        print("\nStep 1: Audio Conversion", end="... ")
        input_audio = "example_audio.m4a"

        # Convert audio using clean architecture
        converter = FFmpegAudioConverter()
        converted_path = converter.convert(Path(input_audio), output_dir="content/audio_conversion")
        print("✓")
        print(f"Converted audio file: {converted_path}")

        # 2. Transcription
        print("Step 2: Transcription", end="... ")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("Missing OpenAI API key")

        # Set up output directory for transcription
        transcription_output_dir = "content/transcription"
        os.makedirs(transcription_output_dir, exist_ok=True)

        # Use the clean architecture transcriber
        transcriber = WhisperOpenAITranscriber()
        transcript_segments = transcriber.transcribe(audio_path=converted_path, api_key=openai_key)

        # Extract full transcript
        transcript = " ".join([seg.text for seg in transcript_segments])

        # Save transcript to file (like in the original implementation)
        transcript_path = os.path.join(transcription_output_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write(transcript)

        # Save segments to file
        segments_path = os.path.join(transcription_output_dir, "segments.json")
        segments_data = [
            {"text": seg.text, "start": seg.start, "end": seg.end} for seg in transcript_segments
        ]
        with open(segments_path, "w") as f:
            json.dump(segments_data, f, indent=2)

        # Create summary file
        summary_path = os.path.join(transcription_output_dir, "transcription_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Transcription completed at: {datetime.now().isoformat()}\n")
            f.write(f"Input file: {converted_path}\n")
            f.write(f"Transcript length: {len(transcript)} characters\n")
            f.write(f"Number of segments: {len(transcript_segments)}\n")

        print("✓")
        print(f"Transcript saved to: {transcript_path}")
        print(f"Transcript: {transcript[:100]}...")  # Print first 100 chars of transcript

        # 3. Diarization
        print("Step 3: Diarization", end="... ")
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_key:
            raise ValueError("Missing Hugging Face API key")

        # Set up output directory for diarization
        diarization_output_dir = "content/diarization"
        os.makedirs(diarization_output_dir, exist_ok=True)

        # Use the clean architecture diarizer
        # The diarizer now handles all file saving internally, just like the original implementation
        diarizer = PyannoteDiarizer()
        diarization_segments = diarizer.diarize(
            audio_path=converted_path, api_key=hf_key, output_dir=diarization_output_dir
        )

        print("✓")
        print(f"Diarization segments: {len(diarization_segments)} found")
        print(f"Diarization results saved to: {diarization_output_dir}")

        # The segments already have properly formatted speaker IDs now
        for i, segment in enumerate(diarization_segments[:3]):  # Print first 3 segments
            print(
                f"  Segment {i}: Speaker {segment.speaker_id}, {segment.start:.2f}s - {segment.end:.2f}s"
            )

        # 4. Speaker Assignment
        print("Step 4: Speaker Assignment", end="... ")

        # Set up output directory for speaker assignment
        speaker_output_dir = "content/speaker_assignment"
        os.makedirs(speaker_output_dir, exist_ok=True)

        # Use our clean architecture GPTSpeakerAssigner, which maintains the same output format
        # but follows clean architecture principles
        speaker_assigner = GPTSpeakerAssigner()
        labeled_segments = speaker_assigner.assign_speakers(
            transcript_segments=transcript_segments,
            diarization_segments=diarization_segments,
            output_dir=speaker_output_dir,
            api_key=openai_key,
            model="gpt-4o",
            temperature=0.3,
            max_tokens=16384,
            force_no_version=True,  # Force using exact model name without version
        )

        print("✓\n")

        # Final output
        print(f"✨ Processing complete! Labeled transcript available at:")
        print(f"content/speaker_assignment/labeled_transcript.txt")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
