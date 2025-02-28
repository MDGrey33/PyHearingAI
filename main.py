import os
import json
from dotenv import load_dotenv
from modules.audio_conversion import convert_audio_to_wav
from modules.transcription import transcribe_audio
from modules.diarization import diarize_audio
from modules.speaker_assignment import assign_speakers

def main():
    """Main pipeline that processes an audio file through conversion, transcription, diarization, and speaker assignment."""
    # Load environment variables
    load_dotenv()
    
    try:
        # 1. Audio Conversion
        print("\nStep 1: Audio Conversion", end="... ")
        input_audio = "example_audio.m4a"
        converted_path = convert_audio_to_wav(input_audio)
        print("✓")
        
        # 2. Transcription
        print("Step 2: Transcription", end="... ")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("Missing OpenAI API key")
        transcript, segments = transcribe_audio(converted_path, openai_key)
        print("✓")
        
        # 3. Diarization
        print("Step 3: Diarization", end="... ")
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_key:
            raise ValueError("Missing Hugging Face API key")
        diarization_segments = diarize_audio(converted_path, hf_key)
        print("✓")
        
        # 4. Speaker Assignment
        print("Step 4: Speaker Assignment", end="... ")
        transcript_segments = [{
            "text": transcript,
            "start": diarization_segments[0]["start"] if diarization_segments else 0,
            "end": diarization_segments[-1]["end"] if diarization_segments else len(transcript)
        }]
        labeled_transcript = assign_speakers(transcript_segments, diarization_segments)
        print("✓\n")
        
        # Final output
        print(f"✨ Processing complete! Labeled transcript available at:")
        print(f"content/speaker_assignment/labeled_transcript.txt")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()


