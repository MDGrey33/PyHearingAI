import os
import json
import torch
from datetime import datetime
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

def diarize_audio(audio_path, api_key, output_dir="content/diarization"):
    """
    Performs speaker diarization using Pyannote.audio and saves results in the specified output directory.
    Returns a list of speaker segments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize pipeline
        print("\nInitializing speaker diarization pipeline...")
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=api_key
            )
        except Exception as auth_error:
            error_message = str(auth_error)
            if "401 Client Error" in error_message:
                raise Exception(
                    "Authentication failed. Please:\n"
                    "1. Verify your HUGGINGFACE_API_KEY is correct\n"
                    "2. Accept the user agreement at: https://hf.co/pyannote/speaker-diarization-3.1\n"
                    "3. Accept the user agreement at: https://hf.co/pyannote/segmentation-3.1"
                )
            raise
        
        # Log pipeline initialization
        with open(os.path.join(output_dir, "pipeline_info.txt"), "w") as f:
            f.write(f"Pipeline initialized at: {datetime.now().isoformat()}\n")
            f.write(f"Model: pyannote/speaker-diarization-3.1\n")
            f.write(f"Input file: {audio_path}\n")
        
        # Use GPU if available
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            device_info = "GPU (CUDA)"
        else:
            device_info = "CPU"
        
        # Log device information
        with open(os.path.join(output_dir, "pipeline_info.txt"), "a") as f:
            f.write(f"Processing device: {device_info}\n")
        
        print("\nProcessing audio file...")
        print(f"Using device: {device_info}")
        
        # Run diarization with progress monitoring
        with ProgressHook() as hook:
            diarization = pipeline(
                audio_path,
                min_speakers=1,
                max_speakers=5,
                hook=hook
            )
        
        # Save RTTM file
        rttm_path = os.path.join(output_dir, "diarization.rttm")
        with open(rttm_path, "w") as rttm:
            diarization.write_rttm(rttm)
        
        # Convert diarization to JSON format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = {
                "start": turn.start,
                "end": turn.end,
                "speaker": f"SPEAKER_{speaker}"
            }
            segments.append(segment)
        
        # Save segments as JSON
        segments_path = os.path.join(output_dir, "segments.json")
        with open(segments_path, "w") as f:
            json.dump(segments, f, indent=2)
        
        # Create summary file
        summary_path = os.path.join(output_dir, "diarization_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Diarization completed at: {datetime.now().isoformat()}\n")
            f.write(f"Input file: {audio_path}\n")
            f.write(f"Number of segments: {len(segments)}\n")
            f.write(f"Processing device: {device_info}\n")
            
            # Calculate some statistics
            total_duration = sum(seg["end"] - seg["start"] for seg in segments)
            unique_speakers = len(set(seg["speaker"] for seg in segments))
            
            f.write(f"Total audio duration processed: {total_duration:.2f} seconds\n")
            f.write(f"Number of unique speakers detected: {unique_speakers}\n")
        
        return segments
        
    except Exception as e:
        error_log_path = os.path.join(output_dir, "diarization_error.log")
        with open(error_log_path, "w") as f:
            f.write(f"Error in diarization:\n{str(e)}")
        raise Exception(f"Diarization error: {str(e)}")

if __name__ == "__main__":
    # Test the module
    from dotenv import load_dotenv
    load_dotenv()
    
    test_file = "content/audio_conversion/example_audio_converted.wav"
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if os.path.exists(test_file) and api_key:
        try:
            segments = diarize_audio(test_file, api_key)
            print(f"Successfully diarized {test_file}")
            print(f"Found {len(segments)} segments")
            print(f"Number of unique speakers: {len(set(seg['speaker'] for seg in segments))}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Test file not found or API key not set") 