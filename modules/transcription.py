import os
import json
import requests
from datetime import datetime

def transcribe_audio(audio_path, api_key, output_dir="content/transcription"):
    """
    Transcribes audio using OpenAI's Whisper API and saves results in the specified output directory.
    Returns a tuple of (transcript text, segments).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    OPENAI_WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    try:
        with open(audio_path, "rb") as audio_file:
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            files = {"file": (audio_path, audio_file, "audio/wav")}
            data = {
                "model": "whisper-1",
                "language": "en",
                "response_format": "json"
            }
            
            # Create request log
            request_log = {
                "timestamp": datetime.now().isoformat(),
                "url": OPENAI_WHISPER_URL,
                "headers": {"Authorization": "Bearer [REDACTED]"},
                "data": data,
                "file_info": {
                    "name": os.path.basename(audio_path),
                    "size": os.path.getsize(audio_path)
                }
            }
            
            # Save request information
            with open(os.path.join(output_dir, "request_info.json"), "w") as f:
                json.dump(request_log, f, indent=2)
            
            # Make the API request
            response = requests.post(OPENAI_WHISPER_URL, headers=headers, files=files, data=data)
            
            # Save raw response
            response_log_path = os.path.join(output_dir, "api_response.json")
            with open(response_log_path, "w") as f:
                json.dump(response.json(), f, indent=2)
            
            # Handle errors
            if response.status_code != 200:
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                raise Exception(f"Whisper API error (status {response.status_code}): {error_msg}")
            
            response_json = response.json()
            
            # Save the transcript
            transcript_path = os.path.join(output_dir, "transcript.txt")
            with open(transcript_path, "w") as f:
                f.write(response_json.get("text", ""))
            
            # Save segments if available
            segments = response_json.get("segments", [])
            if segments:
                segments_path = os.path.join(output_dir, "segments.json")
                with open(segments_path, "w") as f:
                    json.dump(segments, f, indent=2)
            
            # Create summary file
            summary_path = os.path.join(output_dir, "transcription_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"Transcription completed at: {datetime.now().isoformat()}\n")
                f.write(f"Input file: {audio_path}\n")
                f.write(f"Transcript length: {len(response_json.get('text', ''))} characters\n")
                f.write(f"Number of segments: {len(segments)}\n")
                f.write(f"Response status: {response.status_code}\n")
            
            return response_json.get("text", ""), segments
            
    except Exception as e:
        error_log_path = os.path.join(output_dir, "transcription_error.log")
        with open(error_log_path, "w") as f:
            f.write(f"Error transcribing {audio_path}:\n{str(e)}")
        raise Exception(f"Transcription error: {str(e)}")

if __name__ == "__main__":
    # Test the module
    from dotenv import load_dotenv
    load_dotenv()
    
    test_file = "content/audio_conversion/example_audio_converted.wav"
    api_key = os.getenv("OPENAI_API_KEY")
    
    if os.path.exists(test_file) and api_key:
        try:
            transcript, segments = transcribe_audio(test_file, api_key)
            print(f"Successfully transcribed {test_file}")
            print(f"Transcript length: {len(transcript)} characters")
            print(f"Number of segments: {len(segments)}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Test file not found or API key not set") 