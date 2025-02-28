import os
import json
import requests
from datetime import datetime

def assign_speakers(transcript_segments, diarization_segments, output_dir="content/speaker_assignment"):
    """
    Uses GPT-4 to intelligently match transcript text with speaker segments.
    Returns the labeled transcript with speaker assignments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OpenAI API key in .env file")
        
        # Extract full transcript from segments
        transcript = transcript_segments[0]["text"] if transcript_segments else ""
        
        # Prepare the prompt for GPT
        prompt = f"""Provide the transcribed dialogue with clear speaker distinctions based on the transcript and segments.

Transcript:
{transcript}

Speaker Segments:
{json.dumps(diarization_segments, indent=2)}"""
        
        # Call GPT-4 API
        print("Calling GPT-4 to analyze speaker segments...")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 16384
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"GPT API error: {response.text}")
        
        # Extract the segmented transcript
        segmented_transcript = response.json()["choices"][0]["message"]["content"]
        
        # Save the complete response for debugging
        with open(os.path.join(output_dir, "gpt_response.json"), "w") as f:
            json.dump(response.json(), f, indent=2)
        
        # Save the segmented transcript
        with open(os.path.join(output_dir, "labeled_transcript.txt"), "w") as f:
            f.write(segmented_transcript)
        
        # Create summary file
        summary_path = os.path.join(output_dir, "assignment_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Processing completed at: {datetime.now().isoformat()}\n")
            f.write(f"Input transcript length: {len(transcript)} characters\n")
            f.write(f"Number of diarization segments: {len(diarization_segments)}\n")
            f.write(f"Number of unique speakers: {len(set(seg['speaker'] for seg in diarization_segments))}\n\n")
            f.write("Processing details:\n")
            f.write(f"- Model: {response.json()['model']}\n")
            f.write(f"- Processing time: {response.json()['usage']['total_tokens']} tokens\n")
        
        # Create processing log
        log_path = os.path.join(output_dir, "processing_log.txt")
        with open(log_path, "w") as f:
            f.write(f"Processing started at: {datetime.now().isoformat()}\n")
            f.write(f"Transcript length: {len(transcript)} characters\n")
            f.write(f"Number of diarization segments: {len(diarization_segments)}\n\n")
            f.write("Prompt sent to GPT:\n")
            f.write("-" * 40 + "\n")
            f.write(prompt)
            f.write("\n" + "-" * 40 + "\n")
        
        return segmented_transcript
        
    except Exception as e:
        error_log_path = os.path.join(output_dir, "assignment_error.log")
        with open(error_log_path, "w") as f:
            f.write(f"Error in speaker assignment:\n{str(e)}")
        raise Exception(f"Speaker assignment error: {str(e)}")

if __name__ == "__main__":
    # Test the module
    try:
        # Load test data
        with open("content/transcription/transcript.txt", "r") as f:
            transcript = f.read()
        
        with open("content/diarization/segments.json", "r") as f:
            diarization_segments = json.load(f)
        
        # Create test transcript segment
        transcript_segments = [{"text": transcript}]
        
        # Process the segments
        labeled_transcript = assign_speakers(transcript_segments, diarization_segments)
        print("Successfully created labeled transcript")
        print("\nPreview:")
        print("-" * 40)
        preview = labeled_transcript[:500] + "..." if len(labeled_transcript) > 500 else labeled_transcript
        print(preview)
        
    except Exception as e:
        print(f"Error: {e}") 