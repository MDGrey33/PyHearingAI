import os
from pydub import AudioSegment

def convert_audio_to_wav(audio_path, output_dir="content/audio_conversion"):
    """
    Converts an audio file to WAV format and saves it in the specified output directory.
    Returns the path to the converted file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    try:
        # Load and convert the audio
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Save the converted file
        output_path = os.path.join(output_dir, f"{base_name}_converted.wav")
        audio.export(output_path, format="wav")
        
        # Create info file
        info_path = os.path.join(output_dir, "conversion_info.txt")
        with open(info_path, "w") as f:
            f.write(f"Original file: {audio_path}\n")
            f.write(f"Original format: {os.path.splitext(audio_path)[1]}\n")
            f.write(f"Original duration: {len(audio) / 1000:.2f} seconds\n")
            f.write(f"Original channels: {audio.channels}\n")
            f.write(f"Original frame rate: {audio.frame_rate} Hz\n")
            f.write(f"Converted file: {output_path}\n")
            f.write(f"Conversion timestamp: {os.path.getctime(output_path)}\n")
        
        return output_path
        
    except Exception as e:
        error_log_path = os.path.join(output_dir, "conversion_error.log")
        with open(error_log_path, "w") as f:
            f.write(f"Error converting {audio_path}:\n{str(e)}")
        raise Exception(f"Error converting audio file: {str(e)}")

if __name__ == "__main__":
    # Test the module
    test_file = "example_audio.m4a"
    if os.path.exists(test_file):
        try:
            output_path = convert_audio_to_wav(test_file)
            print(f"Successfully converted {test_file} to {output_path}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Test file {test_file} not found") 