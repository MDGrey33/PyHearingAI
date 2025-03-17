# PyHearingAI Examples

This directory contains example scripts demonstrating how to use PyHearingAI for various audio transcription and speaker diarization tasks.

## Available Examples

### Simple Transcription

[simple_transcription.py](simple_transcription.py) - Basic example showing how to transcribe an audio file with speaker diarization.

```bash
python simple_transcription.py path/to/audio.mp3 [output.txt]
```

### Chunked Transcription for Large Files

[transcribe_chunked_example.py](transcribe_chunked_example.py) - Example showing how to process large audio files in manageable chunks to avoid memory issues.

```bash
python transcribe_chunked_example.py path/to/large_audio.mp3 [output.txt]
```

### Batch Processing with Pipeline Session

[pipeline_session_example.py](pipeline_session_example.py) - Example showing how to efficiently process multiple audio files by reusing resources.

```bash
python pipeline_session_example.py audio1.mp3 audio2.mp3 audio3.mp3
```

### Memory Management

[memory_management_example.py](memory_management_example.py) - Example showing how to control memory usage and clean up resources.

```bash
python memory_management_example.py audio_file.mp3 [memory_limit_mb] [output.txt]
```

### Time Range Processing

[time_range_example.py](time_range_example.py) - Example showing how to process only a specific portion of an audio file.

```bash
# Process from a specific start time to the end
python time_range_example.py audio_file.mp3 30.0

# Process a specific time range (start to end)
python time_range_example.py audio_file.mp3 30.0 60.0

# Process a sample of specific duration from the start
python time_range_example.py audio_file.mp3 --sample 30.0
```

## Running the Examples

To run these examples, you need to have PyHearingAI installed. You can install it using:

```bash
pip install pyhearingai
```

Alternatively, if you're running the examples from the repository, the scripts will automatically add the project root to the Python path.

## Additional Information

Each example script includes detailed comments explaining how it works. You can also run any example with the `-h` flag to see usage information.

For more detailed documentation, refer to the main [PyHearingAI README](../README.md) and the [documentation directory](../docs).
