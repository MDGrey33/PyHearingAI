# Memory Management in PyHearingAI

PyHearingAI is designed to handle audio files of various sizes while managing system resources efficiently. This document explains how memory is managed in PyHearingAI and provides best practices for working with large audio files.

## Memory Usage Patterns

The memory usage during audio processing depends on several factors:

1. **Audio File Size**: Larger audio files require more memory for loading and processing
2. **Chunk Size**: Smaller chunks require less memory but increase processing time
3. **Model Selection**: Different transcription and diarization models have different memory requirements
4. **Parallel Processing**: Multi-worker processing increases memory usage

## Resource Manager

PyHearingAI includes a resource management system that:

1. **Monitors Memory Usage**: Tracks memory consumption during processing
2. **Releases Resources**: Automatically cleans up resources when no longer needed
3. **Prevents Memory Leaks**: Ensures that AI models are properly unloaded

## Best Practices for Large Files

When processing large audio files, follow these best practices:

### 1. Use Chunked Processing

Always use the `transcribe_chunked` function for large files:

```python
from pyhearingai import transcribe_chunked

result = transcribe_chunked(
    "large_audio.mp3",
    chunk_size_seconds=30.0,  # Process in 30-second chunks
    max_workers=2             # Limit to 2 parallel workers
)
```

### 2. Set Memory Limits

Set a memory limit to prevent excessive memory usage:

```python
from pyhearingai import set_memory_limit

# Set a 2GB memory limit
set_memory_limit(2048)  # MB
```

### 3. Process Time Ranges

For extremely large files, process specific time ranges:

```python
from pyhearingai import transcribe_chunked

# Process only minutes 5-15 of a long recording
result = transcribe_chunked(
    "very_large_audio.mp3",
    start_time=300,  # 5 minutes in seconds
    end_time=900     # 15 minutes in seconds
)
```

### 4. Clean Up Resources

Manually clean up resources when processing multiple files in a session:

```python
from pyhearingai import transcribe_chunked, cleanup_resources

# Process first file
result1 = transcribe_chunked("file1.mp3")

# Clean up resources
cleanup_resources()

# Process next file
result2 = transcribe_chunked("file2.mp3")
```

## Pipeline Sessions

For processing multiple files efficiently, use a pipeline session:

```python
from pyhearingai import PipelineSession

with PipelineSession(max_workers=2, chunk_size_seconds=30.0) as session:
    # Process multiple files using the same models
    result1 = session.process("file1.mp3")
    result2 = session.process("file2.mp3")
    result3 = session.process("file3.mp3")

    # Resources are automatically cleaned up when the session ends
```

## Cache Management

PyHearingAI caches processed chunks and job data to support idempotent processing. The cache can grow large over time, especially when processing many files.

### Cache Location

The cache is stored in:
- Linux/macOS: `~/.local/share/pyhearingai/`
- Windows: `%LOCALAPPDATA%\pyhearingai\`

### Cleaning the Cache

Use the provided cleanup script to manage cache size:

```bash
# Remove old cache entries, keeping the 5 most recent jobs
python examples/cleanup_cache.py

# Keep only the 10 most recent jobs
python examples/cleanup_cache.py --keep 10

# Reset the jobs database (creates a backup first)
python examples/cleanup_cache.py --reset-db
```

## Memory Profiling

To monitor memory usage during processing:

```python
from pyhearingai import transcribe_chunked
import psutil

# Get memory usage before processing
mem_before = psutil.Process().memory_info().rss / 1024 / 1024
print(f"Memory usage before: {mem_before:.2f} MB")

# Process audio file
result = transcribe_chunked("audio.mp3")

# Get memory usage after processing
mem_after = psutil.Process().memory_info().rss / 1024 / 1024
print(f"Memory usage after: {mem_after:.2f} MB")

# Clean up resources
from pyhearingai import cleanup_resources
cleanup_resources()

# Get memory usage after cleanup
mem_final = psutil.Process().memory_info().rss / 1024 / 1024
print(f"Memory usage after cleanup: {mem_final:.2f} MB")
```

## Technical Details

### Memory Management Implementation

The PyHearingAI resource management system is implemented in the `ResourceManager` class, which:

1. Tracks all loaded models and resources
2. Monitors memory usage using the `psutil` library
3. Implements reference counting for shared resources
4. Provides automatic and manual cleanup mechanisms

### Memory-Efficient Processing

PyHearingAI uses several techniques for memory-efficient processing:

1. **Streaming Audio Loading**: Loads audio in chunks rather than all at once
2. **Temporary File Management**: Cleans up temporary files after use
3. **Model Sharing**: Reuses models across multiple chunks
4. **Resource Supervision**: Monitors and manages resource usage during processing

## Troubleshooting

### High Memory Usage

If you encounter high memory usage:

1. **Reduce Chunk Size**: Use smaller chunks (e.g., 15-30 seconds)
2. **Limit Workers**: Reduce the number of parallel workers
3. **Process in Batches**: Process parts of the audio file separately
4. **Clean Cache**: Run the cleanup script to free disk space

### Out of Memory Errors

If you encounter out-of-memory errors:

1. **Set Memory Limit**: Use `set_memory_limit()` to prevent excessive memory usage
2. **Use Time Ranges**: Process smaller portions of the audio file
3. **Restart Python**: Start a fresh Python process for each large file
4. **Upgrade Hardware**: Consider using a machine with more RAM for very large files
