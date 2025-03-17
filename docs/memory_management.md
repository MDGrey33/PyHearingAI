# Memory Management in PyHearingAI

This document explains the memory management features in PyHearingAI, designed to help control resource usage when processing large audio files.

## Overview

PyHearingAI includes memory management features that allow you to:

1. Set a global memory limit that the application will try to respect
2. Manually clean up resources when needed
3. Monitor memory usage and automatically clean up resources when memory usage gets high
4. Track resources to ensure they are properly released

These features are particularly useful when processing large audio files or when processing multiple files in batch.

## Setting Memory Limits

### Using Python API

You can set a global memory limit for PyHearingAI using the `set_memory_limit` function:

```python
from pyhearingai import set_memory_limit

# Limit memory usage to 4GB
set_memory_limit(4096)  # In MB
```

When a memory limit is set, PyHearingAI will monitor memory usage and take action when memory usage approaches the limit. This includes:

- Cleaning up cached data
- Releasing resources that are no longer needed
- Throttling processing to avoid memory spikes

### Using Environment Variables

You can also set memory limits using environment variables:

```bash
# Set a 4GB memory limit
export PYHEARINGAI_MEMORY_LIMIT=4096
```

This is particularly useful when running scripts or the CLI without modifying code.

### Using CLI Options

When using the PyHearingAI CLI, several parameters can indirectly affect memory usage:

```bash
# Control chunk size (smaller chunks use less memory)
python -m pyhearingai audio.mp3 --chunk-size 5.0

# Limit parallel processing (fewer workers use less memory)
python -m pyhearingai audio.mp3 --max-workers 2

# Specify a custom cache directory
python -m pyhearingai audio.mp3 --cache-dir /path/to/cache
```

## Manual Resource Cleanup

You can manually clean up resources at any time using the `cleanup_resources` function:

```python
from pyhearingai import cleanup_resources

# After processing several files
freed_mb = cleanup_resources()
print(f"Freed {freed_mb:.2f} MB of memory")
```

This is useful when you want to explicitly free memory between processing tasks.

## Integration with Existing Classes

The memory management features are integrated with the existing classes in PyHearingAI:

- `WorkflowOrchestrator` tracks the resources it creates (services, repositories, etc.) and registers them for cleanup
- Services like `DiarizationService`, `TranscriptionService`, and `ReconciliationService` implement `close` methods to properly release resources
- The `ResourceManager` monitors memory usage and initiates cleanup when needed

## Example Usage

Here's an example of processing a large audio file with memory management:

```python
from pyhearingai import transcribe_chunked, set_memory_limit, cleanup_resources

# Set a memory limit
set_memory_limit(2048)  # 2GB limit

# Process a large file in chunks to manage memory usage
result = transcribe_chunked(
    "very_long_recording.mp3",
    chunk_size_seconds=300,  # 5-minute chunks
    overlap_seconds=10       # 10 seconds overlap
)

# Explicitly clean up resources after processing
freed_mb = cleanup_resources()
print(f"Freed {freed_mb:.2f} MB of memory")
```

For more examples, see the [memory_management_example.py](../examples/memory_management_example.py) file.

## Best Practices

To get the most out of the memory management features, follow these best practices:

1. **Set a reasonable memory limit**: Set a limit below your system's total memory to leave room for other applications. A good rule of thumb is 70-80% of your total memory.

2. **Use chunked processing for large files**: The `transcribe_chunked` function is designed to process large files in manageable chunks, keeping memory usage under control.

3. **Use pipeline_session for multiple files**: The `pipeline_session` context manager efficiently reuses resources across multiple files and ensures proper cleanup when done.

4. **Clean up explicitly when appropriate**: Call `cleanup_resources` between processing large batches of files to ensure memory is released.

5. **Monitor memory usage**: If you're processing very large files or many files in batch, monitor memory usage to ensure it stays within acceptable limits.

6. **Adjust chunk size based on file length**: For very long files, use smaller chunk sizes (30-60 seconds). For shorter files, larger chunk sizes (300-600 seconds) may be more efficient.

7. **Limit worker processes when memory constrained**: If memory is limited, reduce the number of parallel workers to prevent memory spikes.

## Memory Management via CLI

To manage memory effectively from the command line:

```bash
# Set memory limit via environment variable
export PYHEARINGAI_MEMORY_LIMIT=2048

# Process with smaller chunks and limited workers
python -m pyhearingai long_recording.mp3 --chunk-size 30.0 --max-workers 2 --verbose
```

This combination of options provides good control over memory usage without requiring code changes.

## Under the Hood

The memory management system consists of the following components:

- **ResourceManager**: A singleton class that tracks resources and provides memory monitoring and cleanup functions.
- **WorkflowOrchestrator**: The main orchestration class that integrates with the ResourceManager to track services and resources.
- **ResourceSupervisor**: A monitoring utility that watches CPU and memory usage and triggers callbacks when resource usage gets high.
- **Config module**: Provides the `set_memory_limit` function and keeps track of the global memory limit setting.

These components work together to provide a comprehensive memory management system for PyHearingAI.

## Limitations

While the memory management features help control resource usage, they have some limitations:

1. **Not a hard limit**: The memory limit is not a hard limit. The system tries to respect it by cleaning up resources, but it may exceed the limit temporarily during processing.

2. **Dependent on system support**: Some features depend on the availability of system monitoring tools like `psutil`. If these are not available, the system will still function but with reduced capability.

3. **Manual intervention may be needed**: For extremely memory-intensive operations, manual intervention (such as calling `cleanup_resources`) may still be needed.

For more information about memory management in PyHearingAI, see the [Resource Management](resource_management.md) documentation.
