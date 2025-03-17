"""
End-to-end tests for the size-aware audio processing pipeline.

These tests verify that the complete audio processing pipeline works correctly
with the size-aware audio conversion and validation components.

Run with:
    pytest tests/integration/test_size_aware_end_to_end.py -v

To run only the short test:
    pytest tests/integration/test_size_aware_end_to_end.py::test_short_conversation_end_to_end -v

To run only the long test:
    pytest tests/integration/test_size_aware_end_to_end.py::test_long_conversation_end_to_end -v
"""

import os
import pytest
import tempfile
import shutil
import subprocess
from pathlib import Path

# Mark these tests as end_to_end so they can be skipped in regular test runs
pytestmark = pytest.mark.end_to_end

# Path to test data
TEST_DATA_DIR = Path("test data")
SHORT_AUDIO = TEST_DATA_DIR / "short_conversation.m4a"
LONG_AUDIO = TEST_DATA_DIR / "long_conversatio.m4a"  # Note the typo in the filename


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Clean up
    shutil.rmtree(temp_dir)


def run_transcription_command(audio_file, output_dir, start_time=0, end_time=None, max_chunk_size=None):
    """Run the transcription command and return the result."""
    cmd = ["python", "-m", "pyhearingai", str(audio_file), "-o", str(output_dir), "--force"]
    
    if start_time is not None:
        cmd.extend(["--start-time", str(start_time)])
    
    if end_time is not None:
        cmd.extend(["--end-time", str(end_time)])
    
    if max_chunk_size is not None:
        cmd.extend(["--chunk-size", str(max_chunk_size)])
    
    # Run the command and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False  # Don't raise an exception on non-zero exit
    )
    
    return result


def verify_output_files(output_dir, base_filename):
    """Verify that the expected output files exist."""
    # Check for the main transcript file, could be in output_dir or in the same directory as the input file
    transcript_file_in_output_dir = output_dir / f"{base_filename}.txt"
    transcript_file_in_input_dir = TEST_DATA_DIR / f"{base_filename}.txt"
    
    # Try both locations
    if transcript_file_in_output_dir.exists():
        transcript_file = transcript_file_in_output_dir
    elif transcript_file_in_input_dir.exists():
        transcript_file = transcript_file_in_input_dir
    else:
        assert False, f"Transcript file not found in {output_dir} or {TEST_DATA_DIR}"
    
    # Check that the transcript has content
    with open(transcript_file, "r") as f:
        content = f.read()
    
    assert len(content) > 0, "Transcript file is empty"
    
    # Check for other potential output formats
    potential_formats = ["json", "srt", "vtt", "md"]
    found_formats = []
    
    # Check both locations for additional formats
    for fmt in potential_formats:
        format_file_in_output_dir = output_dir / f"{base_filename}.{fmt}"
        format_file_in_input_dir = TEST_DATA_DIR / f"{base_filename}.{fmt}"
        
        if format_file_in_output_dir.exists():
            found_formats.append(fmt)
        elif format_file_in_input_dir.exists():
            found_formats.append(fmt)
    
    # We should have at least one format (txt)
    assert len(content) > 0, f"No content found in transcript file"
    
    return content, found_formats


@pytest.mark.on_demand
def test_short_conversation_end_to_end(temp_output_dir):
    """Test the end-to-end pipeline with a short conversation."""
    # Skip if the test file doesn't exist
    if not SHORT_AUDIO.exists():
        pytest.skip(f"Test file {SHORT_AUDIO} not found")
    
    print(f"\nRunning test with audio file: {SHORT_AUDIO}")
    print(f"Output directory: {temp_output_dir}")
    
    # Run the transcription command with a 4-minute limit
    result = run_transcription_command(
        SHORT_AUDIO,
        temp_output_dir,
        start_time=0,
        end_time=240  # 4 minutes
    )
    
    # Print command output for debugging
    print("\nCommand stdout:")
    print(result.stdout)
    print("\nCommand stderr:")
    print(result.stderr)
    
    # List files in output directory
    print("\nFiles in output directory:")
    for file in temp_output_dir.iterdir():
        print(f"  {file.name} ({file.stat().st_size} bytes)")
        
    # List files in test data directory
    print("\nFiles in test data directory:")
    for file in TEST_DATA_DIR.iterdir():
        print(f"  {file.name} ({file.stat().st_size} bytes)")
    
    # Check that the command completed successfully
    assert result.returncode == 0, f"Command failed with output: {result.stderr}"
    
    # Verify the output files
    base_filename = SHORT_AUDIO.stem
    content, formats = verify_output_files(temp_output_dir, base_filename)
    
    # Log the results
    print(f"\nTranscription completed successfully for {SHORT_AUDIO.name}")
    print(f"Output formats: {', '.join(formats)}")
    print(f"Transcript length: {len(content)} characters")
    print(f"First 100 characters: {content[:100]}...")


@pytest.mark.on_demand
def test_long_conversation_end_to_end(temp_output_dir):
    """Test the end-to-end pipeline with a long conversation."""
    # Skip if the test file doesn't exist
    if not LONG_AUDIO.exists():
        pytest.skip(f"Test file {LONG_AUDIO} not found")
    
    # Run the transcription command with chunking enabled
    result = run_transcription_command(
        LONG_AUDIO,
        temp_output_dir,
        max_chunk_size=300  # 5-minute chunks
    )
    
    # Check that the command completed successfully
    assert result.returncode == 0, f"Command failed with output: {result.stderr}"
    
    # Verify the output files
    base_filename = LONG_AUDIO.stem
    content, formats = verify_output_files(temp_output_dir, base_filename)
    
    # Log the results
    print(f"\nTranscription completed successfully for {LONG_AUDIO.name}")
    print(f"Output formats: {', '.join(formats)}")
    print(f"Transcript length: {len(content)} characters")
    print(f"First 100 characters: {content[:100]}...")


@pytest.mark.on_demand
def test_size_constrained_conversion(temp_output_dir):
    """Test the size-constrained audio conversion with a specific size limit."""
    # Skip if the test file doesn't exist
    if not SHORT_AUDIO.exists():
        pytest.skip(f"Test file {SHORT_AUDIO} not found")
    
    # Create a custom command that forces size constraints
    cmd = [
        "python", "-m", "pyhearingai", 
        str(SHORT_AUDIO), 
        "-o", str(temp_output_dir),
        "--force",
        "--start-time", "0",
        "--end-time", "60",  # Just process 1 minute
        "--max-file-size", "1",  # 1MB size limit
        "--verbose"  # Enable verbose logging
    ]
    
    # Run the command and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
    )
    
    # The command might fail due to size constraints, but we should see evidence
    # of the size-aware conversion attempting to reduce the file size
    
    # Check for evidence of size-aware conversion in the logs
    size_aware_indicators = [
        "size constraint",
        "quality adjustments",
        "Attempting quality adjustments",
        "conversion exceeds size limit"
    ]
    
    found_indicators = []
    for indicator in size_aware_indicators:
        if indicator.lower() in result.stderr.lower() or indicator.lower() in result.stdout.lower():
            found_indicators.append(indicator)
    
    # We should see at least one indicator of size-aware conversion
    assert len(found_indicators) > 0, "No evidence of size-aware conversion found in the output"
    
    print(f"\nSize-constrained conversion test completed")
    print(f"Found indicators of size-aware conversion: {', '.join(found_indicators)}")


if __name__ == "__main__":
    # This allows running the tests directly from the command line
    pytest.main(["-v", __file__]) 