"""
Pyannote diarizer implementation.

This module provides a concrete implementation of the Diarizer port
using the Pyannote.audio library for speaker diarization.
"""

import json
import logging
import os
import re
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import tempfile
import filelock
import torch
import multiprocessing
import platform
import subprocess

# Import the real Pyannote pipeline
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

from pyhearingai.core.models import DiarizationSegment
from pyhearingai.core.ports import Diarizer
from pyhearingai.infrastructure.registry import register_diarizer

logger = logging.getLogger(__name__)

class DiarizationTimeoutError(Exception):
    """Exception raised when diarization exceeds the timeout limit."""
    pass

@register_diarizer("pyannote")
class PyannoteDiarizer(Diarizer):
    """Diarizer implementation using the Pyannote.audio library."""

    _shared_pipeline = None
    _lock = filelock.FileLock("/tmp/pyannote_init.lock")

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the diarizer."""
        self._api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self._pipeline = None
        
        # Calculate optimal number of workers for the executor
        # For M3/M2 chips, use more workers to maximize performance
        cpu_count = multiprocessing.cpu_count()
        # Use half the available cores for the executor to avoid overwhelming the system
        # but ensure we have at least 2 workers for parallel operations
        optimal_workers = max(2, cpu_count // 2)
        logger.debug(f"Initializing ThreadPoolExecutor with {optimal_workers} workers (system has {cpu_count} cores)")
        self._executor = ThreadPoolExecutor(max_workers=optimal_workers)
        
        # Determine and store the best available device
        self._device = self._determine_device()
        logger.info(f"Pyannote will use device: {self._device}")
        
        # Create cache directory with proper permissions
        cache_dir = Path(os.path.expanduser("~/.cache/torch/pyannote"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _determine_device(self):
        """Determine the best available device for PyTorch, prioritizing MPS for Apple Silicon."""
        # Check for MPS (Apple Silicon M-series) first
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        # Then check for CUDA GPU
        elif torch.cuda.is_available():
            return "cuda"
        # Fall back to CPU
        else:
            return "cpu"

    @classmethod
    def _initialize_shared_pipeline(cls, use_auth_token: Optional[str] = None):
        """Initialize the shared pipeline with proper locking."""
        if cls._shared_pipeline is None:
            with cls._lock:
                if cls._shared_pipeline is None:
                    try:
                        # Initialize pipeline with device placement
                        pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization@2.1",
                            use_auth_token=use_auth_token
                        )
                        
                        # Configure pipeline for best performance based on available hardware
                        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            # For Apple Silicon (M1/M2/M3), enable MPS
                            logger.info("Configuring pipeline for Apple Silicon with MPS")
                            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                        
                        cls._shared_pipeline = pipeline
                        logger.info("Initialized shared PyAnnote pipeline")
                    except Exception as e:
                        logger.error(f"Failed to initialize shared pipeline: {str(e)}")
                        raise

    @property
    def pipeline(self):
        """Get the diarization pipeline, initializing if needed."""
        if self._pipeline is None:
            try:
                # Initialize shared pipeline if not done
                self._initialize_shared_pipeline(self._api_key)
                # Use the shared pipeline
                self._pipeline = self._shared_pipeline
            except Exception as e:
                logger.error(f"Error initializing pipeline: {str(e)}")
                raise
        return self._pipeline

    def _run_diarization_with_timeout(self, audio_path: str, timeout: int = 7200) -> any:
        """
        Run diarization with a timeout.
        
        Args:
            audio_path: Path to the audio file
            timeout: Timeout in seconds (default: 2 hours)
            
        Returns:
            Diarization result
            
        Raises:
            DiarizationTimeoutError: If the process exceeds the timeout
            Exception: For other diarization errors
        """
        try:
            future = self._executor.submit(self.pipeline, audio_path)
            return future.result(timeout=timeout)
            
        except TimeoutError:
            # Clean up the pipeline on timeout
            self._pipeline = None
            raise DiarizationTimeoutError(f"Diarization timed out after {timeout} seconds")
        except Exception as e:
            self._pipeline = None
            raise Exception(f"Diarization failed: {str(e)}")

    def diarize(self, audio_path: Path, **kwargs) -> List[DiarizationSegment]:
        """
        Perform speaker diarization on an audio file using Pyannote.

        Args:
            audio_path: Path to the audio file to diarize
            **kwargs: Additional diarization options
                - api_key: Hugging Face API key (overrides environment variable)
                - output_dir: Directory to save output files (default: "content/diarization")
                - disable_progress: Disable progress hook for parallel processing
                - timeout: Timeout in seconds (default: 7200)
                - batch_size: Batch size for processing (default: auto)

        Returns:
            List of segments with speaker identification and timing information
            
        Raises:
            DiarizationTimeoutError: If the process exceeds the timeout
            FileNotFoundError: If the audio file is not found
            Exception: For other diarization errors
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Extract options
        api_key = kwargs.get("api_key", self._api_key)
        output_dir = kwargs.get("output_dir", "content/diarization")
        disable_progress = kwargs.get("disable_progress", False)
        timeout = kwargs.get("timeout", 7200)  # Default 2 hours
        batch_size = kwargs.get("batch_size", None)  # Auto-determine batch size

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results
        segments = []
        raw_segments = []
        device_info = "CPU"

        try:
            # Set device-specific optimizations
            if self._device == "mps":
                device_info = f"MPS (Apple Silicon)"
                logger.info(f"Using {device_info} for diarization")
                # Set environment variables for MPS
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                
                # Determine optimal batch size for Apple Silicon if not specified
                if batch_size is None:
                    # Check if we're on an M3 Max or similar high-end chip
                    is_high_end = False
                    cpu_count = multiprocessing.cpu_count()
                    
                    # Better detection for M3 Max
                    if platform.system() == "Darwin" and "arm" in platform.machine():
                        try:
                            # Try to detect M3 Max using system_profiler
                            result = subprocess.run(
                                ["system_profiler", "SPHardwareDataType"], 
                                capture_output=True, 
                                text=True, 
                                check=False
                            )
                            chip_info = result.stdout.lower()
                            is_high_end = "m3 max" in chip_info or "m3max" in chip_info or cpu_count >= 12
                            if "m3 max" in chip_info or "m3max" in chip_info:
                                logger.info("Detected M3 Max processor for diarization")
                        except Exception as e:
                            logger.debug(f"Error detecting processor type: {e}")
                            # Fallback to CPU count for detection
                            is_high_end = cpu_count >= 12
                    
                    # M3 Max or high core count Apple Silicon can handle larger batches
                    if is_high_end:
                        batch_size = 64
                        logger.info(f"Using high-performance batch size {batch_size} for M3 Max")
                    else:
                        # Standard M-series
                        batch_size = 32
                        logger.info(f"Using standard batch size {batch_size} for Apple Silicon")
                else:
                    logger.info(f"Using user-specified batch size {batch_size} for MPS device")
                
            elif self._device == "cuda":
                # For CUDA devices
                device_info = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
                logger.info(f"Using {device_info} for diarization")
                
                # Optimize CUDA settings if batch size not specified
                if batch_size is None:
                    # Determine based on GPU memory
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # in GB
                    batch_size = max(16, min(128, int(gpu_mem * 4)))  # scale with GPU memory
                logger.info(f"Using batch size {batch_size} for CUDA device")
                
            else:
                # CPU optimizations
                device_info = "CPU"
                logger.info(f"Using {device_info} for diarization")
                
                # For CPUs, use a smaller batch size but utilize all cores
                if batch_size is None:
                    cpu_count = multiprocessing.cpu_count()
                    batch_size = max(8, min(32, cpu_count * 2))  # scale with CPU count
                logger.info(f"Using batch size {batch_size} for CPU with {multiprocessing.cpu_count()} cores")
                
                # Set optimal thread settings for CPU
                torch.set_num_threads(multiprocessing.cpu_count())
                
            # If we have a batch size from above, pass it to the pipeline
            if batch_size is not None:
                kwargs["batch_size"] = batch_size

            # Run diarization with timeout
            logger.info(f"Starting diarization with {timeout}s timeout")
            start_time = datetime.now()
            
            diarization = self._run_diarization_with_timeout(str(audio_path), timeout)
            
            logger.info(f"Diarization completed in {(datetime.now() - start_time).total_seconds():.2f}s")

            # Process results
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Normalize speaker format:
                # 1. Strip any existing "SPEAKER_" prefix to avoid double prefixing
                # 2. Ensure numeric format (01, 02, etc.) with leading zeros

                # Extract the actual speaker identifier (number or string)
                # Handle various cases like "SPEAKER_01", "01", "1", etc.
                if isinstance(speaker, str) and speaker.startswith("SPEAKER_"):
                    speaker_num = speaker[8:]  # Remove "SPEAKER_" prefix
                else:
                    speaker_num = speaker

                # Remove any additional 'SPEAKER' text that might be present
                if isinstance(speaker_num, str) and "SPEAKER" in speaker_num:
                    # Extract just the numeric or identifier part
                    # Try to extract numbers from the string
                    numbers = re.findall(r"\d+", speaker_num)
                    if numbers:
                        speaker_num = numbers[0]  # Use the first number found

                # Then format with leading zeros for single digits
                # Try to convert to int first to handle both numeric and string inputs
                try:
                    speaker_num = int(speaker_num)
                    speaker_id = f"SPEAKER_{speaker_num:02d}"  # Format with leading zero
                except (ValueError, TypeError):
                    # If conversion fails, use as is with prefix
                    speaker_id = f"SPEAKER_{speaker_num}"

                # Create raw segment dict (as in original)
                raw_segment = {"start": turn.start, "end": turn.end, "speaker": speaker_id}
                raw_segments.append(raw_segment)

                # Create a DiarizationSegment (for clean architecture)
                diarization_segment = DiarizationSegment(
                    speaker_id=speaker_id, start=turn.start, end=turn.end, score=1.0
                )
                segments.append(diarization_segment)

            # Save segments as JSON (as in original)
            segments_path = os.path.join(output_dir, "segments.json")
            with open(segments_path, "w") as f:
                json.dump(raw_segments, f, indent=2)

            # Create summary file (as in original)
            summary_path = os.path.join(output_dir, "diarization_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"Diarization completed at: {datetime.now().isoformat()}\n")
                f.write(f"Input file: {audio_path}\n")
                f.write(f"Number of segments: {len(raw_segments)}\n")
                f.write(f"Processing device: {device_info}\n")

                # Calculate some statistics
                total_duration = sum(seg["end"] - seg["start"] for seg in raw_segments)
                unique_speakers = len(set(seg["speaker"] for seg in raw_segments))

                f.write(f"Total audio duration processed: {total_duration:.2f} seconds\n")
                f.write(f"Number of unique speakers detected: {unique_speakers}\n")

            logger.debug(f"Found {len(segments)} diarization segments")
            return segments

        except Exception as e:
            # Log error and create error file (as in original)
            error_log_path = os.path.join(output_dir, "diarization_error.log")
            with open(error_log_path, "w") as f:
                f.write(f"Error in diarization:\n{str(e)}")

            logger.error(f"Error diarizing with Pyannote: {str(e)}")
            raise Exception(f"Diarization error: {str(e)}")

    def _mock_diarize(
        self, audio_path: Path, output_dir: str = "content/diarization"
    ) -> List[DiarizationSegment]:
        """Fallback mock implementation for testing without Pyannote."""
        logger.warning("Using mock diarization data - ONLY FOR TESTING!")

        # Generate mock segments based on file
        if "example_audio" in str(audio_path):
            # Original mock segments for the example audio
            mock_segments = [
                ("0", 0.0, 2.5),
                ("1", 2.7, 5.2),
                ("0", 5.4, 8.1),
                ("1", 8.3, 10.0),
                ("0", 10.2, 12.8),
            ]
        else:
            # Extended mock segments for other audio files
            mock_segments = [
                ("0", 0.0, 5.0),
                ("1", 5.0, 10.0),
                ("0", 10.0, 15.0),
                ("1", 15.0, 20.0),
                ("0", 20.0, 25.0),
                ("1", 25.0, 30.0),
                ("0", 30.0, 35.0),
                ("1", 35.0, 40.0),
            ]

        # Convert to both DiarizationSegment objects and raw dict segments
        segments = []
        raw_segments = []

        for speaker_num, start, end in mock_segments:
            # Extract the actual speaker identifier (number or string)
            # Handle various cases like "SPEAKER_01", "01", "1", etc.
            if isinstance(speaker_num, str) and speaker_num.startswith("SPEAKER_"):
                speaker_num = speaker_num[8:]  # Remove "SPEAKER_" prefix

            # Remove any additional 'SPEAKER' text that might be present
            if isinstance(speaker_num, str) and "SPEAKER" in speaker_num:
                # Try to extract numbers from the string
                numbers = re.findall(r"\d+", speaker_num)
                if numbers:
                    speaker_num = numbers[0]  # Use the first number found

            # Format with leading zeros for single digits
            try:
                speaker_num = int(speaker_num)
                speaker_id = f"SPEAKER_{speaker_num:02d}"  # Format with leading zero
            except (ValueError, TypeError):
                speaker_id = f"SPEAKER_{speaker_num}"

            # Create raw segment dict
            raw_segment = {"start": start, "end": end, "speaker": speaker_id}
            raw_segments.append(raw_segment)

            # Create DiarizationSegment
            segment = DiarizationSegment(speaker_id=speaker_id, start=start, end=end, score=1.0)
            segments.append(segment)

        # Save mock outputs to match real output format
        # Save segments as JSON
        os.makedirs(output_dir, exist_ok=True)
        segments_path = os.path.join(output_dir, "segments.json")
        with open(segments_path, "w") as f:
            json.dump(raw_segments, f, indent=2)

        # Create summary file
        summary_path = os.path.join(output_dir, "diarization_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Mock diarization completed at: {datetime.now().isoformat()}\n")
            f.write(f"Input file: {audio_path}\n")
            f.write(f"Number of segments: {len(raw_segments)}\n")
            f.write(f"Processing device: CPU (mock)\n")

            # Calculate some statistics
            total_duration = sum(seg["end"] - seg["start"] for seg in raw_segments)
            unique_speakers = len(set(seg["speaker"] for seg in raw_segments))

            f.write(f"Total audio duration processed: {total_duration:.2f} seconds\n")
            f.write(f"Number of unique speakers detected: {unique_speakers}\n")
            f.write("NOTE: THIS IS MOCK DATA - NOT REAL DIARIZATION\n")

        logger.debug(f"Created {len(segments)} mock diarization segments")
        return segments

    def close(self):
        """Release any resources used by the diarizer."""
        if self._pipeline is not None:
            # Clean up resources if needed
            try:
                # Explicitly delete the pipeline to free GPU memory
                del self._pipeline
                self._pipeline = None
                
                # Call torch garbage collector if available
                if PYANNOTE_AVAILABLE:
                    if self._device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif self._device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        # MPS doesn't have an explicit memory management function like CUDA,
                        # but we can force a garbage collection
                        import gc
                        gc.collect()
                    
                logger.debug(f"Pyannote diarizer resources released from {self._device} device")
            except Exception as e:
                logger.warning(f"Error while closing Pyannote diarizer: {str(e)}")
                
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.close()
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=False)
