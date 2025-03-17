#!/usr/bin/env python3
"""
Test script for the ResponsesReconciliationAdapter.

This script tests the ResponsesReconciliationAdapter implementation
for token-efficient reconciliation of diarization and transcription results.
"""

import os
import sys
import logging
import time
from pathlib import Path

from pyhearingai.application.transcribe import transcribe
from pyhearingai.core.idempotent import ProcessingJob
from pyhearingai.reconciliation.service import ReconciliationService
from pyhearingai.infrastructure.repositories.json_repositories import (
    JsonJobRepository,
    JsonChunkRepository,
)
from pyhearingai.diarization.repositories.diarization_repository import DiarizationRepository
from pyhearingai.transcription.repositories.transcription_repository import TranscriptionRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("responses_adapter_test")

def test_responses_adapter(audio_path, output_path=None, start_time=None, end_time=None):
    """
    Test the ResponsesReconciliationAdapter with a small audio file.
    
    Args:
        audio_path: Path to the audio file to process
        output_path: Optional path to save the output
        start_time: Optional start time in seconds to process
        end_time: Optional end time in seconds to process
    """
    start = time.time()
    logger.info(f"Testing ResponsesReconciliationAdapter with audio: {audio_path}")
    
    # Process the job using the standard transcribe function but use ResponsesReconciliationAdapter
    result = transcribe(
        audio_path=audio_path,
        output_path=output_path,
        start_time=start_time,
        end_time=end_time,
        use_responses_api=True,  # Enable the Responses API adapter
        verbose=True
    )
    
    if result:
        logger.info(f"✅ Test completed successfully: {len(result.segments)} segments processed")
        logger.info(f"Total processing time: {time.time() - start:.2f} seconds")
        
        # Print a summary of the transcript
        if result.segments:
            logger.info("Transcript summary:")
            for i, segment in enumerate(result.segments[:3]):
                logger.info(f"  {i+1}. Speaker {segment.speaker_id}: {segment.text[:50]}...")
            
            if len(result.segments) > 3:
                logger.info(f"  ... and {len(result.segments) - 3} more segments")
        
        return True
    else:
        logger.error("❌ Test failed: No result returned")
        return False

def test_with_existing_job(job_id):
    """
    Test the ResponsesReconciliationAdapter with an existing job.
    
    Args:
        job_id: ID of an existing job to reconcile
    """
    logger.info(f"Testing ResponsesReconciliationAdapter with existing job: {job_id}")
    
    # Initialize repositories
    job_repo = JsonJobRepository()
    
    # Get the job
    job = job_repo.get_by_id(job_id)
    if not job:
        logger.error(f"Job not found: {job_id}")
        return False
    
    logger.info(f"Found job: {job.id} with audio: {job.original_audio_path}")
    
    # Initialize the reconciliation service with ResponsesReconciliationAdapter
    service = ReconciliationService(use_responses_api=True)
    
    start = time.time()
    try:
        # Force reconciliation to bypass cache
        segments = service.reconcile(job, force=True)
        
        logger.info(f"✅ Reconciliation completed successfully: {len(segments)} segments")
        logger.info(f"Total reconciliation time: {time.time() - start:.2f} seconds")
        
        # Print a summary of the reconciled segments
        if segments:
            logger.info("Reconciled segments summary:")
            for i, segment in enumerate(segments[:3]):
                logger.info(f"  {i+1}. Speaker {segment.speaker_id}: {segment.text[:50]}...")
            
            if len(segments) > 3:
                logger.info(f"  ... and {len(segments) - 3} more segments")
        
        return True
    except Exception as e:
        logger.error(f"❌ Reconciliation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_medium_audio(audio_path, duration=300):
    """
    Test with a medium-sized audio file, processing a specified duration.
    
    Args:
        audio_path: Path to the audio file
        duration: Duration in seconds to process (default: 5 minutes)
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Testing with medium-sized audio: {audio_path} (processing {duration}s)")
    
    # Process only the specified duration
    start_time = 0
    end_time = duration
    
    # Use a tempfile for output
    import tempfile
    output_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    output_path = output_file.name
    output_file.close()
    
    try:
        # Run the test
        success = test_responses_adapter(
            audio_path=audio_path,
            output_path=output_path,
            start_time=start_time,
            end_time=end_time
        )
        
        # If successful, check the output file
        if success and os.path.exists(output_path):
            with open(output_path, "r") as f:
                content = f.read()
                
            # Log some stats about the output
            line_count = len(content.splitlines())
            word_count = len(content.split())
            logger.info(f"Output stats: {line_count} lines, {word_count} words")
            logger.info(f"Output saved to: {output_path}")
            
        return success
    except Exception as e:
        logger.error(f"Error in medium audio test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run the test script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the ResponsesReconciliationAdapter")
    parser.add_argument("--audio", help="Path to an audio file to process")
    parser.add_argument("--output", help="Path to save the output")
    parser.add_argument("--job-id", help="ID of an existing job to reconcile")
    parser.add_argument("--start-time", type=float, help="Start time in seconds")
    parser.add_argument("--end-time", type=float, help="End time in seconds")
    parser.add_argument("--medium", action="store_true", help="Run the medium audio test")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds for medium test")
    
    args = parser.parse_args()
    
    if args.job_id:
        # Test with an existing job
        success = test_with_existing_job(args.job_id)
    elif args.medium:
        # Test with a medium-sized audio file
        if args.audio:
            success = test_medium_audio(args.audio, args.duration)
        else:
            # Try to find a default medium-sized test file
            medium_file = Path("test data/long_conversatio.m4a")
            if medium_file.exists():
                logger.info(f"Using default medium test file: {medium_file}")
                success = test_medium_audio(str(medium_file), args.duration)
            else:
                logger.error("No medium-sized audio file specified, and default file not found")
                parser.print_help()
                return 1
    elif args.audio:
        # Test with a new audio file
        success = test_responses_adapter(args.audio, args.output, args.start_time, args.end_time)
    else:
        # Default to a known test file if it exists
        test_file = Path("test data/short_conversation.m4a")
        if test_file.exists():
            logger.info(f"Using default test file: {test_file}")
            success = test_responses_adapter(str(test_file), None, args.start_time, args.end_time)
        else:
            logger.error("No audio file or job ID specified, and default test file not found")
            parser.print_help()
            return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 