#!/usr/bin/env python3
"""
Test script for the ReconciliationService.

This script demonstrates how to use the ReconciliationService to reconcile
diarization and transcription results into a coherent final output.
"""

import argparse
import logging
import sys
from pathlib import Path

from pyhearingai.core.idempotent import ProcessingJob, ProcessingStatus
from pyhearingai.reconciliation.service import ReconciliationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    """Run the test script."""
    parser = argparse.ArgumentParser(description="Test the ReconciliationService")
    parser.add_argument("job_id", help="ID of the job to reconcile")
    parser.add_argument("--output", type=str, help="Path to save the output file")
    parser.add_argument(
        "--format", choices=["txt", "json", "srt", "vtt", "md"], default="txt", help="Output format"
    )
    parser.add_argument("--model", default="gpt-4", help="GPT model to use")
    parser.add_argument(
        "--force", action="store_true", help="Force reconciliation even if results exist"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure we have a job ID
    job_id = args.job_id

    # First, retrieve the job from the repository
    from pyhearingai.infrastructure.repositories.json_repositories import JsonJobRepository

    job_repo = JsonJobRepository()
    job = job_repo.get_by_id(job_id)

    if not job:
        logger.error(f"Job not found: {job_id}")
        return 1

    logger.info(f"Retrieved job {job.id} for reconciliation")

    # Initialize the reconciliation service
    service = ReconciliationService(model=args.model)

    try:
        # Reconcile the results
        logger.info("Reconciling diarization and transcription results...")
        segments = service.reconcile(job, force=args.force)

        logger.info(f"Reconciliation completed with {len(segments)} segments")

        # Format and save the output if requested
        if args.output:
            output_path = Path(args.output)
            service.save_output_file(job, output_path, args.format)
            logger.info(f"Output saved to {output_path}")
        else:
            # Just print a summary
            logger.info("Reconciled segments summary:")
            for i, segment in enumerate(segments[:5]):  # Show first 5 segments
                logger.info(
                    f"  {i+1}. Speaker {segment.speaker_id}: {segment.start:.2f}s-{segment.end:.2f}s: {segment.text[:50]}..."
                )

            if len(segments) > 5:
                logger.info(f"  ... and {len(segments) - 5} more segments")

            logger.info("To save the output, use the --output argument")

        return 0
    except Exception as e:
        logger.error(f"Error during reconciliation: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
