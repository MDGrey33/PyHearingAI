#!/usr/bin/env python
"""
Cleanup script for PyHearingAI cache.

This script cleans up the PyHearingAI cache directory to free up disk space.
It removes cached chunks and job data that are no longer needed.
"""

import os
import shutil
from pathlib import Path
import argparse


def get_cache_dir():
    """Get the PyHearingAI cache directory."""
    home_dir = Path.home()
    return home_dir / ".local" / "share" / "pyhearingai"


def cleanup_chunks(cache_dir, keep_latest=5):
    """
    Clean up chunk directories, keeping only the most recent ones.

    Args:
        cache_dir: The cache directory
        keep_latest: Number of most recent jobs to keep
    """
    chunks_dir = cache_dir / "chunks"
    if not chunks_dir.exists():
        print(f"Chunks directory not found: {chunks_dir}")
        return 0

    # Get all job directories
    job_dirs = []
    for job_dir in chunks_dir.iterdir():
        if job_dir.is_dir():
            job_dirs.append((job_dir, job_dir.stat().st_mtime))

    # Sort by modification time (newest first)
    job_dirs.sort(key=lambda x: x[1], reverse=True)

    # Keep the most recent ones
    jobs_to_keep = job_dirs[:keep_latest] if len(job_dirs) > keep_latest else job_dirs
    jobs_to_delete = job_dirs[keep_latest:] if len(job_dirs) > keep_latest else []

    print(f"Found {len(job_dirs)} job directories")
    print(f"Keeping {len(jobs_to_keep)} recent jobs")
    print(f"Deleting {len(jobs_to_delete)} old jobs")

    # Delete old job directories
    freed_space = 0
    for job_dir, _ in jobs_to_delete:
        try:
            # Calculate size before deletion
            job_size = sum(f.stat().st_size for f in job_dir.glob("**/*") if f.is_file())
            freed_space += job_size

            # Delete the directory
            shutil.rmtree(job_dir)
            print(f"Deleted {job_dir.name} ({job_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"Error deleting {job_dir}: {e}")

    return freed_space


def cleanup_jobs_db(cache_dir):
    """Clean up the jobs database."""
    jobs_db = cache_dir / "jobs.json"
    if not jobs_db.exists():
        print(f"Jobs database not found: {jobs_db}")
        return 0

    # Make a backup
    backup_path = jobs_db.with_suffix(".json.bak")
    shutil.copy2(jobs_db, backup_path)
    print(f"Created backup of jobs database: {backup_path}")

    # Get file size
    jobs_db_size = jobs_db.stat().st_size

    # Replace with empty jobs array
    with open(jobs_db, "w") as f:
        f.write("[]")

    print(f"Reset jobs database: {jobs_db}")
    return jobs_db_size


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clean up PyHearingAI cache")
    parser.add_argument(
        "--keep", type=int, default=5, help="Number of most recent jobs to keep (default: 5)"
    )
    parser.add_argument(
        "--reset-db", action="store_true", help="Reset the jobs database (creates a backup first)"
    )
    args = parser.parse_args()

    cache_dir = get_cache_dir()
    print(f"Cache directory: {cache_dir}")

    # Clean up chunks
    freed_space_chunks = cleanup_chunks(cache_dir, args.keep)

    # Clean up jobs database if requested
    freed_space_db = 0
    if args.reset_db:
        freed_space_db = cleanup_jobs_db(cache_dir)

    # Report total freed space
    total_freed = freed_space_chunks + freed_space_db
    print(f"Total space freed: {total_freed / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
