#!/usr/bin/env python3
"""
Cleanup script for PyHearingAI.

This script performs a comprehensive cleanup of all temporary files,
caches, and processing artifacts created by PyHearingAI.
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Define paths
HOME = Path.home()
PYHEARING_SHARE = HOME / ".local" / "share" / "pyhearingai"
PYANNOTE_CACHE = HOME / ".cache" / "torch" / "pyannote"
HUGGINGFACE_CACHE = HOME / ".cache" / "huggingface"
TMP_DIR = Path("/tmp")


class CleanupTask:
    """Represents a cleanup task with its status and any errors encountered."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.success = False
        self.error = None
        self.items_removed = 0

    def __str__(self):
        status = "✅" if self.success else "❌"
        result = f"{status} {self.name}: {self.description}"
        if self.items_removed > 0:
            result += f" ({self.items_removed} items removed)"
        if self.error:
            result += f"\n    Error: {self.error}"
        return result


def kill_processes(pattern: str) -> CleanupTask:
    """Kill processes matching the given pattern."""
    task = CleanupTask("Kill Processes", f"Terminating processes matching '{pattern}'")
    try:
        # Use pkill to terminate processes
        result = subprocess.run(["pkill", "-f", pattern], capture_output=True, text=True)
        task.success = result.returncode in (0, 1)  # 1 means no processes found, which is ok
        if result.stderr:
            task.error = result.stderr
    except Exception as e:
        task.error = str(e)
        task.success = False
    return task


def remove_directory_contents(path: Path, pattern: str = "*") -> CleanupTask:
    """Remove contents of a directory matching the given pattern."""
    task = CleanupTask("Remove Directory", f"Cleaning {path}")
    try:
        if not path.exists():
            task.success = True
            return task

        for item in path.glob(pattern):
            if item.is_file():
                item.unlink()
                task.items_removed += 1
            elif item.is_dir():
                shutil.rmtree(item)
                task.items_removed += 1
        task.success = True
    except Exception as e:
        task.error = str(e)
        task.success = False
    return task


def cleanup_pyhearing_directories() -> List[CleanupTask]:
    """Clean up PyHearingAI directories."""
    tasks = []

    # Clean up jobs directory
    jobs_dir = PYHEARING_SHARE / "jobs"
    task = remove_directory_contents(jobs_dir)
    task.name = "Clean Jobs"
    tasks.append(task)

    # Clean up chunks directory
    chunks_dir = PYHEARING_SHARE / "chunks"
    task = remove_directory_contents(chunks_dir)
    task.name = "Clean Chunks"
    tasks.append(task)

    # Clean up diarization directory
    diarization_dir = PYHEARING_SHARE / "diarization"
    task = remove_directory_contents(diarization_dir)
    task.name = "Clean Diarization"
    tasks.append(task)

    return tasks


def cleanup_cache_directories() -> List[CleanupTask]:
    """Clean up cache directories."""
    tasks = []

    # Clean up PyAnnote cache
    task = remove_directory_contents(PYANNOTE_CACHE)
    task.name = "Clean PyAnnote Cache"
    tasks.append(task)

    # Clean up specific HuggingFace model caches
    pattern = "models--pyannote--*"
    task = remove_directory_contents(HUGGINGFACE_CACHE / "hub", pattern)
    task.name = "Clean HuggingFace Cache"
    tasks.append(task)

    return tasks


def cleanup_temp_files() -> List[CleanupTask]:
    """Clean up temporary files."""
    tasks = []

    # Clean up /tmp directory
    task = remove_directory_contents(TMP_DIR, "pyhearingai*")
    task.name = "Clean Temp Files"
    tasks.append(task)

    # Clean up audio chunks in current directory
    patterns = ["converted.wav", "chunk_*.wav"]
    for pattern in patterns:
        task = remove_directory_contents(Path.cwd(), pattern)
        task.name = f"Clean {pattern}"
        tasks.append(task)

    return tasks


def verify_directories() -> CleanupTask:
    """Verify that all directories are in a clean state."""
    task = CleanupTask("Verify Directories", "Checking directory structure")
    try:
        # List of directories to verify
        dirs_to_check = [
            PYHEARING_SHARE / "jobs",
            PYHEARING_SHARE / "chunks",
            PYHEARING_SHARE / "diarization",
        ]

        # Create directories if they don't exist
        for dir_path in dirs_to_check:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Verify they're empty or only contain expected empty subdirectories
        for dir_path in dirs_to_check:
            files = list(dir_path.glob("*"))
            if any(f.is_file() for f in files):
                raise Exception(f"Directory {dir_path} contains files")

        task.success = True
    except Exception as e:
        task.error = str(e)
        task.success = False

    return task


def main(args: List[str] = None) -> int:
    """Main entry point for the cleanup script."""
    if args is None:
        args = sys.argv[1:]

    logger.info("Starting PyHearingAI cleanup...")
    all_tasks = []

    # Kill any running processes
    all_tasks.append(kill_processes("transcribe"))

    # Clean up directories
    all_tasks.extend(cleanup_pyhearing_directories())
    all_tasks.extend(cleanup_cache_directories())
    all_tasks.extend(cleanup_temp_files())

    # Verify directories
    all_tasks.append(verify_directories())

    # Print results
    logger.info("\nCleanup Results:")
    for task in all_tasks:
        logger.info(str(task))

    # Calculate success
    success = all(task.success for task in all_tasks)
    total_items = sum(task.items_removed for task in all_tasks)

    logger.info("\nSummary:")
    logger.info(f"Total items removed: {total_items}")
    logger.info(f"Status: {'Success' if success else 'Failed'}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
