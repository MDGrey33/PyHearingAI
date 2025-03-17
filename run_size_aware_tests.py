#!/usr/bin/env python
"""
Run the size-aware audio processing end-to-end tests.

This script provides a convenient way to run the size-aware audio processing
end-to-end tests on demand.

Usage:
    python run_size_aware_tests.py [--short | --long | --size-constrained | --all]

Options:
    --short            Run only the short conversation test
    --long             Run only the long conversation test
    --size-constrained Run only the size-constrained conversion test
    --all              Run all tests (default)
"""

import argparse
import subprocess
import sys


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run size-aware audio processing end-to-end tests")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--short", action="store_true", help="Run only the short conversation test")
    group.add_argument("--long", action="store_true", help="Run only the long conversation test")
    group.add_argument(
        "--size-constrained",
        action="store_true",
        help="Run only the size-constrained conversion test",
    )
    group.add_argument("--all", action="store_true", help="Run all tests (default)")

    return parser.parse_args()


def run_tests(args):
    """Run the specified tests."""
    test_file = "tests/integration/test_size_aware_end_to_end.py"

    if args.short:
        test_spec = f"{test_file}::test_short_conversation_end_to_end"
    elif args.long:
        test_spec = f"{test_file}::test_long_conversation_end_to_end"
    elif args.size_constrained:
        test_spec = f"{test_file}::test_size_constrained_conversion"
    else:  # Run all tests by default
        test_spec = test_file

    cmd = ["python", "-m", "pytest", "-v", test_spec]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    return result.returncode


if __name__ == "__main__":
    args = parse_args()

    # If no arguments are provided, default to running all tests
    if not (args.short or args.long or args.size_constrained):
        args.all = True

    sys.exit(run_tests(args))
