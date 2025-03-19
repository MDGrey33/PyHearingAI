# Test Plan for PyHearingAI

## Overview

This document outlines the testing strategy for PyHearingAI, focusing on the status of existing tests, identified issues, and recommendations for future test and feature development.

## Current Test Status

### CLI Tests

All CLI tests are now passing or appropriately skipped. The main test file (`tests/test_cli.py`) contains tests for:

- Basic CLI functionality (help, version, error handling)
- Transcription options
- Job management (listing, resuming)

Tests for the following features are currently skipped as they are not yet implemented in the CLI:

1. Job cancellation (`--cancel`)
2. Job deletion (`--delete`)

### Repository Tests

Most repository tests are passing with one skipped test:

- `test_serialization_edge_cases` - Skipped because the `ProcessingJob` class doesn't have a `processing_options` attribute which the test expects.

## Issues Identified

1. The CLI doesn't currently support job cancellation or deletion operations.
2. The `ProcessingJob` class doesn't have a `processing_options` attribute which makes complex serialization testing difficult.
3. Other parts of the codebase have failing tests related to `ProcessingJob` constructor parameter issues.

## Recommendations

### Short-term Improvements

1. **Implement CLI Job Management Features**:
   - Add `--cancel` option to the CLI argument parser to allow canceling jobs
   - Add `--delete` option to the CLI argument parser to allow deleting jobs
   - Test these features with the existing tests (which are currently skipped)

2. **Fix ProcessingJob Class**:
   - Add a `processing_options` attribute to the `ProcessingJob` class
   - Ensure the constructor properly handles all parameters (status, output_path, processing_options, etc.)
   - Update tests to use the correct pattern for creating and initializing ProcessingJob objects

### Long-term Improvements

1. **Improve Test Coverage**:
   - The current test coverage is 28.59%, far below the required 89.5%
   - Focus on testing core functionality and high-risk areas first
   - Consider using tools like pytest-cov to identify untested code paths

2. **Mock External Dependencies**:
   - Improve integration tests by properly mocking external dependencies
   - Address the batch_size parameter error in diarization tests

3. **Refactor Tests**:
   - Organize tests more systematically, grouping related tests together
   - Reduce duplication in test setup code through better use of fixtures
   - Ensure consistent test naming and documentation

## Implementation Plan

### Phase 1: Fix Existing Tests

1. Update `ProcessingJob` to support all required attributes and parameters
2. Fix constructor parameter issues in tests
3. Ensure repository serialization tests pass

### Phase 2: Implement Missing CLI Features

1. Add `--cancel` option to CLI:
   ```python
   input_group.add_argument("--cancel", type=str, help="Cancel the processing job with the specified ID")
   ```

2. Add `--delete` option to CLI:
   ```python
   input_group.add_argument("--delete", type=str, help="Delete the processing job with the specified ID")
   ```

3. Implement handlers in the `main()` function:
   ```python
   # Handle cancel command
   if args.cancel:
       job_id = args.cancel
       job_repo = JsonJobRepository()
       job = job_repo.get_by_id(job_id)

       if not job:
           print(f"Error: Job not found with ID: {job_id}", file=sys.stderr)
           return 1

       # Update job status to FAILED to indicate cancellation
       job.status = ProcessingStatus.FAILED
       job_repo.save(job)
       print(f"Job canceled: {job_id}")
       return 0

   # Handle delete command
   if args.delete:
       job_id = args.delete
       job_repo = JsonJobRepository()
       job = job_repo.get_by_id(job_id)

       if not job:
           print(f"Error: Job not found with ID: {job_id}", file=sys.stderr)
           return 1

       # Delete the job
       if job_repo.delete(job_id):
           print(f"Job deleted: {job_id}")
           return 0
       else:
           print(f"Error: Failed to delete job with ID: {job_id}", file=sys.stderr)
           return 1
   ```

4. Enable the skipped tests by removing the `@pytest.mark.skip` decorators

### Phase 3: Improve Test Coverage

1. Identify areas with low coverage
2. Write additional tests focusing on high-risk or complex functionality
3. Review and optimize existing tests

## Conclusion

The test suite provides good coverage for the CLI and repository components, but several features are not yet implemented or tested. By following the recommendations in this test plan, we can improve both the feature set and test coverage of PyHearingAI.

---

Last updated: 2025-03-19
