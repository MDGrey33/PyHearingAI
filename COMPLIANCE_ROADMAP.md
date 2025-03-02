# Code Compliance Improvement Roadmap

This document outlines our plan to gradually improve code compliance and quality in the PyHearingAI project.

## Current Configuration

- **flake8**: Lenient configuration that ignores many common issues:
  - Line length limit: 120 characters
  - Ignored rules: E203, W503, E501, F401, F541, E402, F841, F403, F405
  - Special exceptions for `__init__.py` and test files

- **pytest**: Coverage requirement set to minimum (1%)
  - Currently skipping `test_whisper_openai_transcriber_basic` test

## Phase 1: Basic Compliance (Current)

- ✅ All pre-commit hooks passing with current code
- ✅ Minimum test coverage established
- ✅ Basic formatting with black and isort

## Phase 2: Improved Code Quality

- [ ] Fix unused imports (F401)
- [ ] Fix line length issues (E501)
- [ ] Resolve wildcard imports (F403, F405)
- [ ] Improve test coverage to 25%
- [ ] Fix the failing transcriber test

## Phase 3: Comprehensive Compliance

- [ ] Increase test coverage to 70%
- [ ] Enable more flake8 checks
- [ ] Implement type hints and mypy validation
- [ ] Add docstring coverage

## Phase 4: Production-Ready

- [ ] Test coverage minimum 80%
- [ ] Full compliance with flake8 standards
- [ ] Complete documentation
- [ ] Full type annotation

## How to Update

To gradually increase compliance requirements:

1. Edit the `.flake8` file to remove ignored rules
2. Update the test coverage threshold in `.pre-commit-config.yaml` and `pyproject.toml`
3. Remove the skip for specific tests when they are fixed

As we fix issues in the codebase, we will update the compliance requirements to ensure code quality continues to improve.
