import os
import shutil
import tempfile
from pathlib import Path

import pytest


# Path fixtures
@pytest.fixture(scope="session")
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def example_audio_path(fixtures_dir):
    """Return the path to the example audio file."""
    return fixtures_dir / "example_audio.m4a"


@pytest.fixture(scope="session")
def reference_transcript_path(fixtures_dir):
    """Return the path to the reference transcript file."""
    return fixtures_dir / "labeled_transcript.txt"


# Environment fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# Utility functions
def clean_text(text):
    """Clean text by removing speaker labels, punctuation, extra spaces, and line breaks."""
    import re

    # Remove speaker labels like "**Speaker 1:**"
    text = re.sub(r"\*\*Speaker \d+:\*\*", "", text)
    # Remove any non-alphanumeric characters (except spaces)
    text = re.sub(r"[^\w\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace and line breaks
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_similarity(text1, text2):
    """Calculate similarity ratio between two texts."""
    import difflib

    # Clean both texts
    text1 = clean_text(text1)
    text2 = clean_text(text2)

    # Split into words for word-level comparison
    words1 = set(text1.split())
    words2 = set(text2.split())

    # Calculate word overlap
    common_words = words1.intersection(words2)
    total_words = words1.union(words2)

    word_similarity = len(common_words) / max(len(total_words), 1)

    # Use SequenceMatcher for character-level similarity
    char_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()

    # Combine both metrics (weighted average)
    combined_similarity = (word_similarity * 0.7) + (char_similarity * 0.3)

    return combined_similarity, word_similarity, char_similarity


# Make these utility functions available to tests
@pytest.fixture
def text_utils():
    """Return utility functions for text processing and comparison."""
    return {"clean_text": clean_text, "calculate_similarity": calculate_similarity}
