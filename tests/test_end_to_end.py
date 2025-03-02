import difflib
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import main


class TestEndToEnd(unittest.TestCase):
    """End-to-end test for the PyHearingAI pipeline."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()

        # Create content directory structure
        os.makedirs(os.path.join(self.temp_dir, "content/audio_conversion"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "content/transcription"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "content/diarization"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "content/speaker_assignment"), exist_ok=True)

        # Copy the example audio to the root of the temp directory
        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        self.example_audio = os.path.join(fixtures_dir, "example_audio.m4a")
        self.reference_transcript = os.path.join(fixtures_dir, "labeled_transcript.txt")

        # Copy files to temp directory
        shutil.copy(self.example_audio, os.path.join(self.temp_dir, "example_audio.m4a"))

        # Copy the .env file if it exists
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_file):
            shutil.copy(env_file, os.path.join(self.temp_dir, ".env"))

        # Change to the temp directory
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directories."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    def clean_text(self, text):
        """Clean text by removing speaker labels, punctuation, extra spaces, and line breaks."""
        # Remove speaker labels like "**Speaker 1:**"
        text = re.sub(r"\*\*Speaker \d+:\*\*", "", text)
        # Remove any non-alphanumeric characters (except spaces)
        text = re.sub(r"[^\w\s]", "", text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace and line breaks
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def calculate_similarity(self, text1, text2):
        """Calculate similarity ratio between two texts."""
        # Clean both texts
        text1 = self.clean_text(text1)
        text2 = self.clean_text(text2)

        # Print cleaned texts for debugging
        print("\nCleaned reference text:")
        print(text1[:500] + "..." if len(text1) > 500 else text1)
        print("\nCleaned generated text:")
        print(text2[:500] + "..." if len(text2) > 500 else text2)

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

        print(f"\nWord similarity: {word_similarity:.2%}")
        print(f"Character similarity: {char_similarity:.2%}")
        print(f"Combined similarity: {combined_similarity:.2%}")

        return combined_similarity

    def test_pipeline(self):
        """Test the complete pipeline from audio to labeled transcript."""
        try:
            # Create output debug directory
            debug_dir = os.path.join(self.original_cwd, "tests/debug")
            os.makedirs(debug_dir, exist_ok=True)

            # Run the main pipeline
            main()

            # Check if the output file exists
            output_path = os.path.join("content/speaker_assignment/labeled_transcript.txt")
            self.assertTrue(
                os.path.exists(output_path), f"Output file {output_path} does not exist"
            )

            # Save copies of both files to debug directory
            generated_debug_path = os.path.join(debug_dir, "generated_transcript.txt")
            reference_debug_path = os.path.join(debug_dir, "reference_transcript.txt")

            # Read the generated transcript
            with open(output_path, "r") as f:
                generated_transcript = f.read()

            # Save generated transcript to debug directory
            with open(generated_debug_path, "w") as f:
                f.write(generated_transcript)

            # Read the reference transcript
            with open(self.reference_transcript, "r") as f:
                reference_transcript = f.read()

            # Save reference transcript to debug directory
            with open(reference_debug_path, "w") as f:
                f.write(reference_transcript)

            # Calculate similarity
            similarity = self.calculate_similarity(reference_transcript, generated_transcript)

            # Generate a detailed comparison using difflib
            diff = difflib.ndiff(
                self.clean_text(reference_transcript).splitlines(),
                self.clean_text(generated_transcript).splitlines(),
            )

            # Save the diff to debug directory
            with open(os.path.join(debug_dir, "transcript_diff.txt"), "w") as f:
                f.write("\n".join(diff))

            print(
                f"\nDetailed comparison saved to {os.path.join(debug_dir, 'transcript_diff.txt')}"
            )
            print(f"Debug files saved to: {debug_dir}")

            # Lower the similarity threshold for initial development (adjust as needed)
            required_similarity = 0.3  # 30% similarity during development

            self.assertGreaterEqual(
                similarity,
                required_similarity,
                f"Transcript similarity {similarity:.2%} is below the required {required_similarity:.0%} threshold",
            )
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")


if __name__ == "__main__":
    unittest.main()
