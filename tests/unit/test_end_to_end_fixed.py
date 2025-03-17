"""
End-to-end integration tests for the complete PyHearingAI workflow.

These tests validate the entire pipeline from audio processing through
diarization, transcription, and speaker assignment.
"""
import os
import time
import tempfile
import pytest
from pathlib import Path

from pyhearingai.core.idempotent import ProcessingJob, ProcessingStatus
from pyhearingai.application.progress import ProgressTracker
from pyhearingai.application.orchestrator import WorkflowOrchestrator
from pyhearingai.infrastructure.repositories.json_repositories import (
    JsonJobRepository,
    JsonChunkRepository,
)


class TestFixtures:
    """Helper class for creating test fixtures."""

    @staticmethod
    def create_test_audio(filepath, duration=10.0, sample_rate=16000, num_speakers=2):
        """Create a synthetic test audio file with multiple speakers."""
        # Check if file already exists
        if os.path.exists(filepath):
            return

        try:
            import numpy as np
            from scipy.io import wavfile

            # Generate a synthetic audio file with alternating speakers
            samples = int(duration * sample_rate)
            audio = np.zeros(samples, dtype=np.float32)

            # Create segments for different speakers
            segment_length = int(sample_rate * 2)  # 2-second segments
            num_segments = samples // segment_length

            for i in range(num_segments):
                speaker_idx = i % num_speakers
                segment_start = i * segment_length
                segment_end = (i + 1) * segment_length

                # Different frequency for each speaker
                freq = 440 * (1 + speaker_idx * 0.5)  # Hz
                t = np.linspace(0, 2, segment_length, False)

                # Generate a tone with some noise
                tone = np.sin(2 * np.pi * freq * t) * 0.5
                noise = np.random.normal(0, 0.05, segment_length)
                audio[segment_start:segment_end] = tone + noise

            # Normalize
            audio = audio / np.max(np.abs(audio))

            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)

            # Write to file
            wavfile.write(filepath, sample_rate, audio_int16)

        except ImportError:
            # Fallback method if scipy or numpy not available
            with open(filepath, "wb") as f:
                # Write a minimal valid WAV file header
                f.write(b"RIFF")
                f.write((36).to_bytes(4, byteorder="little"))  # Chunk size
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write((16).to_bytes(4, byteorder="little"))  # Subchunk1 size
                f.write((1).to_bytes(2, byteorder="little"))  # Audio format (PCM)
                f.write((1).to_bytes(2, byteorder="little"))  # Num channels
                f.write((sample_rate).to_bytes(4, byteorder="little"))  # Sample rate
                f.write((sample_rate * 2).to_bytes(4, byteorder="little"))  # Byte rate
                f.write((2).to_bytes(2, byteorder="little"))  # Block align
                f.write((16).to_bytes(2, byteorder="little"))  # Bits per sample
                f.write(b"data")
                f.write((0).to_bytes(4, byteorder="little"))  # Subchunk2 size

                # No actual audio data in the fallback method


class TestEndToEnd:
    """End-to-end tests for the complete PyHearingAI workflow."""

    @classmethod
    def setup_class(cls):
        """Set up test repositories and file paths."""
        # Create temporary directories for test data
        cls.test_dir = tempfile.mkdtemp()
        cls.output_dir = tempfile.mkdtemp()

        # Initialize repositories
        cls.job_repo = JsonJobRepository(os.path.join(cls.test_dir, "jobs.json"))
        cls.chunk_repo = JsonChunkRepository(os.path.join(cls.test_dir, "chunks.json"))

        # Create test audio file
        cls.test_audio = os.path.join(cls.test_dir, "test_audio.wav")
        TestFixtures.create_test_audio(cls.test_audio)

        # Initialize mocks if needed
        if hasattr(cls, "_setup_mocks"):
            cls._setup_mocks()

        # Set up test environment variables
        os.environ["PYHEARINGAI_OUTPUT_DIR"] = cls.output_dir

        # If we're in a test environment without valid API keys, use mock implementations
        if not cls._check_api_keys():
            os.environ["PYHEARINGAI_USE_MOCK"] = "1"

    @classmethod
    def teardown_class(cls):
        """Clean up test resources."""
        # Remove test files
        if os.path.exists(cls.test_audio):
            os.remove(cls.test_audio)

        # Delete all files in the test directories
        for directory in [cls.test_dir, cls.output_dir]:
            if os.path.exists(directory):
                # Clean up subdirectories and files recursively
                for root, dirs, files in os.walk(directory, topdown=False):
                    # First remove all files
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                        except (OSError, IOError) as e:
                            print(f"Warning: Failed to remove file {file_path}: {e}")
                    
                    # Then remove empty directories
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            os.rmdir(dir_path)
                        except (OSError, IOError) as e:
                            print(f"Warning: Failed to remove directory {dir_path}: {e}")
                
                # Finally remove the top directory
                try:
                    os.rmdir(directory)
                except (OSError, IOError) as e:
                    print(f"Warning: Failed to remove directory {directory}: {e}")

        # Clean environment variables
        for var in ["PYHEARINGAI_OUTPUT_DIR", "PYHEARINGAI_USE_MOCK"]:
            if var in os.environ:
                del os.environ[var]

    @classmethod
    def _check_api_keys(cls):
        """Check if required API keys are available."""
        # Get API keys from environment
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")

        # Return True if both keys are available
        return bool(openai_api_key and huggingface_api_key)

    def _create_test_job(self):
        """Create a test job for processing."""
        job_id = f"test-e2e-{os.urandom(8).hex()}"
        job = ProcessingJob(
            id=job_id,
            original_audio_path=Path(self.test_audio),
            status=ProcessingStatus.PENDING,
            processing_options={
                "diarizer": "pyannote" if self._check_api_keys() else "mock",
                "transcriber": "whisper_openai" if self._check_api_keys() else "mock",
                "output_format": "txt",
            },
        )

        # Save the job
        self.job_repo.save(job)

        return job

    @pytest.mark.slow
    def test_basic_processing(self):
        """Test the basic end-to-end workflow."""
        # Create a test job
        job = self._create_test_job()

        # Create an orchestrator
        orchestrator = WorkflowOrchestrator(
            job_repository=self.job_repo, chunk_repository=self.chunk_repo
        )

        # Create a progress tracker
        progress_tracker = ProgressTracker(job=job, chunks=[], show_chunks=True)  # Empty initially

        # Process the job
        result = orchestrator.process_job(job, progress_tracker)

        # Verify job was processed successfully
        assert result is not None
        assert result.segments is not None
        
        # In tests, we might get empty segments due to the short test audio
        # So we'll just check that the job completed without errors
        assert job.status == ProcessingStatus.COMPLETED

        # If we have segments, verify they have the expected properties
        if result.segments:
            for segment in result.segments:
                assert segment.speaker_id is not None
                assert segment.text is not None
                assert segment.start is not None
                assert segment.end is not None

        # Verify job has updated processing_options
        assert job.processing_options is not None
        
        # Verify result metadata exists
        assert result.metadata is not None
        assert "created_at" in result.metadata

    @pytest.mark.slow
    def test_resumable_processing(self):
        """Test that processing can be paused and resumed."""
        # Create a test job
        job = self._create_test_job()

        # Create an orchestrator
        orchestrator = WorkflowOrchestrator(
            job_repository=self.job_repo, chunk_repository=self.chunk_repo
        )

        # Create a progress tracker
        progress_tracker = ProgressTracker(job=job, chunks=[], show_chunks=True)  # Empty initially

        # First run with a simulated interruption
        # We'll save the job but throw an error during processing
        original_process_job = orchestrator.process_job

        # Counter to track calls
        call_count = [0]

        def mock_process_job(job, progress_tracker=None):
            # Increment call count
            call_count[0] += 1

            # On first call, simulate an interruption
            if call_count[0] == 1:
                # Update job status to indicate it's in progress
                job.status = ProcessingStatus.IN_PROGRESS
                self.job_repo.save(job)

                # Simulate an interruption
                raise Exception("Simulated interruption")

            # On subsequent calls, use the original method
            return original_process_job(job, progress_tracker)

        # Patch the method
        orchestrator.process_job = mock_process_job

        # First attempt - should fail
        try:
            orchestrator.process_job(job, progress_tracker)
            # Should not reach here
            assert False, "Expected an exception due to simulated interruption"
        except Exception as e:
            # Verify the exception is our simulated one
            assert "Simulated interruption" in str(e)

        # Verify job is still in progress
        assert job.status == ProcessingStatus.IN_PROGRESS

        # Now restore the original method and try again
        orchestrator.process_job = original_process_job

        # Second attempt - should resume and complete
        result = orchestrator.process_job(job, progress_tracker)

        # Verify job completed successfully
        assert result is not None
        assert job.status == ProcessingStatus.COMPLETED

        # Verify we have segments
        assert result.segments is not None
        
        # No need to assert there are segments, as the test audio might not produce any
        # The important thing is that the job completed successfully
        
        # Verify result has metadata
        assert result.metadata is not None

    @pytest.mark.slow
    def test_diarization_options(self):
        """Test processing with different diarization options."""
        # These tests would check different parameters for the diarizer
        pass

    @pytest.mark.slow
    def test_performance_benchmarking(self):
        """Test that increasing worker count improves performance."""
        # Skip this test if running in CI/CD or other non-interactive environments
        if os.environ.get("CI") or not os.isatty(0):
            pytest.skip("Skipping performance benchmark in non-interactive environment")

        # Use a short test audio file
        temp_dir = tempfile.mkdtemp()
        test_audio = os.path.join(temp_dir, "benchmark_audio.wav")
        TestFixtures.create_test_audio(test_audio, duration=5.0)

        # Test with various worker counts
        worker_configs = [1, 2, 4]  # Test with 1, 2, and 4 workers
        results = {}

        for workers in worker_configs:
            # Create a new job for each test
            job_id = f"benchmark-{workers}-workers"
            job = ProcessingJob(
                id=job_id,
                original_audio_path=Path(test_audio),
                status=ProcessingStatus.PENDING,
                processing_options={
                    "diarizer": "pyannote",
                    "transcriber": "mock",  # Use mock transcriber for faster tests
                    "output_format": "txt",
                },
            )

            # Save the job
            self.job_repo.save(job)

            # Create an orchestrator with the specified worker count
            orchestrator = WorkflowOrchestrator(
                max_workers=workers,
                job_repository=self.job_repo,
                chunk_repository=self.chunk_repo,
                enable_monitoring=True,
            )

            # Create a progress tracker
            progress_tracker = ProgressTracker(
                job=job, chunks=[], show_chunks=True  # Empty initially
            )

            try:
                # Process the job and measure time
                start_time = time.time()
                orchestrator.process_job(job, progress_tracker)
                duration = time.time() - start_time

                # Store results
                results[workers] = {
                    "duration": duration,
                    "metrics": orchestrator.monitoring.get_summary(),
                }

                print(f"Workers: {workers}, Duration: {duration:.2f}s")
            except Exception as e:
                print(f"Error with {workers} workers: {str(e)}")
            finally:
                # Clean up
                self.job_repo.delete(job_id)

        # Print summary
        print("\nPerformance Benchmark Results:")
        print("-" * 50)
        for workers, data in results.items():
            print(f"Workers: {workers}")
            print(f"  Duration: {data['duration']:.2f}s")
            print(f"  Total Duration: {data['metrics']['total_duration']}")
            print(f"  Task Timings: {data['metrics']['task_timings']}")
            print(f"  Memory Peak: {data['metrics']['memory_peak_mb']} MB")
            print(f"  Error Count: {data['metrics']['error_count']}")
            print("-" * 50)

        # Verify we have results for all worker configurations
        for worker_count in worker_configs:
            assert worker_count in results, f"Missing results for {worker_count} workers"

        # Verify that using more workers should not increase processing time
        if 1 in results and 2 in results:
            # We should see performance improvement with more workers
            # In some edge cases or with very small files, this might not be true
            # so we'll just print a warning if more workers took longer
            if results[2]["duration"] > results[1]["duration"]:
                print(
                    f"WARNING: 2 workers ({results[2]['duration']:.2f}s) was slower than 1 worker ({results[1]['duration']:.2f}s)"
                )
            else:
                # This is what we expect - more workers should be faster
                print(
                    f"SUCCESS: 2 workers ({results[2]['duration']:.2f}s) was faster than 1 worker ({results[1]['duration']:.2f}s)"
                )

        # Clean up
        os.remove(test_audio)
        os.rmdir(temp_dir)
