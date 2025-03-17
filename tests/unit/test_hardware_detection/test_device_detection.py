"""
Unit tests for hardware acceleration and device detection.

This module tests the hardware detection capabilities of the PyannoteDiarizer class,
particularly the detection and utilization of MPS (Apple Silicon), CUDA, and CPU devices.
"""

import os
import platform
import subprocess
from unittest.mock import MagicMock, patch

import pytest
import torch

from pyhearingai.infrastructure.diarizers.pyannote import PyannoteDiarizer


class TestHardwareDetection:
    """Tests for hardware detection in the PyannoteDiarizer class."""

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_device_detection_mps(self, mock_cuda_available, mock_mps_available):
        """Test detection of MPS (Apple Silicon) device."""
        # Mock MPS as available, CUDA as unavailable
        mock_mps_available.return_value = True
        mock_cuda_available.return_value = False

        # Create diarizer
        diarizer = PyannoteDiarizer()

        # Verify MPS was detected
        assert diarizer._device == "mps"
        mock_mps_available.assert_called_once()
        mock_cuda_available.assert_not_called()  # Should not check CUDA if MPS is available

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_device_detection_cuda(self, mock_cuda_available, mock_mps_available):
        """Test detection of CUDA GPU device."""
        # Mock MPS as unavailable, CUDA as available
        mock_mps_available.return_value = False
        mock_cuda_available.return_value = True

        # Create diarizer
        diarizer = PyannoteDiarizer()

        # Verify CUDA was detected
        assert diarizer._device == "cuda"
        mock_mps_available.assert_called_once()
        mock_cuda_available.assert_called_once()

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_device_detection_cpu(self, mock_cuda_available, mock_mps_available):
        """Test fallback to CPU when no acceleration is available."""
        # Mock both MPS and CUDA as unavailable
        mock_mps_available.return_value = False
        mock_cuda_available.return_value = False

        # Create diarizer
        diarizer = PyannoteDiarizer()

        # Verify CPU was selected as fallback
        assert diarizer._device == "cpu"
        mock_mps_available.assert_called_once()
        mock_cuda_available.assert_called_once()

    @patch("platform.system")
    @patch("platform.machine")
    @patch("subprocess.run")
    @patch("torch.backends.mps.is_available")
    def test_m3_max_detection(self, mock_mps_available, mock_subprocess, mock_machine, mock_system):
        """Test detection of M3 Max processor for optimization."""
        # Mock environment for Apple Silicon
        mock_mps_available.return_value = True
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"

        # Mock subprocess result for system_profiler
        mock_process = MagicMock()
        mock_process.stdout = "Hardware: MacBook Pro (M3 Max, 2023)"
        mock_subprocess.return_value = mock_process

        # Create diarizer and patch internal methods to avoid actual initialization
        with patch.object(PyannoteDiarizer, "_initialize_shared_pipeline"):
            diarizer = PyannoteDiarizer()

            # Call diarize with minimal mocks to trigger processor detection
            with patch.object(diarizer, "pipeline", MagicMock()):
                with patch.object(diarizer, "_run_diarization_with_timeout", MagicMock()):
                    # We need minimal mocks to avoid actual diarization
                    mock_file = MagicMock()
                    mock_file.exists.return_value = True

                    # Call diarize to trigger batch size detection
                    diarizer.diarize(mock_file)

                    # Verify system_profiler was called
                    mock_subprocess.assert_called_with(
                        ["system_profiler", "SPHardwareDataType"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    @patch("torch.set_num_threads")
    def test_cpu_optimization(self, mock_set_threads, mock_cuda_available, mock_mps_available):
        """Test CPU optimization when no GPU is available."""
        # Mock both MPS and CUDA as unavailable
        mock_mps_available.return_value = False
        mock_cuda_available.return_value = False

        # Create diarizer
        diarizer = PyannoteDiarizer()

        # Call diarize with minimal mocks
        with patch.object(diarizer, "_initialize_shared_pipeline"):
            with patch.object(diarizer, "pipeline", MagicMock()):
                with patch.object(diarizer, "_run_diarization_with_timeout", MagicMock()):
                    # We need minimal mocks to avoid actual diarization
                    mock_file = MagicMock()
                    mock_file.exists.return_value = True

                    # Call diarize to trigger CPU optimization
                    diarizer.diarize(mock_file)

                    # Verify CPU threads were optimized
                    mock_set_threads.assert_called_once()

    @patch("torch.backends.mps.is_available")
    @patch("multiprocessing.cpu_count")
    def test_optimal_worker_count(self, mock_cpu_count, mock_mps_available):
        """Test calculation of optimal worker count based on available CPUs."""
        # Mock CPU count
        mock_cpu_count.return_value = 12
        mock_mps_available.return_value = True

        # Create diarizer
        with patch.object(PyannoteDiarizer, "_initialize_shared_pipeline"):
            diarizer = PyannoteDiarizer()

            # For a 12-core system, we expect optimal_workers to be 6 (half of available cores)
            # This is tested by checking the max_workers parameter of the ThreadPoolExecutor
            assert diarizer._executor._max_workers == 6

            # Verify CPU count was checked
            mock_cpu_count.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
