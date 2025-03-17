"""
Tests for worker components.

This module contains tests for the worker components, including the Monitoring class
and ThreadPoolExecutor pattern used in the orchestrator.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from pyhearingai.application.orchestrator import Monitoring


class TestMonitoring:
    """Tests for the Monitoring class."""

    def test_monitoring_initialization(self):
        """Test that monitoring is initialized correctly."""
        # With monitoring enabled
        monitoring = Monitoring(enabled=True)
        assert monitoring.enabled is True
        assert monitoring.start_time > 0
        assert monitoring.timings == {}
        assert monitoring.memory_usage == []
        assert monitoring.errors == []

        # With monitoring disabled
        monitoring_disabled = Monitoring(enabled=False)
        assert monitoring_disabled.enabled is False

    def test_task_timing(self):
        """Test task timing functionality."""
        monitoring = Monitoring(enabled=True)

        # Start a task
        monitoring.start_task("test_task")
        assert "test_task" in monitoring.timings
        assert monitoring.timings["test_task"]["start"] > 0
        assert monitoring.timings["test_task"]["end"] is None
        assert monitoring.timings["test_task"]["duration"] is None

        # End the task
        time.sleep(0.1)  # Small delay to ensure measurable duration
        duration = monitoring.end_task("test_task")

        assert duration > 0
        assert monitoring.timings["test_task"]["end"] > monitoring.timings["test_task"]["start"]
        assert monitoring.timings["test_task"]["duration"] > 0

    def test_error_logging(self):
        """Test error logging functionality."""
        monitoring = Monitoring(enabled=True)

        # Create an error to log
        error = ValueError("Test error")
        monitoring.log_error("test_task", error)

        assert len(monitoring.errors) == 1
        error_info = monitoring.errors[0]
        assert error_info["task"] == "test_task"
        assert error_info["error"] == "Test error"
        assert error_info["type"] == "ValueError"
        assert "timestamp" in error_info
        assert "traceback" in error_info

    @patch("pyhearingai.application.orchestrator.Monitoring.log_memory_usage")
    def test_summary_generation(self, mock_log_memory):
        """Test generation of monitoring summary."""
        monitoring = Monitoring(enabled=True)

        # Create some test data
        monitoring.start_task("task1")
        time.sleep(0.1)
        monitoring.end_task("task1")

        monitoring.start_task("task2")
        time.sleep(0.2)
        monitoring.end_task("task2")

        monitoring.log_error("task3", ValueError("Another test error"))

        # Mock memory usage
        monitoring.memory_usage = [
            {"timestamp": time.time(), "memory_mb": 100.5},
            {"timestamp": time.time(), "memory_mb": 102.3},
        ]

        # Get summary
        summary = monitoring.get_summary()

        assert "total_duration" in summary
        assert "tasks" in summary
        assert "task1" in summary["tasks"]
        assert "task2" in summary["tasks"]
        assert "memory" in summary
        assert summary["memory"]["max"] == 102.3
        assert summary["errors"] == 1

    def test_disabled_monitoring(self):
        """Test that disabled monitoring doesn't record anything."""
        monitoring = Monitoring(enabled=False)

        # Try to use monitoring features
        monitoring.start_task("test_task")
        duration = monitoring.end_task("test_task")
        monitoring.log_error("test_task", ValueError("Test error"))
        monitoring.log_memory_usage()

        # Verify nothing was recorded
        assert duration == 0.0
        assert not monitoring.timings
        assert not monitoring.errors
        assert not monitoring.memory_usage

        # Summary should be empty too
        summary = monitoring.get_summary()
        assert summary == {}


# Test ThreadPoolExecutor pattern used in the orchestrator
class TestThreadPoolExecutor:
    """Tests for the thread pool executor pattern."""

    def test_executor_task_processing(self):
        """Test that executor processes tasks correctly."""

        # Function to test
        def square(x):
            return x * x

        # Create executor with max_workers=2
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit 5 tasks
            futures = [executor.submit(square, i) for i in range(1, 6)]

            # Get results
            results = [future.result() for future in futures]

            # Verify results
            assert results == [1, 4, 9, 16, 25]

    def test_executor_exception_handling(self):
        """Test that executor handles exceptions correctly."""

        # Function that raises an exception
        def raise_error(x):
            if x == 3:
                raise ValueError("Error in task")
            return x

        # Create executor
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit tasks
            futures = [executor.submit(raise_error, i) for i in range(1, 5)]

            # Process results and catch exceptions
            results = []
            exceptions = []

            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    exceptions.append(str(e))

            # Verify results and exceptions
            assert results == [1, 2, 4]
            assert len(exceptions) == 1
            assert "Error in task" in exceptions[0]
