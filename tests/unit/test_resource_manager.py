"""
Tests for the resource manager and memory management functionality.

This module contains unit tests for the resource manager and memory
management functionality, including memory limits and resource cleanup.
"""

import gc
import unittest
from unittest.mock import MagicMock, patch

import pytest

from pyhearingai.application.resource_manager import (
    ResourceManager,
    get_resource_manager,
    cleanup_resources,
)
from pyhearingai.config import MEMORY_LIMIT, set_memory_limit


class TestResourceManager(unittest.TestCase):
    """Unit tests for the resource manager."""

    def setUp(self):
        """Set up test case."""
        # Store original memory limit to restore later
        self.original_memory_limit = MEMORY_LIMIT

        # Create a fresh instance for each test
        ResourceManager._instance = None
        self.resource_manager = ResourceManager.get_instance()

    def tearDown(self):
        """Tear down test case."""
        # Restore original memory limit
        set_memory_limit(self.original_memory_limit)

        # Clear instance
        ResourceManager._instance = None

    def test_singleton_pattern(self):
        """Test that ResourceManager follows the singleton pattern."""
        # Get two instances
        manager1 = ResourceManager.get_instance()
        manager2 = ResourceManager.get_instance()

        # Check they are the same instance
        self.assertIs(manager1, manager2)

        # Check get_resource_manager returns the same instance
        manager3 = get_resource_manager()
        self.assertIs(manager1, manager3)

    def test_set_memory_limit(self):
        """Test setting memory limit."""
        # Set memory limit
        new_limit = 2048
        self.resource_manager.memory_limit = new_limit

        # Check it was set correctly
        self.assertEqual(self.resource_manager.memory_limit, new_limit)

    @patch("pyhearingai.application.resource_manager.ResourceManager.get_memory_usage")
    def test_is_memory_usage_high(self, mock_get_memory_usage):
        """Test checking if memory usage is high."""
        # Set memory limit
        self.resource_manager.memory_limit = 1000  # 1000 MB limit

        # Test when memory usage is low
        mock_get_memory_usage.return_value = 500  # 500 MB used
        self.assertFalse(self.resource_manager.is_memory_usage_high())

        # Test when memory usage is high
        mock_get_memory_usage.return_value = 900  # 900 MB used (90% of limit)
        self.assertTrue(self.resource_manager.is_memory_usage_high())

        # Test when memory limit is 0 (unlimited)
        self.resource_manager.memory_limit = 0
        mock_get_memory_usage.return_value = 9999  # High memory usage
        self.assertFalse(self.resource_manager.is_memory_usage_high())

    def test_register_cleanup_callback(self):
        """Test registering cleanup callbacks."""
        # Create mock callback
        mock_callback = MagicMock()

        # Register callback
        self.resource_manager.register_cleanup_callback(mock_callback)

        # Check it was registered
        self.assertIn(mock_callback, self.resource_manager._cleanup_callbacks)

        # Register same callback again
        self.resource_manager.register_cleanup_callback(mock_callback)

        # Check it was not registered twice
        self.assertEqual(self.resource_manager._cleanup_callbacks.count(mock_callback), 1)

    def test_cleanup_resources(self):
        """Test cleaning up resources."""
        # Create mock callbacks
        mock_callback1 = MagicMock()
        mock_callback2 = MagicMock()

        # Register callbacks
        self.resource_manager.register_cleanup_callback(mock_callback1)
        self.resource_manager.register_cleanup_callback(mock_callback2)

        # Call cleanup_resources
        with patch("gc.collect"):
            self.resource_manager.cleanup_resources()

        # Check callbacks were called
        mock_callback1.assert_called_once()
        mock_callback2.assert_called_once()

    def test_cleanup_resources_with_exception(self):
        """Test cleaning up resources when a callback raises an exception."""
        # Create mock callbacks
        mock_callback1 = MagicMock()
        mock_callback2 = MagicMock(side_effect=RuntimeError("Test error"))
        mock_callback3 = MagicMock()

        # Register callbacks
        self.resource_manager.register_cleanup_callback(mock_callback1)
        self.resource_manager.register_cleanup_callback(mock_callback2)
        self.resource_manager.register_cleanup_callback(mock_callback3)

        # Call cleanup_resources
        with patch("gc.collect"):
            self.resource_manager.cleanup_resources()

        # Check all callbacks were called
        mock_callback1.assert_called_once()
        mock_callback2.assert_called_once()
        mock_callback3.assert_called_once()

    def test_track_resource(self):
        """Test tracking resources."""
        # Create mock resource
        resource = MagicMock()

        # Track resource
        self.resource_manager.track_resource(resource)

        # Check it was tracked
        self.assertIn(id(resource), self.resource_manager._tracked_resources)

    def test_global_cleanup_resources(self):
        """Test the global cleanup_resources function."""
        # Create a mock for the instance's cleanup_resources method
        with patch.object(ResourceManager, "cleanup_resources", return_value=42.0) as mock_cleanup:
            # Call the global function
            result = cleanup_resources()

            # Check the instance method was called
            mock_cleanup.assert_called_once()

            # Check the result was returned
            self.assertEqual(result, 42.0)

    def test_set_memory_limit_global(self):
        """Test the global set_memory_limit function."""
        # Create a fresh instance before testing
        ResourceManager._instance = None

        # Set memory limit using global function
        set_memory_limit(1024)

        # Check it was set in the global variable
        from pyhearingai.config import MEMORY_LIMIT

        self.assertEqual(MEMORY_LIMIT, 1024)

        # Check it was set in a new resource manager instance
        manager = ResourceManager.get_instance()
        self.assertEqual(manager.memory_limit, 1024)
