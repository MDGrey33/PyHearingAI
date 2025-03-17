"""
Resource management for PyHearingAI.

This module implements resource management functionality to control memory usage
and provide cleanup mechanisms for resources used during audio processing.
"""

import gc
import logging
import os
import threading
import weakref
from typing import Any, Callable, Dict, List, Optional, Set

from pyhearingai.config import MEMORY_LIMIT
from pyhearingai.workers.supervisor import ResourceSupervisor

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manager for controlling memory usage and resource cleanup.

    This class provides centralized resource management for PyHearingAI,
    including memory usage monitoring, resource cleanup, and throttling
    of resource-intensive operations.
    """

    _instance = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "ResourceManager":
        """
        Get the singleton instance of the ResourceManager.

        Returns:
            The singleton ResourceManager instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = ResourceManager()
            return cls._instance

    def __init__(self):
        """Initialize the resource manager."""
        # Always use the most up-to-date value from the config
        from pyhearingai.config import MEMORY_LIMIT

        self._memory_limit = MEMORY_LIMIT  # in MB
        self._cleanup_callbacks: List[Callable] = []
        self._tracked_resources: Set[int] = set()

        # Create a resource supervisor for monitoring
        self._supervisor = ResourceSupervisor(
            name="pyhearingai-resource-supervisor",
            memory_high_threshold=85.0,  # Trigger at 85% of limit
            enable_monitoring=self._memory_limit > 0,
        )

        # Register callback for high resource usage
        self._supervisor.register_high_resource_callback(self.cleanup_resources)

        logger.debug(f"Initialized ResourceManager with memory limit: {self._memory_limit} MB")

    @property
    def memory_limit(self) -> int:
        """
        Get the current memory limit in MB.

        Returns:
            int: Memory limit in MB (0 = no limit)
        """
        return self._memory_limit

    @memory_limit.setter
    def memory_limit(self, limit_mb: int):
        """
        Set the memory limit in MB.

        Args:
            limit_mb: Memory limit in MB (0 = no limit)
        """
        self._memory_limit = limit_mb

        # Update supervisor monitoring state
        if limit_mb > 0 and not self._supervisor._running:
            self._supervisor.start()
        elif limit_mb == 0 and self._supervisor._running:
            self._supervisor.stop()

        logger.info(f"Set memory limit to {limit_mb} MB")

    def register_cleanup_callback(self, callback: Callable):
        """
        Register a callback function to be called during resource cleanup.

        Args:
            callback: Function to call during cleanup
        """
        if callback not in self._cleanup_callbacks:
            self._cleanup_callbacks.append(callback)
            # Use a safer way to get the function name
            callback_name = getattr(callback, "__name__", str(callback))
            logger.debug(f"Registered cleanup callback: {callback_name}")

    def track_resource(self, resource: Any):
        """
        Track a resource for potential cleanup.

        Args:
            resource: Resource to track
        """
        resource_id = id(resource)
        self._tracked_resources.add(resource_id)

        # Create a finalizer to remove from tracked set when garbage collected
        weakref.finalize(resource, self._remove_tracked_resource, resource_id)

        # Use a safer way to get the class name
        class_name = getattr(resource, "__class__", type(resource)).__name__
        logger.debug(f"Tracking resource: {class_name} (id: {resource_id})")

    def _remove_tracked_resource(self, resource_id: int):
        """
        Remove a resource from the tracked set.

        Args:
            resource_id: ID of the resource to remove
        """
        if resource_id in self._tracked_resources:
            self._tracked_resources.remove(resource_id)
            logger.debug(f"Resource removed from tracking (id: {resource_id})")

    def get_memory_usage(self) -> float:
        """
        Get the current memory usage in MB.

        Returns:
            float: Current memory usage in MB
        """
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except ImportError:
            logger.warning("psutil not available, cannot accurately measure memory usage")
            return 0.0

    def is_memory_usage_high(self) -> bool:
        """
        Check if memory usage is approaching the limit.

        Returns:
            bool: True if memory usage is high, False otherwise
        """
        if self._memory_limit <= 0:
            return False

        memory_usage = self.get_memory_usage()
        threshold = self._memory_limit * 0.85  # 85% of limit

        return memory_usage >= threshold

    def cleanup_resources(self):
        """
        Clean up resources to reduce memory usage.

        This method will call all registered cleanup callbacks and
        force garbage collection to free memory.

        Returns:
            float: Amount of memory freed in MB
        """
        if not self._cleanup_callbacks and not self._tracked_resources:
            logger.debug("No resources to clean up")
            return 0.0

        memory_before = self.get_memory_usage()
        logger.info(f"Starting resource cleanup, current memory usage: {memory_before:.2f} MB")

        # Call all cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
                # Use a safer way to get the function name
                callback_name = getattr(callback, "__name__", str(callback))
                logger.debug(f"Called cleanup callback: {callback_name}")
            except Exception as e:
                # Use a safer way to get the function name
                callback_name = getattr(callback, "__name__", str(callback))
                logger.error(f"Error in cleanup callback {callback_name}: {str(e)}")

        # Force garbage collection
        gc.collect()

        memory_after = self.get_memory_usage()
        memory_freed = memory_before - memory_after

        logger.info(f"Resource cleanup completed, freed {memory_freed:.2f} MB")
        return memory_freed

    def start_monitoring(self):
        """Start resource monitoring."""
        if not self._supervisor._running and self._memory_limit > 0:
            self._supervisor.start()
            logger.info("Started resource monitoring")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        if self._supervisor._running:
            self._supervisor.stop()
            logger.info("Stopped resource monitoring")

    def __enter__(self):
        """Start monitoring when used as a context manager."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and clean up when exiting context."""
        self.stop_monitoring()
        self.cleanup_resources()


# Convenience function for accessing the global resource manager
def get_resource_manager() -> ResourceManager:
    """
    Get the global ResourceManager instance.

    Returns:
        ResourceManager: The global resource manager
    """
    return ResourceManager.get_instance()


# Convenience function for cleaning up resources
def cleanup_resources() -> float:
    """
    Clean up resources to reduce memory usage.

    This function calls the global ResourceManager's cleanup_resources method.

    Returns:
        float: Amount of memory freed in MB

    Examples:
        >>> from pyhearingai import cleanup_resources
        >>> freed_mb = cleanup_resources()
        >>> print(f"Freed {freed_mb:.2f} MB of memory")
    """
    return get_resource_manager().cleanup_resources()
