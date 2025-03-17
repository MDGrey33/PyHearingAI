"""
Supervisor for monitoring system resources and controlling worker scaling.

This module provides a supervisor that monitors CPU and memory usage,
and controls worker scaling to prevent resource exhaustion.
"""

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

# Try to import psutil for better resource monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourceSupervisor:
    """
    Supervisor for monitoring system resources and controlling worker scaling.

    This class monitors CPU and memory usage and provides signals to
    worker pools to scale up or down based on resource utilization.
    """

    def __init__(
        self,
        name: str = "resource-supervisor",
        poll_interval: float = 5.0,
        cpu_high_threshold: float = 80.0,  # 80% CPU usage
        cpu_low_threshold: float = 60.0,  # 60% CPU usage
        memory_high_threshold: float = 80.0,  # 80% memory usage
        memory_low_threshold: float = 60.0,  # 60% memory usage
        enable_monitoring: bool = True,
    ):
        """
        Initialize a resource supervisor.

        Args:
            name: Name for the supervisor for logging
            poll_interval: Interval between resource checks in seconds
            cpu_high_threshold: High CPU usage threshold in percent
            cpu_low_threshold: Low CPU usage threshold in percent
            memory_high_threshold: High memory usage threshold in percent
            memory_low_threshold: Low memory usage threshold in percent
            enable_monitoring: Whether to enable monitoring thread
        """
        self.name = name
        self.poll_interval = poll_interval
        self.cpu_high_threshold = cpu_high_threshold
        self.cpu_low_threshold = cpu_low_threshold
        self.memory_high_threshold = memory_high_threshold
        self.memory_low_threshold = memory_low_threshold

        # Resource state
        self.current_cpu_percent = 0.0
        self.current_memory_percent = 0.0
        self.cpu_throttling = False
        self.memory_throttling = False

        # Monitoring thread
        self._monitoring_thread = None
        self._running = False
        self._lock = threading.RLock()

        # Callbacks for resource events
        self._high_resource_callbacks: List[Callable] = []
        self._low_resource_callbacks: List[Callable] = []

        # Start monitoring if enabled
        if enable_monitoring:
            self.start()

        logger.debug(
            f"Initialized {self.name} with CPU thresholds {cpu_low_threshold:.1f}%/{cpu_high_threshold:.1f}% "
            f"and memory thresholds {memory_low_threshold:.1f}%/{memory_high_threshold:.1f}%"
        )

    def start(self):
        """Start resource monitoring."""
        with self._lock:
            if self._running:
                return

            self._running = True

            # Start monitoring thread
            self._monitoring_thread = threading.Thread(
                target=self._monitor_resources, name=f"{self.name}-monitor", daemon=True
            )
            self._monitoring_thread.start()

            logger.info(f"Started {self.name}")

    def stop(self):
        """Stop resource monitoring."""
        with self._lock:
            if not self._running:
                return

            self._running = False

            # Wait for monitoring thread to exit
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=2)
                self._monitoring_thread = None

            logger.info(f"Stopped {self.name}")

    def _monitor_resources(self):
        """Monitor system resources in a separate thread."""
        logger.debug(f"Started resource monitor for {self.name}")

        if not PSUTIL_AVAILABLE:
            logger.warning(
                f"psutil is not available, resource monitoring will use fallback methods. "
                f"Install psutil for better resource monitoring."
            )

        while self._running:
            try:
                self._check_resources()
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                time.sleep(1)  # Shorter interval on error

        logger.debug(f"Resource monitor for {self.name} exiting")

    def _check_resources(self):
        """Check current CPU and memory usage."""
        # Get CPU usage
        cpu_percent = self._get_cpu_percent()
        memory_percent = self._get_memory_percent()

        # Update current state
        self.current_cpu_percent = cpu_percent
        self.current_memory_percent = memory_percent

        # Check for high resource usage
        cpu_high = cpu_percent >= self.cpu_high_threshold
        memory_high = memory_percent >= self.memory_high_threshold

        # Check for low resource usage
        cpu_low = cpu_percent <= self.cpu_low_threshold
        memory_low = memory_percent <= self.memory_low_threshold

        # Check for throttling transitions
        if not self.cpu_throttling and cpu_high:
            self.cpu_throttling = True
            logger.warning(f"CPU usage high: {cpu_percent:.1f}%, throttling enabled")
            self._notify_high_resource()
        elif self.cpu_throttling and cpu_low:
            self.cpu_throttling = False
            logger.info(f"CPU usage normal: {cpu_percent:.1f}%, throttling disabled")
            self._notify_low_resource()

        if not self.memory_throttling and memory_high:
            self.memory_throttling = True
            logger.warning(f"Memory usage high: {memory_percent:.1f}%, throttling enabled")
            self._notify_high_resource()
        elif self.memory_throttling and memory_low:
            self.memory_throttling = False
            logger.info(f"Memory usage normal: {memory_percent:.1f}%, throttling disabled")
            self._notify_low_resource()

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage in percent."""
        if PSUTIL_AVAILABLE:
            return psutil.cpu_percent(interval=None)

        # Fallback method
        try:
            # Try to read from /proc/stat on Linux
            if os.path.exists("/proc/stat"):
                with open("/proc/stat", "r") as f:
                    line = f.readline()
                    if line.startswith("cpu "):
                        values = list(map(float, line.split()[1:]))
                        idle = values[3]
                        total = sum(values)
                        return 100.0 * (1.0 - idle / total)
        except Exception:
            pass

        # Default to a moderate value if we can't determine
        return 50.0

    def _get_memory_percent(self) -> float:
        """Get current memory usage in percent."""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().percent

        # Fallback method
        try:
            # Try to read from /proc/meminfo on Linux
            if os.path.exists("/proc/meminfo"):
                mem_info = {}
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if value.endswith("kB"):
                                value = float(value[:-2]) * 1024
                            mem_info[key] = float(value)

                if "MemTotal" in mem_info and "MemAvailable" in mem_info:
                    return 100.0 * (1.0 - mem_info["MemAvailable"] / mem_info["MemTotal"])
        except Exception:
            pass

        # Default to a moderate value if we can't determine
        return 50.0

    def register_high_resource_callback(self, callback: Callable):
        """
        Register a callback for high resource usage events.

        Args:
            callback: Callable to invoke on high resource usage
        """
        with self._lock:
            if callback not in self._high_resource_callbacks:
                self._high_resource_callbacks.append(callback)

    def register_low_resource_callback(self, callback: Callable):
        """
        Register a callback for low resource usage events.

        Args:
            callback: Callable to invoke on low resource usage
        """
        with self._lock:
            if callback not in self._low_resource_callbacks:
                self._low_resource_callbacks.append(callback)

    def _notify_high_resource(self):
        """Notify callbacks of high resource usage."""
        for callback in self._high_resource_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in high resource callback: {e}")

    def _notify_low_resource(self):
        """Notify callbacks of low resource usage."""
        for callback in self._low_resource_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in low resource callback: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current resource status.

        Returns:
            Dictionary with resource status information
        """
        return {
            "name": self.name,
            "running": self._running,
            "cpu_percent": self.current_cpu_percent,
            "memory_percent": self.current_memory_percent,
            "cpu_throttling": self.cpu_throttling,
            "memory_throttling": self.memory_throttling,
            "cpu_high_threshold": self.cpu_high_threshold,
            "cpu_low_threshold": self.cpu_low_threshold,
            "memory_high_threshold": self.memory_high_threshold,
            "memory_low_threshold": self.memory_low_threshold,
            "psutil_available": PSUTIL_AVAILABLE,
        }

    def should_throttle(self) -> bool:
        """
        Check if workers should be throttled due to resource constraints.

        Returns:
            True if throttling is recommended, False otherwise
        """
        return self.cpu_throttling or self.memory_throttling

    def get_recommended_workers(self, max_workers: int) -> int:
        """
        Get recommended number of workers based on resource usage.

        Args:
            max_workers: Maximum number of workers configured

        Returns:
            Recommended number of workers
        """
        if not self.should_throttle():
            return max_workers

        # Simple throttling: reduce by 50% when resources are constrained
        return max(1, max_workers // 2)

    def __enter__(self):
        """Start the supervisor when used as a context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the supervisor when exiting a context manager."""
        self.stop()


# Global supervisor instance
global_supervisor = ResourceSupervisor(enable_monitoring=True)
