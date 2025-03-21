"""
Worker system for parallel processing of audio chunks.

This package provides components for managing parallel processing
of audio chunks, enabling efficient utilization of system resources
while maintaining control over resource usage.
"""

from pyhearingai.workers.pool import WorkerPool
from pyhearingai.workers.supervisor import ResourceSupervisor, global_supervisor
from pyhearingai.workers.task import Task, TaskPriority, TaskStatus
from pyhearingai.workers.throttling import MultiRateLimiter, RateLimiter, global_rate_limiters

__all__ = [
    "Task",
    "TaskPriority",
    "TaskStatus",
    "WorkerPool",
    "RateLimiter",
    "MultiRateLimiter",
    "global_rate_limiters",
    "ResourceSupervisor",
    "global_supervisor",
]
