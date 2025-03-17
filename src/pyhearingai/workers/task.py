"""
Task definitions for the worker system.

This module defines the Task class and related enums for managing
parallel processing tasks in the worker system.
"""

import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Result type for tasks


class TaskStatus(enum.Enum):
    """Status of a task in the worker system."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(enum.IntEnum):
    """Priority levels for tasks in the worker system."""

    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


@dataclass
class Task(Generic[T]):
    """
    A task to be executed by the worker system.

    Attributes:
        id: Unique identifier for the task
        func: Function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        status: Current status of the task
        priority: Priority level for scheduling
        created_at: When the task was created
        started_at: When the task started execution (if started)
        completed_at: When the task completed execution (if completed)
        result: Result of the task execution (if completed)
        error: Error information (if failed)
        metadata: Additional task metadata
    """

    # Task function and arguments
    func: Callable[..., T]
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # Task identification and status
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL

    # Timing information
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Result or error information
    result: Optional[T] = None
    error: Optional[Exception] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def execute(self) -> T:
        """
        Execute the task function and update status.

        Returns:
            The result of the task execution

        Raises:
            Exception: Any exception raised during task execution
        """
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Cannot execute task with status {self.status}")

        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

        try:
            logger.debug(f"Executing task {self.id}")
            self.result = self.func(*self.args, **self.kwargs)
            self.complete(self.result)
            logger.debug(f"Task {self.id} completed successfully")
            return self.result
        except Exception as e:
            self.fail(e)
            logger.error(f"Task {self.id} failed: {str(e)}")
            raise

    def complete(self, result: T) -> None:
        """
        Mark the task as completed with the given result.

        Args:
            result: The result of the task execution
        """
        self.result = result
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()

    def fail(self, error: Exception) -> None:
        """
        Mark the task as failed with the given error.

        Args:
            error: The error that caused the task to fail
        """
        self.error = error
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()

    def cancel(self) -> bool:
        """
        Cancel the task if it's pending.

        Returns:
            True if the task was cancelled, False if it couldn't be cancelled
        """
        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.CANCELLED
            self.completed_at = datetime.now()
            logger.debug(f"Task {self.id} cancelled")
            return True
        return False

    @property
    def duration(self) -> Optional[float]:
        """
        Get the duration of the task execution in seconds.

        Returns:
            Duration in seconds, or None if the task hasn't completed
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def age(self) -> float:
        """
        Get the age of the task in seconds since creation.

        Returns:
            Age in seconds
        """
        return (datetime.now() - self.created_at).total_seconds()

    def __lt__(self, other: "Task") -> bool:
        """Compare tasks based on priority for use in priority queues."""
        if not isinstance(other, Task):
            return NotImplemented
        return self.priority > other.priority  # Higher priority values come first
