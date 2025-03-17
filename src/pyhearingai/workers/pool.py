"""
Worker pool for parallel processing of tasks.

This module provides a worker pool implementation that uses Python's
multiprocessing to execute tasks in parallel with configurable concurrency.
"""

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

from pyhearingai.workers.task import Task, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)
T = TypeVar("T")  # Result type for tasks


class WorkerPool(Generic[T]):
    """
    Pool of worker processes for parallel task execution.

    This class manages a pool of worker processes for executing tasks
    in parallel, with configurable concurrency and resource limits.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        name: str = "worker-pool",
        task_timeout: Optional[float] = None,
        use_processes: bool = True,
    ):
        """
        Initialize a worker pool.

        Args:
            max_workers: Maximum number of concurrent workers. Defaults to CPU count.
            name: Name for the worker pool for logging
            task_timeout: Default timeout for tasks in seconds
            use_processes: Whether to use processes (True) or threads (False)
        """
        self.name = name
        self.max_workers = max_workers or min(32, (os.cpu_count() or 4))
        self.task_timeout = task_timeout
        self.use_processes = use_processes

        # Task management
        self._task_queue = queue.PriorityQueue()
        self._active_tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, Task] = {}
        self._running = False
        self._executor = None
        self._monitor_thread = None
        self._lock = threading.RLock()

        logger.debug(f"Initialized {self.name} with max_workers={self.max_workers}")

    def submit(
        self, func: Callable[..., T], *args, priority: TaskPriority = TaskPriority.NORMAL, **kwargs
    ) -> Task[T]:
        """
        Submit a task to the worker pool.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            priority: Priority level for scheduling
            **kwargs: Keyword arguments for the function

        Returns:
            Task object representing the submitted task
        """
        task = Task(func=func, args=args, kwargs=kwargs, priority=priority)

        with self._lock:
            self._task_queue.put(task)
            logger.debug(f"Submitted task {task.id} to {self.name} with priority {priority.name}")

        # Start pool if not already running
        if not self._running:
            self.start()

        return task

    def submit_task(self, task: Task[T]) -> Task[T]:
        """
        Submit an existing task to the worker pool.

        Args:
            task: Task to submit

        Returns:
            The submitted task
        """
        if task.status != TaskStatus.PENDING:
            raise ValueError(f"Cannot submit task with status {task.status}")

        with self._lock:
            self._task_queue.put(task)
            logger.debug(f"Submitted existing task {task.id} to {self.name}")

        # Start pool if not already running
        if not self._running:
            self.start()

        return task

    def start(self):
        """Start the worker pool if not already running."""
        with self._lock:
            if self._running:
                return

            self._running = True

            # Create executor based on configuration
            if self.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                from concurrent.futures import ThreadPoolExecutor

                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

            # Start monitor thread
            self._monitor_thread = threading.Thread(
                target=self._monitor_tasks, name=f"{self.name}-monitor", daemon=True
            )
            self._monitor_thread.start()

            logger.info(f"Started {self.name} with {self.max_workers} workers")

    def stop(self, wait: bool = True, timeout: Optional[float] = None):
        """
        Stop the worker pool.

        Args:
            wait: Whether to wait for pending tasks to complete
            timeout: Maximum time to wait for task completion
        """
        with self._lock:
            if not self._running:
                return

            self._running = False

            # Cancel pending tasks
            pending_tasks = []
            while not self._task_queue.empty():
                try:
                    task = self._task_queue.get_nowait()
                    task.cancel()
                    pending_tasks.append(task)
                except queue.Empty:
                    break

            logger.debug(f"Cancelled {len(pending_tasks)} pending tasks")

            # Shutdown executor
            if self._executor:
                self._executor.shutdown(wait=wait, cancel_futures=not wait)
                self._executor = None

            # Wait for monitor thread to exit
            if self._monitor_thread and wait:
                self._monitor_thread.join(timeout=timeout)
                self._monitor_thread = None

            logger.info(f"Stopped {self.name}")

    def _monitor_tasks(self):
        """Monitor task execution in a separate thread."""
        logger.debug(f"Started task monitor for {self.name}")

        while self._running:
            try:
                self._process_task_queue()
                time.sleep(0.01)  # Small sleep to avoid busy waiting
            except Exception as e:
                logger.error(f"Error in task monitor: {e}")

        logger.debug(f"Task monitor for {self.name} exiting")

    def _process_task_queue(self):
        """Process tasks from the queue as workers become available."""
        with self._lock:
            # Check if we can submit more tasks
            active_count = len(self._active_tasks)
            if active_count >= self.max_workers:
                return

            # Submit tasks up to max_workers
            available_slots = self.max_workers - active_count
            for _ in range(available_slots):
                if self._task_queue.empty():
                    break

                try:
                    task = self._task_queue.get_nowait()

                    # Skip cancelled tasks
                    if task.status == TaskStatus.CANCELLED:
                        continue

                    # Submit task to executor
                    future = self._executor.submit(task.execute)

                    # Record active task
                    task.future = future
                    self._active_tasks[task.id] = task

                    # Set callback for task completion
                    future.add_done_callback(
                        lambda f, task_id=task.id: self._task_completed(task_id, f)
                    )

                    logger.debug(f"Submitted task {task.id} to executor")
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error submitting task: {e}")

    def _task_completed(self, task_id: str, future):
        """Handle completion of a task."""
        with self._lock:
            if task_id not in self._active_tasks:
                logger.warning(f"Task {task_id} not found in active tasks")
                return

            task = self._active_tasks.pop(task_id)

            # Update task status based on future result
            try:
                # Get the result from the future to propagate any exceptions
                result = future.result()

                # If task.execute didn't update the result, update it here
                if task.status == TaskStatus.PENDING or task.status == TaskStatus.RUNNING:
                    task.complete(result)
                    logger.debug(f"Updated task {task_id} result and status to COMPLETED")
            except Exception as e:
                # If task.execute didn't update the status, mark it as failed
                if task.status == TaskStatus.PENDING or task.status == TaskStatus.RUNNING:
                    task.fail(e)
                    logger.error(f"Task {task_id} failed with error: {e}")

            # Move to completed tasks
            self._completed_tasks[task_id] = task

            logger.debug(f"Task {task_id} completed with status {task.status.name}")

    def get_task(self, task_id: str) -> Optional[Task[T]]:
        """
        Get a task by ID.

        Args:
            task_id: ID of the task

        Returns:
            Task object, or None if not found
        """
        with self._lock:
            # Check active tasks first
            if task_id in self._active_tasks:
                return self._active_tasks[task_id]

            # Then check completed tasks
            if task_id in self._completed_tasks:
                return self._completed_tasks[task_id]

        return None

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task by ID.

        Args:
            task_id: ID of the task

        Returns:
            True if the task was cancelled, False otherwise
        """
        with self._lock:
            # Check active tasks
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]

                # Can only cancel pending tasks
                if task.status == TaskStatus.PENDING:
                    if hasattr(task, "future") and task.future:
                        task.future.cancel()

                    task.cancel()
                    return True

            # Check task queue
            for task in list(self._task_queue.queue):
                if task.id == task_id:
                    task.cancel()
                    return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the worker pool.

        Returns:
            Dictionary with worker pool status information
        """
        with self._lock:
            pending_count = self._task_queue.qsize()
            active_count = len(self._active_tasks)
            completed_count = len(self._completed_tasks)

            # Count tasks by status
            status_counts = {status.name: 0 for status in TaskStatus}

            # Count active tasks
            for task in self._active_tasks.values():
                status_counts[task.status.name] += 1

            # Count completed tasks
            for task in self._completed_tasks.values():
                status_counts[task.status.name] += 1

            # Try to count pending tasks
            try:
                for task in list(self._task_queue.queue):
                    status_counts[task.status.name] += 1
            except Exception:
                # Queue might not support iteration
                pass

            return {
                "name": self.name,
                "running": self._running,
                "max_workers": self.max_workers,
                "pending_tasks": pending_count,
                "active_tasks": active_count,
                "completed_tasks": completed_count,
                "total_tasks": pending_count + active_count + completed_count,
                "status_counts": status_counts,
            }

    def __enter__(self):
        """Start the worker pool when used as a context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the worker pool when exiting a context manager."""
        self.stop(wait=True)

    def wait_for(
        self, task_ids: Union[str, List[str]], timeout: Optional[float] = None
    ) -> Dict[str, Task[T]]:
        """
        Wait for specific tasks to complete.

        Args:
            task_ids: Task ID or list of task IDs to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Dictionary mapping task IDs to completed tasks
        """
        if isinstance(task_ids, str):
            task_ids = [task_ids]

        task_ids = set(task_ids)
        completed_tasks = {}
        start_time = time.time()

        while task_ids and (timeout is None or time.time() - start_time < timeout):
            with self._lock:
                # Check for completed tasks
                for task_id in list(task_ids):
                    if task_id in self._completed_tasks:
                        completed_tasks[task_id] = self._completed_tasks[task_id]
                        task_ids.remove(task_id)

            if not task_ids:
                break

            time.sleep(0.1)

        return completed_tasks

    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()

        while self._running and (timeout is None or time.time() - start_time < timeout):
            with self._lock:
                if self._task_queue.empty() and not self._active_tasks:
                    return True

            time.sleep(0.1)

        return False
