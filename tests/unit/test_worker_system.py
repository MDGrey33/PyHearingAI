#!/usr/bin/env python
"""
Test script for the worker system.

This script demonstrates how to use the worker system for parallel processing
of tasks, with resource monitoring and throttling.

Usage:
    python test_worker_system.py
"""

import logging
import random
import time
from typing import List

from pyhearingai.workers import (
    Task,
    TaskPriority,
    WorkerPool,
    global_rate_limiters,
    global_supervisor,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_workers")


def simulate_work(duration: float, task_id: int) -> str:
    """
    Simulate a task that takes some time to complete.

    Args:
        duration: Duration of the task in seconds
        task_id: ID of the task for reporting

    Returns:
        Result message
    """
    logger.info(f"Starting task {task_id} with duration {duration:.2f}s")

    # Simulate some CPU work
    start_time = time.time()
    end_time = start_time + duration

    # Do some actual work to simulate CPU usage
    count = 0
    while time.time() < end_time:
        count += 1
        if count % 1000000 == 0:
            # Check if we're done
            if time.time() >= end_time:
                break

    elapsed = time.time() - start_time
    logger.info(f"Completed task {task_id} in {elapsed:.2f}s")

    return f"Task {task_id} completed in {elapsed:.2f}s"


def simulate_api_call(service: str, request_id: int) -> str:
    """
    Simulate an API call with rate limiting.

    Args:
        service: Name of the API service
        request_id: ID of the request for reporting

    Returns:
        Result message
    """
    # Wait for rate limit token
    logger.info(f"Waiting for {service} rate limit token for request {request_id}")
    global_rate_limiters.wait(service)

    logger.info(f"Making {service} API call for request {request_id}")

    # Simulate API call duration
    duration = random.uniform(0.5, 2.0)
    time.sleep(duration)

    logger.info(f"Completed {service} API call for request {request_id}")
    return f"{service} API call {request_id} completed in {duration:.2f}s"


def test_worker_pool():
    """Test the worker pool with simulated tasks."""
    logger.info("Testing worker pool with simulated tasks")

    # Create a worker pool with 4 workers
    with WorkerPool(max_workers=4, name="test-pool") as pool:
        # Submit 10 tasks with varying priorities
        tasks: List[Task] = []

        # High priority tasks
        logger.info("Submitting high priority tasks")
        for i in range(3):
            duration = random.uniform(1.0, 3.0)
            task = pool.submit(simulate_work, duration, i, priority=TaskPriority.HIGH)
            tasks.append(task)

        # Normal priority tasks
        logger.info("Submitting normal priority tasks")
        for i in range(3, 7):
            duration = random.uniform(1.0, 3.0)
            task = pool.submit(simulate_work, duration, i, priority=TaskPriority.NORMAL)
            tasks.append(task)

        # Low priority tasks
        logger.info("Submitting low priority tasks")
        for i in range(7, 10):
            duration = random.uniform(1.0, 3.0)
            task = pool.submit(simulate_work, duration, i, priority=TaskPriority.LOW)
            tasks.append(task)

        # Wait for all tasks to complete
        logger.info("Waiting for tasks to complete")
        completed_tasks = pool.wait_for([task.id for task in tasks])

        # Print task results
        logger.info("Task results:")
        for task_id, task in completed_tasks.items():
            logger.info(f"  {task_id}: {task.result}")

    logger.info("Worker pool test completed")


def test_rate_limiting():
    """Test rate limiting for API calls."""
    logger.info("Testing rate limiting for API calls")

    # Create a worker pool with 8 workers
    with WorkerPool(max_workers=8, name="api-pool") as pool:
        # Submit 20 API calls (10 per service)
        tasks: List[Task] = []

        # OpenAI API calls (limited to 1 per second in this test)
        global_rate_limiters.add_limiter("test_openai", requests_per_minute=60)

        logger.info("Submitting OpenAI API calls")
        for i in range(10):
            task = pool.submit(simulate_api_call, "test_openai", i, priority=TaskPriority.NORMAL)
            tasks.append(task)

        # HuggingFace API calls (limited to 0.5 per second in this test)
        global_rate_limiters.add_limiter("test_huggingface", requests_per_minute=30)

        logger.info("Submitting HuggingFace API calls")
        for i in range(10, 20):
            task = pool.submit(
                simulate_api_call, "test_huggingface", i, priority=TaskPriority.NORMAL
            )
            tasks.append(task)

        # Wait for all tasks to complete
        logger.info("Waiting for API calls to complete")
        completed_tasks = pool.wait_for([task.id for task in tasks])

        # Print task results
        logger.info("API call results:")
        for task_id, task in completed_tasks.items():
            logger.info(f"  {task_id}: {task.result}")

    logger.info("Rate limiting test completed")


def test_resource_monitoring():
    """Test resource monitoring and throttling."""
    logger.info("Testing resource monitoring and throttling")

    # Create a test callback for high resource usage
    def on_high_resource():
        logger.warning("High resource usage detected, throttling workers")

    # Create a test callback for low resource usage
    def on_low_resource():
        logger.info("Resource usage normal, restoring worker count")

    # Register callbacks with the global supervisor
    global_supervisor.register_high_resource_callback(on_high_resource)
    global_supervisor.register_low_resource_callback(on_low_resource)

    # Get current resource usage
    status = global_supervisor.get_status()
    logger.info(
        f"Current resource usage: CPU {status['cpu_percent']:.1f}%, Memory {status['memory_percent']:.1f}%"
    )

    # Create a worker pool with dynamic resource-based scaling
    max_workers = 8
    logger.info(f"Creating worker pool with max {max_workers} workers")

    with WorkerPool(max_workers=max_workers, name="resource-pool") as pool:
        # Submit CPU-intensive tasks to trigger resource monitoring
        tasks: List[Task] = []

        logger.info("Submitting CPU-intensive tasks")
        for i in range(16):
            # Longer tasks to see resource monitoring in action
            duration = random.uniform(3.0, 6.0)
            task = pool.submit(simulate_work, duration, i, priority=TaskPriority.NORMAL)
            tasks.append(task)

        # Wait for some tasks to complete
        logger.info("Waiting for initial tasks to start")
        time.sleep(2)

        # Get resource status during processing
        status = global_supervisor.get_status()
        logger.info(
            f"Resource usage during processing: "
            f"CPU {status['cpu_percent']:.1f}%, Memory {status['memory_percent']:.1f}%"
        )

        # Check if throttling is active
        if global_supervisor.should_throttle():
            recommended = global_supervisor.get_recommended_workers(max_workers)
            logger.warning(f"Throttling active, recommended workers: {recommended}")

        # Wait for all tasks to complete
        logger.info("Waiting for all tasks to complete")
        pool.wait_all(timeout=60)

    logger.info("Resource monitoring test completed")


def main():
    """Run all worker system tests."""
    logger.info("Starting worker system tests")

    test_worker_pool()
    logger.info("-" * 40)

    test_rate_limiting()
    logger.info("-" * 40)

    test_resource_monitoring()

    logger.info("All worker system tests completed")


if __name__ == "__main__":
    main()
