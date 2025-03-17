"""
Rate limiting utilities for API calls.

This module provides rate limiting functionality to control
the frequency of API calls and prevent exceeding rate limits.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for controlling API call frequency.

    This class implements a token bucket algorithm for rate limiting,
    allowing a configurable number of requests per time period.
    """

    def __init__(
        self,
        requests_per_minute: float,
        max_burst: Optional[int] = None,
        name: str = "rate-limiter",
    ):
        """
        Initialize a rate limiter.

        Args:
            requests_per_minute: Maximum number of requests per minute
            max_burst: Maximum burst size (defaults to 1 minute worth of requests)
            name: Name for the rate limiter for logging
        """
        self.name = name
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_minute / 60.0
        self.max_burst = max_burst or int(requests_per_minute)

        # Token bucket state
        self.tokens = self.max_burst
        self.last_refill = time.time()
        self.lock = threading.RLock()

        logger.debug(
            f"Initialized {self.name} with {requests_per_minute} requests/minute "
            f"({self.requests_per_second:.2f} requests/second), max burst {self.max_burst}"
        )

    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Calculate tokens to add
        new_tokens = elapsed * self.requests_per_second

        # Update token count
        self.tokens = min(self.tokens + new_tokens, self.max_burst)
        self.last_refill = now

    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the rate limiter.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds

        Returns:
            True if tokens were acquired, False if timeout occurred
        """
        if tokens > self.max_burst:
            logger.warning(
                f"Requested {tokens} tokens exceeds max burst {self.max_burst}, "
                f"this will always time out"
            )
            return False

        start_time = time.time()

        with self.lock:
            while True:
                # Refill tokens
                self._refill_tokens()

                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logger.debug(
                        f"Acquired {tokens} tokens from {self.name}, "
                        f"{self.tokens:.2f} remaining"
                    )
                    return True

                # Check timeout
                if timeout is not None and time.time() - start_time >= timeout:
                    logger.debug(
                        f"Timeout waiting for {tokens} tokens from {self.name}, "
                        f"only {self.tokens:.2f} available"
                    )
                    return False

                # Calculate wait time for next token
                wait_time = (tokens - self.tokens) / self.requests_per_second
                wait_time = min(wait_time, 0.1)  # Cap at 100ms to check for timeout

                # Release lock while waiting
                self.lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self.lock.acquire()

    def wait(self, tokens: int = 1):
        """
        Wait until tokens are available and acquire them.

        Args:
            tokens: Number of tokens to acquire
        """
        while not self.acquire(tokens, timeout=None):
            pass

    @property
    def available_tokens(self) -> float:
        """Get the current number of available tokens."""
        with self.lock:
            self._refill_tokens()
            return self.tokens


class MultiRateLimiter:
    """
    Multiple rate limiters for different API endpoints or services.

    This class manages multiple rate limiters for different endpoints,
    allowing different rate limits for different API services.
    """

    def __init__(self, name: str = "multi-rate-limiter"):
        """
        Initialize a multi-rate limiter.

        Args:
            name: Name for the multi-rate limiter for logging
        """
        self.name = name
        self.limiters: Dict[str, RateLimiter] = {}
        self.lock = threading.RLock()

        logger.debug(f"Initialized {self.name}")

    def add_limiter(
        self, name: str, requests_per_minute: float, max_burst: Optional[int] = None
    ) -> RateLimiter:
        """
        Add a rate limiter for a specific service or endpoint.

        Args:
            name: Name of the service or endpoint
            requests_per_minute: Maximum number of requests per minute
            max_burst: Maximum burst size

        Returns:
            The created rate limiter
        """
        with self.lock:
            limiter = RateLimiter(
                requests_per_minute=requests_per_minute,
                max_burst=max_burst,
                name=f"{self.name}:{name}",
            )
            self.limiters[name] = limiter

            logger.debug(f"Added rate limiter for {name} to {self.name}")
            return limiter

    def get_limiter(self, name: str) -> Optional[RateLimiter]:
        """
        Get a rate limiter by name.

        Args:
            name: Name of the service or endpoint

        Returns:
            Rate limiter, or None if not found
        """
        return self.limiters.get(name)

    def acquire(self, name: str, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from a specific rate limiter.

        Args:
            name: Name of the service or endpoint
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds

        Returns:
            True if tokens were acquired, False if timeout occurred or limiter not found
        """
        limiter = self.get_limiter(name)
        if not limiter:
            logger.warning(f"Rate limiter for {name} not found in {self.name}")
            return False

        return limiter.acquire(tokens, timeout)

    def wait(self, name: str, tokens: int = 1):
        """
        Wait until tokens are available from a specific rate limiter and acquire them.

        Args:
            name: Name of the service or endpoint
            tokens: Number of tokens to acquire
        """
        limiter = self.get_limiter(name)
        if not limiter:
            logger.warning(f"Rate limiter for {name} not found in {self.name}")
            return

        limiter.wait(tokens)


# Global rate limiter for common services
global_rate_limiters = MultiRateLimiter(name="global")

# Configure default rate limiters for common services
global_rate_limiters.add_limiter("openai", requests_per_minute=60)  # 3500/min limit by default
global_rate_limiters.add_limiter("huggingface", requests_per_minute=30)
global_rate_limiters.add_limiter("default", requests_per_minute=10)
