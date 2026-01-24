"""Memory limiting and monitoring for training runs.

This module provides:
1. Hard memory limits to prevent system-wide memory exhaustion
2. Memory monitoring utilities
3. Aggressive cleanup when approaching limits

On Windows, uses periodic monitoring since resource limits aren't supported.
"""

import gc
import logging
import os
import sys
from typing import Optional, Callable

import psutil

logger = logging.getLogger(__name__)


# Default memory limit: 4GB - leaves room for browser and other apps
DEFAULT_MEMORY_LIMIT_MB = 4000

# Warning threshold: start aggressive cleanup at 80% of limit
WARNING_THRESHOLD_RATIO = 0.80

# Critical threshold: emergency measures at 95% of limit
CRITICAL_THRESHOLD_RATIO = 0.95


class MemoryLimiter:
    """Enforces memory limits and provides monitoring.

    Usage:
        limiter = MemoryLimiter(limit_mb=4000)
        limiter.start()

        # In training loop:
        if limiter.check_and_cleanup():
            # Memory was critically high, consider stopping
            pass
    """

    def __init__(
        self,
        limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        warning_callback: Optional[Callable[[float], None]] = None,
        critical_callback: Optional[Callable[[float], None]] = None
    ):
        """Initialize memory limiter.

        Args:
            limit_mb: Maximum memory usage in MB (default 4000 = 4GB)
            warning_callback: Called when warning threshold reached
            critical_callback: Called when critical threshold reached
        """
        self.limit_mb = limit_mb
        self.limit_bytes = limit_mb * 1024 * 1024
        self.warning_threshold = int(self.limit_bytes * WARNING_THRESHOLD_RATIO)
        self.critical_threshold = int(self.limit_bytes * CRITICAL_THRESHOLD_RATIO)

        self.warning_callback = warning_callback
        self.critical_callback = critical_callback

        self.process = psutil.Process(os.getpid())
        self._warning_logged = False
        self._critical_logged = False
        self._cleanup_count = 0

    def start(self) -> None:
        """Start memory limiting.

        On Unix, sets soft resource limit.
        On Windows, just logs the limit (enforced via monitoring).
        """
        logger.info(f"Memory limiter started: {self.limit_mb} MB limit")
        logger.info(f"  Warning threshold: {self.limit_mb * WARNING_THRESHOLD_RATIO:.0f} MB")
        logger.info(f"  Critical threshold: {self.limit_mb * CRITICAL_THRESHOLD_RATIO:.0f} MB")

        # Try to set resource limits on Unix
        if sys.platform != 'win32':
            try:
                import resource
                # Set soft limit (can be raised to hard limit)
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                # Don't exceed existing hard limit
                new_limit = min(self.limit_bytes, hard) if hard > 0 else self.limit_bytes
                resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
                logger.info(f"Set Unix resource limit: {new_limit / 1024 / 1024:.0f} MB")
            except (ImportError, ValueError, resource.error) as e:
                logger.warning(f"Could not set resource limit: {e}")
        else:
            logger.info("Windows detected - using monitoring-based enforcement")

    def get_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except psutil.NoSuchProcess:
            return 0.0

    def get_memory_percent(self) -> float:
        """Get memory usage as percentage of limit."""
        return (self.get_memory_mb() / self.limit_mb) * 100

    def is_warning(self) -> bool:
        """Check if memory usage is at warning level."""
        return self.process.memory_info().rss >= self.warning_threshold

    def is_critical(self) -> bool:
        """Check if memory usage is at critical level."""
        return self.process.memory_info().rss >= self.critical_threshold

    def force_cleanup(self) -> float:
        """Force aggressive garbage collection.

        Returns:
            Memory freed in MB
        """
        before = self.get_memory_mb()

        # Multiple GC passes for thorough cleanup
        gc.collect(0)  # Youngest generation
        gc.collect(1)  # Middle generation
        gc.collect(2)  # Oldest generation

        # Try to release memory back to OS
        try:
            import ctypes
            if sys.platform == 'win32':
                # Windows: try to trim working set
                ctypes.windll.kernel32.SetProcessWorkingSetSize(
                    ctypes.windll.kernel32.GetCurrentProcess(), -1, -1
                )
        except Exception:
            pass

        # PyTorch cache cleanup if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        after = self.get_memory_mb()
        freed = before - after
        self._cleanup_count += 1

        if freed > 0:
            logger.debug(f"GC freed {freed:.1f} MB (cleanup #{self._cleanup_count})")

        return freed

    def check_and_cleanup(self) -> bool:
        """Check memory and perform cleanup if needed.

        Returns:
            True if memory is at critical level (caller should consider stopping)
        """
        memory_mb = self.get_memory_mb()
        memory_bytes = memory_mb * 1024 * 1024

        # Critical level - emergency!
        if memory_bytes >= self.critical_threshold:
            if not self._critical_logged:
                logger.error(
                    f"CRITICAL: Memory at {memory_mb:.0f} MB "
                    f"({self.get_memory_percent():.1f}% of {self.limit_mb} MB limit)"
                )
                self._critical_logged = True

            # Aggressive cleanup
            self.force_cleanup()

            if self.critical_callback:
                self.critical_callback(memory_mb)

            return True

        # Warning level - cleanup
        if memory_bytes >= self.warning_threshold:
            if not self._warning_logged:
                logger.warning(
                    f"Memory warning: {memory_mb:.0f} MB "
                    f"({self.get_memory_percent():.1f}% of {self.limit_mb} MB limit)"
                )
                self._warning_logged = True

            self.force_cleanup()

            if self.warning_callback:
                self.warning_callback(memory_mb)

            # Reset critical flag if we recovered
            self._critical_logged = False
            return False

        # Normal - reset flags
        self._warning_logged = False
        self._critical_logged = False
        return False

    def get_status(self) -> dict:
        """Get memory status as a dictionary."""
        memory_mb = self.get_memory_mb()
        return {
            'current_mb': memory_mb,
            'limit_mb': self.limit_mb,
            'percent': self.get_memory_percent(),
            'is_warning': self.is_warning(),
            'is_critical': self.is_critical(),
            'cleanup_count': self._cleanup_count
        }


def get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    return psutil.virtual_memory().available / 1024 / 1024


def get_recommended_limit_mb(reserve_mb: int = 2000) -> int:
    """Get recommended memory limit based on system memory.

    Args:
        reserve_mb: Memory to reserve for other applications (default 2GB)

    Returns:
        Recommended limit in MB
    """
    total = psutil.virtual_memory().total / 1024 / 1024
    available = get_available_memory_mb()

    # Use 50% of total or available minus reserve, whichever is lower
    recommended = min(total * 0.5, available - reserve_mb)

    # Clamp to reasonable range
    recommended = max(1000, min(recommended, 8000))

    return int(recommended)
