"""Memory monitoring and tracking"""

import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time"""
    timestamp: datetime
    rss_mb: float  # Resident Set Size (actual memory used)
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percentage of system memory
    available_mb: float  # Available system memory


class MemoryMonitor:
    """Monitor process memory usage

    Features:
    - Track memory usage over time
    - Alert when threshold exceeded
    - Memory statistics and trends
    - Optional psutil integration
    """

    def __init__(self, threshold_mb: int = 500, use_psutil: bool = True, max_snapshots: int = 100):
        """Initialize memory monitor

        Args:
            threshold_mb: Memory threshold in MB for warnings
            use_psutil: Use psutil if available (more accurate)
            max_snapshots: Maximum number of snapshots to retain (prevents memory leak)
        """
        self.threshold_mb = threshold_mb
        self.use_psutil = use_psutil
        # Use deque with maxlen to prevent unbounded memory growth
        self.snapshots: deque = deque(maxlen=max_snapshots)

        # Try to import psutil
        self.psutil = None
        if use_psutil:
            try:
                import psutil
                self.psutil = psutil
            except ImportError:
                # Fall back to basic /proc/self/status parsing
                pass

    def get_memory_usage(self) -> MemorySnapshot | None:
        """Get current memory usage

        Returns:
            MemorySnapshot or None if unable to read
        """
        if self.psutil:
            return self._get_memory_psutil()
        else:
            return self._get_memory_proc()

    def _get_memory_psutil(self) -> MemorySnapshot | None:
        """Get memory usage using psutil"""
        try:
            process = self.psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            # Get system memory
            virtual_memory = self.psutil.virtual_memory()

            return MemorySnapshot(
                timestamp=datetime.now(),
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                percent=memory_percent,
                available_mb=virtual_memory.available / 1024 / 1024
            )
        except Exception:
            return None

    def _get_memory_proc(self) -> MemorySnapshot | None:
        """Get memory usage from /proc (Linux only, fallback)"""
        try:
            # Read /proc/self/status
            status_file = '/proc/self/status'
            if not os.path.exists(status_file):
                return None

            with open(status_file) as f:
                status = f.read()

            # Parse VmRSS and VmSize
            rss_kb = 0
            vms_kb = 0

            for line in status.split('\n'):
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                elif line.startswith('VmSize:'):
                    vms_kb = int(line.split()[1])

            # Read /proc/meminfo for system memory
            with open('/proc/meminfo') as f:
                meminfo = f.read()

            mem_total_kb = 0
            mem_available_kb = 0

            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    mem_available_kb = int(line.split()[1])

            percent = (rss_kb / mem_total_kb * 100) if mem_total_kb > 0 else 0.0

            return MemorySnapshot(
                timestamp=datetime.now(),
                rss_mb=rss_kb / 1024,
                vms_mb=vms_kb / 1024,
                percent=percent,
                available_mb=mem_available_kb / 1024
            )
        except Exception:
            return None

    def check_memory(self) -> dict | None:
        """Check current memory and return warning if threshold exceeded

        Returns:
            Dictionary with warning info or None if under threshold
        """
        snapshot = self.get_memory_usage()
        if snapshot is None:
            return None

        # Store snapshot (deque automatically handles size limit)
        self.snapshots.append(snapshot)

        # Check threshold
        if snapshot.rss_mb > self.threshold_mb:
            return {
                "level": "warning",
                "message": f"Memory usage high: {snapshot.rss_mb:.1f}MB (threshold: {self.threshold_mb}MB)",
                "rss_mb": snapshot.rss_mb,
                "vms_mb": snapshot.vms_mb,
                "percent": snapshot.percent,
                "available_mb": snapshot.available_mb,
                "threshold_mb": self.threshold_mb
            }

        return None

    def get_statistics(self) -> dict:
        """Get memory usage statistics

        Returns:
            Dictionary with memory statistics
        """
        if not self.snapshots:
            return {
                "count": 0,
                "current_mb": 0.0,
                "max_mb": 0.0,
                "avg_mb": 0.0,
                "trend": "unknown"
            }

        current = self.snapshots[-1]
        rss_values = [s.rss_mb for s in self.snapshots]

        # Calculate trend (comparing first half vs second half)
        if len(rss_values) >= 4:
            mid = len(rss_values) // 2
            first_half_avg = sum(rss_values[:mid]) / mid
            second_half_avg = sum(rss_values[mid:]) / (len(rss_values) - mid)

            if second_half_avg > first_half_avg * 1.2:
                trend = "increasing"
            elif second_half_avg < first_half_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "count": len(self.snapshots),
            "current_mb": current.rss_mb,
            "max_mb": max(rss_values),
            "min_mb": min(rss_values),
            "avg_mb": sum(rss_values) / len(rss_values),
            "trend": trend,
            "percent": current.percent,
            "available_mb": current.available_mb
        }

    def get_peak_memory(self) -> MemorySnapshot | None:
        """Get the snapshot with highest memory usage

        Returns:
            MemorySnapshot with peak usage or None
        """
        if not self.snapshots:
            return None

        return max(self.snapshots, key=lambda s: s.rss_mb)

    def reset(self):
        """Clear all snapshots"""
        self.snapshots.clear()


# Global monitor instance
_global_monitor: MemoryMonitor | None = None


def get_global_monitor() -> MemoryMonitor:
    """Get or create global memory monitor

    Returns:
        Global MemoryMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor


def configure_global_monitor(threshold_mb: int = 500, use_psutil: bool = True) -> None:
    """Configure global memory monitor

    Args:
        threshold_mb: Memory threshold for warnings
        use_psutil: Use psutil if available
    """
    global _global_monitor
    _global_monitor = MemoryMonitor(threshold_mb=threshold_mb, use_psutil=use_psutil)
