'''Resource monitoring utilities for CPU, memory, and GPU.

Provides functions to retrieve current system resource usage and a simple
CLI that prints a JSON summary.

The implementation prefers the `psutil` library when available because it
offers a cross‑platform API. If `psutil` is not installed the functions fall
back to standard library methods (e.g., ``os.getloadavg`` for CPU load) and
to ``/proc`` parsing on Linux.

GPU information is obtained via the ``nvidia-smi`` command. The code parses
the CSV output of ``nvidia-smi`` and returns a list of dictionaries – one
per GPU – containing the name, memory usage and total memory as MB, and
utilization percentage. If ``nvidia-smi`` is not present or returns an error,
an empty list is returned.
'''

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

# Optional psutil import – many parts of SEPilot already use it optionally.
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


@dataclass
class CPUInfo:
    """Snapshot of CPU usage information."""
    percent: float
    per_cpu_percent: list[float]
    load_average: tuple | None


def get_cpu_info() -> CPUInfo:
    """Return current CPU usage.

    - ``percent``: overall CPU utilization percentage.
    - ``per_cpu_percent``: list with utilization for each logical CPU.
    - ``load_average``: (1, 5, 15) minute load average on Unix, ``None`` on
      non‑Unix platforms.
    """
    if psutil:
        overall = psutil.cpu_percent(interval=0.1)
        per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    else:
        # Fallback: use os.getloadavg as a proxy for load; cannot get percent.
        overall = 0.0
        per_cpu = []
    try:
        load_avg = os.getloadavg()
    except Exception:
        load_avg = None
    return CPUInfo(percent=overall, per_cpu_percent=per_cpu, load_average=load_avg)


@dataclass
class MemoryInfo:
    """Snapshot of memory usage information."""
    total_mb: float
    available_mb: float
    used_mb: float
    free_mb: float
    percent: float


def get_memory_info() -> MemoryInfo:
    """Return current memory usage.

    If ``psutil`` is available the function uses ``psutil.virtual_memory``;
    otherwise it reads ``/proc/meminfo`` on Linux. On non‑Linux platforms
    without ``psutil`` the values default to ``0``.
    """
    if psutil:
        vm = psutil.virtual_memory()
        total = vm.total / (1024 * 1024)
        available = vm.available / (1024 * 1024)
        used = vm.used / (1024 * 1024)
        free = vm.free / (1024 * 1024)
        percent = vm.percent
    else:
        # Very basic Linux fallback
        total = available = used = free = percent = 0.0
        if os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo') as f:
                info = f.read()
            def _parse(key: str) -> int:
                for line in info.split('\n'):
                    if line.startswith(key):
                        return int(line.split()[1]) * 1024
                return 0
            total_bytes = _parse('MemTotal:')
            avail_bytes = _parse('MemAvailable:')
            free_bytes = _parse('MemFree:')
            used_bytes = total_bytes - free_bytes
            total = total_bytes / (1024 * 1024)
            available = avail_bytes / (1024 * 1024)
            free = free_bytes / (1024 * 1024)
            used = used_bytes / (1024 * 1024)
            percent = (used / total * 100) if total else 0.0
    return MemoryInfo(
        total_mb=total,
        available_mb=available,
        used_mb=used,
        free_mb=free,
        percent=percent,
    )


@dataclass
class GPUInfo:
    """Information about a single GPU as reported by nvidia-smi."""
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_percent: int | None = None


def _run_nvidia_smi() -> str | None:
    """Execute ``nvidia-smi`` and return its CSV output.

    The function catches ``FileNotFoundError`` (nvidia-smi not installed) and
    ``subprocess.CalledProcessError`` (command failed). In those cases ``None``
    is returned.
    """
    cmd = [
        'nvidia-smi',
        '--query-gpu=name,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits',
    ]
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return result.decode('utf-8').strip()
    except Exception:
        return None


def get_gpu_info() -> list[GPUInfo]:
    """Return a list of ``GPUInfo`` objects.

    If ``nvidia-smi`` is unavailable an empty list is returned.
    """
    output = _run_nvidia_smi()
    if not output:
        return []
    gpus: list[GPUInfo] = []
    for line in output.split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 4:
            continue
        name, mem_used, mem_total, util = parts
        try:
            mem_used_mb = int(mem_used)
            mem_total_mb = int(mem_total)
            util_percent = int(util) if util else None
        except ValueError:
            continue
        gpus.append(
            GPUInfo(
                name=name,
                memory_used_mb=mem_used_mb,
                memory_total_mb=mem_total_mb,
                utilization_percent=util_percent,
            )
        )
    return gpus


def collect_resource_snapshot() -> dict[str, Any]:
    """Collect CPU, memory and GPU information into a single dictionary.
    The structure is convenient for JSON serialisation.
    """
    cpu = get_cpu_info()
    mem = get_memory_info()
    gpus = get_gpu_info()
    return {
        'cpu': {
            'percent': cpu.percent,
            'per_cpu_percent': cpu.per_cpu_percent,
            'load_average': cpu.load_average,
        },
        'memory': {
            'total_mb': mem.total_mb,
            'available_mb': mem.available_mb,
            'used_mb': mem.used_mb,
            'free_mb': mem.free_mb,
            'percent': mem.percent,
        },
        'gpus': [
            {
                'name': g.name,
                'memory_used_mb': g.memory_used_mb,
                'memory_total_mb': g.memory_total_mb,
                'utilization_percent': g.utilization_percent,
            }
            for g in gpus
        ],
    }


def main() -> None:
    """Entry point for the command line tool.

    The tool prints a JSON representation of the current resource snapshot.
    Example::

        $ python -m sepilot.monitoring.resource_monitor
        {"cpu": {...}, "memory": {...}, "gpus": [...]}
    """
    snapshot = collect_resource_snapshot()
    print(json.dumps(snapshot, indent=2, default=str))


if __name__ == '__main__':
    main()
