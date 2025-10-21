"""
Metrics collection and monitoring for BlackWall
Tracks system performance, detection rates, and operational metrics
"""

import time
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from threading import Lock
from datetime import datetime, timedelta


class MetricsCollector:
    """
    Thread-safe metrics collection system.
    Tracks counters, gauges, histograms, and rates.
    """

    def __init__(self, window_size: int = 3600):
        """
        Initialize metrics collector.

        Args:
            window_size: Time window in seconds for rate calculations
        """
        self.window_size = window_size
        self._lock = Lock()

        # Metrics storage
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._timeseries: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # System start time
        self.start_time = time.time()

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter metric"""
        with self._lock:
            self._counters[name] += value
            self._timeseries[name].append((time.time(), value))

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric"""
        with self._lock:
            self._gauges[name] = value

    def record_value(self, name: str, value: float) -> None:
        """Record a value for histogram calculation"""
        with self._lock:
            self._histograms[name].append(value)

    def get_counter(self, name: str) -> int:
        """Get current counter value"""
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value"""
        with self._lock:
            return self._gauges.get(name)

    def get_rate(self, name: str, window: Optional[int] = None) -> float:
        """
        Calculate rate of a counter over time window.

        Args:
            name: Counter name
            window: Time window in seconds (default: instance window_size)

        Returns:
            Rate (events per second)
        """
        if window is None:
            window = self.window_size

        with self._lock:
            if name not in self._timeseries:
                return 0.0

            now = time.time()
            cutoff = now - window

            # Sum values within window
            count = sum(
                value for ts, value in self._timeseries[name]
                if ts >= cutoff
            )

            return count / window if window > 0 else 0.0

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get statistical summary of histogram"""
        with self._lock:
            if name not in self._histograms or not self._histograms[name]:
                return {'count': 0, 'min': 0, 'max': 0, 'mean': 0, 'p50': 0, 'p95': 0, 'p99': 0}

            values = sorted(self._histograms[name])
            count = len(values)

            return {
                'count': count,
                'min': values[0],
                'max': values[-1],
                'mean': sum(values) / count,
                'p50': values[int(count * 0.50)],
                'p95': values[int(count * 0.95)] if count > 20 else values[-1],
                'p99': values[int(count * 0.99)] if count > 100 else values[-1]
            }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary"""
        with self._lock:
            runtime = time.time() - self.start_time

            metrics = {
                'system': {
                    'uptime_seconds': runtime,
                    'uptime_formatted': str(timedelta(seconds=int(runtime)))
                },
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'rates': {},
                'histograms': {}
            }

            # Calculate rates for all counters
            for name in self._counters:
                metrics['rates'][f'{name}_per_second'] = self.get_rate(name, 60)
                metrics['rates'][f'{name}_per_minute'] = self.get_rate(name, 60) * 60
                metrics['rates'][f'{name}_per_hour'] = self.get_rate(name, 3600) * 3600

            # Get histogram stats
            for name in self._histograms:
                metrics['histograms'][name] = self.get_histogram_stats(name)

            return metrics

    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timeseries.clear()
            self.start_time = time.time()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        with self._lock:
            lines = []

            # Export counters
            for name, value in self._counters.items():
                metric_name = f'blackwall_{name.replace(".", "_")}'
                lines.append(f'# TYPE {metric_name} counter')
                lines.append(f'{metric_name} {value}')

            # Export gauges
            for name, value in self._gauges.items():
                metric_name = f'blackwall_{name.replace(".", "_")}'
                lines.append(f'# TYPE {metric_name} gauge')
                lines.append(f'{metric_name} {value}')

            return '\n'.join(lines)


# Global metrics instance
_global_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance"""
    return _global_metrics
