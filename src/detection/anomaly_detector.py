"""
Anomaly detection using statistical methods and unsupervised learning
"""

from typing import Dict, Any, Optional
from collections import deque
import numpy as np

from src.utils.logger import get_logger


class AnomalyDetector:
    """
    Detects anomalies in network traffic using statistical methods.
    Uses baseline profiling and deviation detection.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize anomaly detector.

        Args:
            window_size: Number of samples for baseline calculation
        """
        self.window_size = window_size
        self.logger = get_logger('blackwall.anomaly')

        # Baseline statistics
        self.packet_sizes = deque(maxlen=window_size)
        self.flow_durations = deque(maxlen=window_size)
        self.packet_rates = deque(maxlen=window_size)

        self.baseline_ready = False

    def update_baseline(self, flow_stats: Dict[str, Any]) -> None:
        """Update baseline statistics with new flow"""
        try:
            # Packet sizes
            all_lengths = (
                flow_stats.get('fwd_packet_lengths', []) +
                flow_stats.get('bwd_packet_lengths', [])
            )
            if all_lengths:
                avg_size = np.mean(all_lengths)
                self.packet_sizes.append(avg_size)

            # Flow duration
            duration = flow_stats.get('duration', 0)
            if duration > 0:
                self.flow_durations.append(duration)

            # Packet rate
            if duration > 0:
                total_packets = flow_stats.get('fwd_packets', 0) + flow_stats.get('bwd_packets', 0)
                rate = total_packets / duration
                self.packet_rates.append(rate)

            # Check if baseline is ready
            if len(self.packet_sizes) >= self.window_size // 2:
                self.baseline_ready = True

        except Exception as e:
            self.logger.error(f"Error updating baseline: {e}")

    def detect_anomaly(self, flow_stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect if a flow is anomalous.

        Args:
            flow_stats: Flow statistics

        Returns:
            Anomaly details or None
        """
        if not self.baseline_ready:
            # Still building baseline
            self.update_baseline(flow_stats)
            return None

        try:
            anomalies = []
            score = 0.0

            # Check packet size anomaly
            all_lengths = (
                flow_stats.get('fwd_packet_lengths', []) +
                flow_stats.get('bwd_packet_lengths', [])
            )
            if all_lengths:
                avg_size = np.mean(all_lengths)
                size_anomaly = self._check_anomaly(avg_size, self.packet_sizes)
                if size_anomaly:
                    anomalies.append('packet_size')
                    score += size_anomaly

            # Check duration anomaly
            duration = flow_stats.get('duration', 0)
            if duration > 0:
                duration_anomaly = self._check_anomaly(duration, self.flow_durations)
                if duration_anomaly:
                    anomalies.append('duration')
                    score += duration_anomaly

            # Check packet rate anomaly
            if duration > 0:
                total_packets = flow_stats.get('fwd_packets', 0) + flow_stats.get('bwd_packets', 0)
                rate = total_packets / duration
                rate_anomaly = self._check_anomaly(rate, self.packet_rates)
                if rate_anomaly:
                    anomalies.append('packet_rate')
                    score += rate_anomaly

            if anomalies:
                # Update baseline after detection
                self.update_baseline(flow_stats)

                return {
                    'is_anomaly': True,
                    'anomalies': anomalies,
                    'score': min(score / 3, 1.0),  # Normalize
                    'confidence': min(score / 2, 1.0)
                }

            # Update baseline with normal flow
            self.update_baseline(flow_stats)
            return None

        except Exception as e:
            self.logger.error(f"Error detecting anomaly: {e}")
            return None

    def _check_anomaly(self, value: float, baseline: deque) -> Optional[float]:
        """
        Check if value is anomalous using z-score.

        Args:
            value: Value to check
            baseline: Baseline values

        Returns:
            Anomaly score (0-1) or None
        """
        if len(baseline) < 10:
            return None

        try:
            baseline_array = np.array(baseline)
            mean = np.mean(baseline_array)
            std = np.std(baseline_array)

            if std == 0:
                return None

            # Calculate z-score
            z_score = abs((value - mean) / std)

            # Threshold: z-score > 3 is anomalous
            if z_score > 3:
                # Return normalized score
                return min(z_score / 10, 1.0)

            return None

        except Exception:
            return None
