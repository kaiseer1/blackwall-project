"""
Real-time threat detection using ML models and signature-based detection
"""

from typing import Dict, List, Optional, Any, Callable
import threading
import queue
import time

from src.ml.model_manager import ModelManager
from src.ml.feature_extractor import FeatureExtractor
from src.network.flow_tracker import Flow
from src.utils.logger import get_logger, ThreatLogger
from src.utils.metrics import get_metrics
from src.storage.database import Database


class ThreatDetector:
    """
    Detects threats using multiple detection methods:
    - ML-based detection
    - Signature-based detection
    - Anomaly detection
    """

    def __init__(
        self,
        model_manager: ModelManager,
        database: Database,
        confidence_threshold: float = 0.75
    ):
        """
        Initialize threat detector.

        Args:
            model_manager: ML model manager
            database: Database instance
            confidence_threshold: Minimum confidence for threat alerts
        """
        self.model_manager = model_manager
        self.database = database
        self.confidence_threshold = confidence_threshold

        self.feature_extractor = FeatureExtractor()
        self.logger = get_logger('blackwall.detection', log_file='logs/detection.log')
        self.threat_logger = ThreatLogger()
        self.metrics = get_metrics()

        self.detection_queue = queue.Queue()
        self.running = False
        self.detection_thread = None

        # Threat callbacks
        self.threat_callbacks: List[Callable] = []

        # Known attack signatures
        self.signatures = self._load_signatures()

    def _load_signatures(self) -> Dict[str, Any]:
        """Load threat signatures"""
        return {
            'port_scan': {
                'description': 'Port scanning activity',
                'indicators': ['multiple_ports', 'rapid_connections']
            },
            'dos_attack': {
                'description': 'Denial of Service attack',
                'indicators': ['high_packet_rate', 'syn_flood']
            },
            'brute_force': {
                'description': 'Brute force authentication',
                'indicators': ['multiple_failed_auth', 'rapid_attempts']
            }
        }

    def add_threat_callback(self, callback: Callable) -> None:
        """Add callback for threat notifications"""
        self.threat_callbacks.append(callback)

    def start(self) -> None:
        """Start threat detection"""
        if self.running:
            return

        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        self.logger.info("Threat detector started")

    def stop(self) -> None:
        """Stop threat detection"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        self.logger.info("Threat detector stopped")

    def analyze_flow(self, flow: Flow) -> Optional[Dict[str, Any]]:
        """
        Analyze a network flow for threats.

        Args:
            flow: Network flow to analyze

        Returns:
            Threat detection result or None
        """
        try:
            # Extract features
            flow_stats = flow.get_statistics()
            features = self.feature_extractor.extract_from_flow(flow_stats)

            if features is None:
                return None

            # ML-based detection
            ml_result = self._ml_detection(features)

            # Signature-based detection
            sig_result = self._signature_detection(flow_stats)

            # Behavioral anomaly detection
            anomaly_result = self._anomaly_detection(flow_stats)

            # Combine results
            threat_detected = False
            confidence = 0.0
            threat_types = []

            if ml_result and ml_result['is_threat']:
                threat_detected = True
                confidence = max(confidence, ml_result['confidence'])
                threat_types.append('ML_Detection')

            if sig_result:
                threat_detected = True
                confidence = max(confidence, sig_result['confidence'])
                threat_types.extend(sig_result['threat_types'])

            if anomaly_result and anomaly_result['is_anomaly']:
                threat_detected = True
                confidence = max(confidence, anomaly_result['confidence'])
                threat_types.append('Anomaly')

            if threat_detected and confidence >= self.confidence_threshold:
                threat_data = {
                    'flow_id': flow.flow_id,
                    'src_ip': flow.src_ip,
                    'dst_ip': flow.dst_ip,
                    'src_port': flow.src_port,
                    'dst_port': flow.dst_port,
                    'protocol': flow.protocol,
                    'threat_types': threat_types,
                    'confidence': confidence,
                    'severity': self._calculate_severity(confidence),
                    'timestamp': time.time(),
                    'details': {
                        'ml_result': ml_result,
                        'signature_result': sig_result,
                        'anomaly_result': anomaly_result
                    }
                }

                # Log threat
                self._handle_threat(threat_data)

                return threat_data

            return None

        except Exception as e:
            self.logger.error(f"Error analyzing flow: {e}", exc_info=True)
            return None

    def _ml_detection(self, features) -> Optional[Dict[str, Any]]:
        """ML-based threat detection"""
        try:
            if self.model_manager.model is None:
                return None

            result = self.model_manager.predict(features)

            if result is None:
                return None

            prediction, confidence = result

            return {
                'is_threat': bool(prediction),
                'confidence': confidence,
                'method': 'machine_learning'
            }

        except Exception as e:
            self.logger.error(f"ML detection error: {e}")
            return None

    def _signature_detection(self, flow_stats: Dict) -> Optional[Dict[str, Any]]:
        """Signature-based threat detection"""
        try:
            threats = []
            max_confidence = 0.0

            # Port scan detection
            if self._detect_port_scan(flow_stats):
                threats.append('port_scan')
                max_confidence = max(max_confidence, 0.85)

            # SYN flood detection
            if self._detect_syn_flood(flow_stats):
                threats.append('syn_flood')
                max_confidence = max(max_confidence, 0.90)

            # High packet rate (potential DoS)
            if self._detect_high_packet_rate(flow_stats):
                threats.append('dos_attack')
                max_confidence = max(max_confidence, 0.80)

            if threats:
                return {
                    'threat_types': threats,
                    'confidence': max_confidence
                }

            return None

        except Exception as e:
            self.logger.error(f"Signature detection error: {e}")
            return None

    def _anomaly_detection(self, flow_stats: Dict) -> Optional[Dict[str, Any]]:
        """Behavioral anomaly detection"""
        try:
            anomalies = []
            score = 0.0

            # Unusual packet sizes
            fwd_lengths = flow_stats.get('fwd_packet_lengths', [])
            if fwd_lengths:
                avg_size = sum(fwd_lengths) / len(fwd_lengths)
                if avg_size > 1400 or avg_size < 40:
                    anomalies.append('unusual_packet_size')
                    score += 0.3

            # Unusual flow duration
            duration = flow_stats.get('duration', 0)
            if duration > 300:  # 5 minutes
                anomalies.append('long_duration')
                score += 0.2

            # High packet count
            total_packets = flow_stats.get('fwd_packets', 0) + flow_stats.get('bwd_packets', 0)
            if total_packets > 1000:
                anomalies.append('high_packet_count')
                score += 0.4

            if anomalies:
                return {
                    'is_anomaly': True,
                    'anomalies': anomalies,
                    'confidence': min(score, 1.0)
                }

            return None

        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
            return None

    def _detect_port_scan(self, flow_stats: Dict) -> bool:
        """Detect port scanning"""
        # Simplified: check for SYN-only packets
        flags = flow_stats.get('flags', {})
        syn_count = flags.get('SYN', 0)
        ack_count = flags.get('ACK', 0)

        return syn_count > 5 and ack_count == 0

    def _detect_syn_flood(self, flow_stats: Dict) -> bool:
        """Detect SYN flood attack"""
        flags = flow_stats.get('flags', {})
        syn_count = flags.get('SYN', 0)
        total_packets = flow_stats.get('fwd_packets', 0) + flow_stats.get('bwd_packets', 0)

        if total_packets == 0:
            return False

        syn_ratio = syn_count / total_packets
        return syn_ratio > 0.8 and syn_count > 20

    def _detect_high_packet_rate(self, flow_stats: Dict) -> bool:
        """Detect high packet rate (potential DoS)"""
        duration = flow_stats.get('duration', 0)
        if duration < 0.001:
            return False

        total_packets = flow_stats.get('fwd_packets', 0) + flow_stats.get('bwd_packets', 0)
        packet_rate = total_packets / duration

        return packet_rate > 1000  # packets per second

    def _calculate_severity(self, confidence: float) -> str:
        """Calculate threat severity"""
        if confidence >= 0.95:
            return 'critical'
        elif confidence >= 0.85:
            return 'high'
        elif confidence >= 0.75:
            return 'medium'
        else:
            return 'low'

    def _handle_threat(self, threat_data: Dict[str, Any]) -> None:
        """Handle detected threat"""
        try:
            # Update metrics
            self.metrics.increment('threats.detected')
            self.metrics.increment(f'threats.severity.{threat_data["severity"]}')

            # Log to database
            self.database.insert_threat(
                threat_type=', '.join(threat_data['threat_types']),
                confidence=threat_data['confidence'],
                src_ip=threat_data['src_ip'],
                dst_ip=threat_data['dst_ip'],
                protocol=threat_data['protocol'],
                src_port=threat_data.get('src_port'),
                dst_port=threat_data.get('dst_port'),
                severity=threat_data['severity'],
                details=threat_data['details']
            )

            # Log to file
            self.threat_logger.log_threat(
                threat_type=', '.join(threat_data['threat_types']),
                confidence=threat_data['confidence'],
                src_ip=threat_data['src_ip'],
                dst_ip=threat_data['dst_ip'],
                protocol=threat_data['protocol'],
                details=threat_data['details']
            )

            # Notify callbacks
            for callback in self.threat_callbacks:
                try:
                    callback(threat_data)
                except Exception as e:
                    self.logger.error(f"Error in threat callback: {e}")

            self.logger.warning(
                f"Threat detected: {threat_data['threat_types']} from {threat_data['src_ip']} "
                f"(confidence: {threat_data['confidence']:.2f})"
            )

        except Exception as e:
            self.logger.error(f"Error handling threat: {e}", exc_info=True)

    def _detection_loop(self) -> None:
        """Background detection loop"""
        while self.running:
            try:
                # This could process queued flows for analysis
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'threats_detected': self.metrics.get_counter('threats.detected'),
            'threats_per_minute': self.metrics.get_rate('threats.detected', 60) * 60
        }
