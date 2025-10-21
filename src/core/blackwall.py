"""
BlackWall Core - Main orchestrator for the security system
Coordinates all subsystems: capture, detection, response, and monitoring
"""

import time
from typing import Optional, List, Dict, Any
import threading

from src.core.config import Config
from src.ml.model_manager import ModelManager
from src.network.packet_capture import PacketCapture
from src.network.flow_tracker import FlowTracker
from src.detection.threat_detector import ThreatDetector
from src.detection.anomaly_detector import AnomalyDetector
from src.honeypot.fpp import FalsePositiveProtection
from src.response.alert_manager import AlertManager
from src.response.firewall import FirewallManager
from src.storage.database import Database
from src.utils.logger import get_logger
from src.utils.metrics import get_metrics


class BlackWall:
    """
    Main BlackWall security system orchestrator.
    Coordinates all subsystems for comprehensive threat detection and response.
    """

    VERSION = "4.0.0"

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize BlackWall system.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = Config(config_path)

        # Setup logging
        self.logger = get_logger(
            'blackwall',
            log_file=f'{self.config.logging.log_dir}/blackwall.log',
            json_format=self.config.logging.json_format
        )

        self.logger.info(f"Initializing BlackWall v{self.VERSION}")

        # Initialize metrics
        self.metrics = get_metrics()

        # Initialize database
        self.database = Database(self.config.database.db_path)

        # Initialize ML components
        self.model_manager = ModelManager(model_dir="models")

        # Initialize network monitoring
        self.packet_capture = PacketCapture(
            interfaces=self.config.network.interfaces,
            buffer_size=self.config.network.packet_buffer_size,
            promiscuous=self.config.network.promiscuous_mode
        )
        self.flow_tracker = FlowTracker()

        # Initialize threat detection
        self.threat_detector = ThreatDetector(
            model_manager=self.model_manager,
            database=self.database,
            confidence_threshold=self.config.ml.confidence_threshold
        )
        self.anomaly_detector = AnomalyDetector()

        # Initialize response systems
        self.alert_manager = AlertManager(self.database)
        self.firewall_manager = FirewallManager(
            enabled=self.config.response.enable_firewall_integration,
            auto_block=self.config.response.auto_block
        )

        # Initialize honeypot
        self.honeypot = FalsePositiveProtection(
            database=self.database,
            ports=self.config.honeypot.ports
        )

        # System state
        self.running = False
        self.start_time = None

        # Processing thread
        self._processing_thread = None

        self.logger.info("BlackWall initialized successfully")

    def start(self) -> None:
        """Start all BlackWall subsystems"""
        if self.running:
            self.logger.warning("BlackWall already running")
            return

        self.logger.info("Starting BlackWall...")

        try:
            # Load ML model
            if self.config.detection.enable_ml_detection:
                self.logger.info("Loading ML model...")
                if not self.model_manager.load_model():
                    self.logger.warning("ML model not found. Run training first.")

            # Start flow tracker
            self.flow_tracker.start()

            # Start threat detector
            self.threat_detector.start()

            # Start firewall manager
            if self.config.response.enable_firewall_integration:
                self.firewall_manager.start()

            # Start honeypot
            if self.config.honeypot.enabled:
                self.honeypot.start()

            # Register packet handler
            self.packet_capture.add_packet_handler(self._handle_packet)

            # Register threat callback
            self.threat_detector.add_threat_callback(self._handle_threat)

            # Start packet capture
            self.packet_capture.start()

            # Start processing thread
            self.running = True
            self.start_time = time.time()
            self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._processing_thread.start()

            self.logger.info("BlackWall started successfully")

        except Exception as e:
            self.logger.error(f"Error starting BlackWall: {e}", exc_info=True)
            self.stop()
            raise

    def stop(self) -> None:
        """Stop all BlackWall subsystems"""
        self.logger.info("Stopping BlackWall...")

        self.running = False

        # Stop packet capture
        self.packet_capture.stop()

        # Stop flow tracker
        self.flow_tracker.stop()

        # Stop threat detector
        self.threat_detector.stop()

        # Stop firewall manager
        if self.config.response.enable_firewall_integration:
            self.firewall_manager.stop()

        # Stop honeypot
        if self.config.honeypot.enabled:
            self.honeypot.stop()

        # Wait for processing thread
        if self._processing_thread:
            self._processing_thread.join(timeout=5)

        self.logger.info("BlackWall stopped")

    def _handle_packet(self, packet: Dict[str, Any]) -> None:
        """
        Handle captured packet.

        Args:
            packet: Packet data
        """
        try:
            # Update metrics
            self.metrics.increment('packets.processed')

            # Track flow
            flow = self.flow_tracker.process_packet(packet)

            # Periodic flow analysis (every N packets)
            if flow and flow.fwd_packets + flow.bwd_packets >= 10:
                # Analyze flow for threats
                if self.config.detection.enable_ml_detection:
                    threat = self.threat_detector.analyze_flow(flow)

        except Exception as e:
            self.logger.error(f"Error handling packet: {e}")

    def _handle_threat(self, threat_data: Dict[str, Any]) -> None:
        """
        Handle detected threat.

        Args:
            threat_data: Threat information
        """
        try:
            # Send alert
            if self.config.response.enable_email_alerts or \
               self.config.response.enable_webhook_alerts:
                self.alert_manager.send_alert(threat_data)

            # Auto-block if enabled
            if self.config.response.auto_block and \
               threat_data.get('confidence', 0) >= self.config.response.auto_block_threshold:

                src_ip = threat_data.get('src_ip')
                if src_ip and not self.honeypot.is_known_attacker(src_ip):
                    # Don't block honeypot interactions yet
                    if self.config.response.enable_firewall_integration:
                        self.firewall_manager.block_ip(
                            ip=src_ip,
                            duration_minutes=self.config.response.block_duration_minutes,
                            reason=f"Threat: {', '.join(threat_data.get('threat_types', []))}"
                        )

        except Exception as e:
            self.logger.error(f"Error handling threat: {e}", exc_info=True)

    def _processing_loop(self) -> None:
        """Background processing loop"""
        while self.running:
            try:
                # Periodic flow analysis
                active_flows = self.flow_tracker.get_active_flows()

                for flow in active_flows:
                    # Analyze flows that have enough packets
                    if flow.fwd_packets + flow.bwd_packets >= 20:
                        if self.config.detection.enable_ml_detection:
                            self.threat_detector.analyze_flow(flow)

                time.sleep(5)  # Process every 5 seconds

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(1)

    def train_model(
        self,
        dataset_path: Optional[str] = None,
        model_type: str = "RandomForest",
        force: bool = False
    ) -> bool:
        """
        Train ML model.

        Args:
            dataset_path: Path to dataset
            model_type: Model type
            force: Force retraining

        Returns:
            True if successful
        """
        return self.model_manager.train_model(
            dataset_path=dataset_path or self.config.ml.dataset_path,
            model_type=model_type or self.config.ml.model_type,
            force=force
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        runtime = time.time() - self.start_time if self.start_time else 0

        # Get subsystem stats
        capture_stats = self.packet_capture.get_stats()
        flow_stats = self.flow_tracker.get_stats()
        detection_stats = self.threat_detector.get_stats()
        firewall_stats = self.firewall_manager.get_stats()
        honeypot_stats = self.honeypot.get_stats()

        # Get metrics
        all_metrics = self.metrics.get_all_metrics()

        return {
            'system': {
                'version': self.VERSION,
                'running': self.running,
                'uptime_seconds': runtime,
                'uptime_formatted': f"{runtime/3600:.2f} hours"
            },
            'capture': capture_stats,
            'flows': flow_stats,
            'detection': detection_stats,
            'firewall': firewall_stats,
            'honeypot': honeypot_stats,
            'metrics': all_metrics,
            'model': self.model_manager.get_model_info()
        }

    def get_recent_threats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent threats from database"""
        return self.database.get_recent_threats(limit=limit)

    def get_threat_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get threat statistics"""
        return self.database.get_threat_statistics(hours=hours)

    def cleanup_old_data(self) -> int:
        """Cleanup old database records"""
        return self.database.cleanup_old_records(
            retention_days=self.config.database.retention_days
        )
