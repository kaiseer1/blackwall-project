"""
Network flow tracking and management
Aggregates packets into flows for analysis
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime, timedelta
import time
import threading

from src.utils.logger import get_logger
from src.utils.metrics import get_metrics


class Flow:
    """Represents a network flow"""

    def __init__(self, flow_id: str, src_ip: str, dst_ip: str, protocol: str,
                 src_port: Optional[int] = None, dst_port: Optional[int] = None):
        self.flow_id = flow_id
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.protocol = protocol
        self.src_port = src_port
        self.dst_port = dst_port

        self.start_time = time.time()
        self.last_packet_time = self.start_time
        self.end_time = None

        # Counters
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0

        # Packet details
        self.fwd_packet_lengths: List[int] = []
        self.bwd_packet_lengths: List[int] = []
        self.fwd_timestamps: List[float] = []
        self.bwd_timestamps: List[float] = []

        # TCP flags
        self.flags: Dict[str, int] = defaultdict(int)

        # Other metadata
        self.is_malicious = False
        self.threat_score = 0.0
        self.packets: List[Dict] = []

    def add_packet(self, packet: Dict[str, Any], is_forward: bool = True) -> None:
        """Add a packet to this flow"""
        packet_len = packet.get('length', 0)
        timestamp = packet.get('timestamp', time.time())

        self.last_packet_time = timestamp
        self.packets.append(packet)

        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += packet_len
            self.fwd_packet_lengths.append(packet_len)
            self.fwd_timestamps.append(timestamp)
        else:
            self.bwd_packets += 1
            self.bwd_bytes += packet_len
            self.bwd_packet_lengths.append(packet_len)
            self.bwd_timestamps.append(timestamp)

        # Track TCP flags
        for flag in packet.get('flags', []):
            self.flags[flag] += 1

    def get_duration(self) -> float:
        """Get flow duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return self.last_packet_time - self.start_time

    def get_statistics(self) -> Dict[str, Any]:
        """Get flow statistics"""
        duration = self.get_duration()

        # Calculate inter-arrival times
        fwd_iats = []
        for i in range(1, len(self.fwd_timestamps)):
            fwd_iats.append(self.fwd_timestamps[i] - self.fwd_timestamps[i-1])

        bwd_iats = []
        for i in range(1, len(self.bwd_timestamps)):
            bwd_iats.append(self.bwd_timestamps[i] - self.bwd_timestamps[i-1])

        return {
            'flow_id': self.flow_id,
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'duration': duration,
            'fwd_packets': self.fwd_packets,
            'bwd_packets': self.bwd_packets,
            'fwd_bytes': self.fwd_bytes,
            'bwd_bytes': self.bwd_bytes,
            'fwd_packet_lengths': self.fwd_packet_lengths,
            'bwd_packet_lengths': self.bwd_packet_lengths,
            'fwd_iats': fwd_iats,
            'bwd_iats': bwd_iats,
            'flags': dict(self.flags),
            'is_malicious': self.is_malicious,
            'threat_score': self.threat_score,
            'start_time': self.start_time,
            'last_packet_time': self.last_packet_time
        }

    def close(self) -> None:
        """Mark flow as closed"""
        self.end_time = self.last_packet_time


class FlowTracker:
    """
    Tracks and manages network flows.
    Aggregates packets into bidirectional flows.
    """

    def __init__(self, timeout: int = 120, cleanup_interval: int = 60):
        """
        Initialize flow tracker.

        Args:
            timeout: Flow timeout in seconds (inactive flows are closed)
            cleanup_interval: How often to cleanup old flows (seconds)
        """
        self.timeout = timeout
        self.cleanup_interval = cleanup_interval

        self.flows: Dict[str, Flow] = {}
        self.closed_flows: List[Flow] = []

        self.logger = get_logger('blackwall.flows', log_file='logs/flows.log')
        self.metrics = get_metrics()

        self._lock = threading.Lock()
        self._cleanup_thread = None
        self.running = False

    def start(self) -> None:
        """Start flow tracker"""
        self.running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        self.logger.info("Flow tracker started")

    def stop(self) -> None:
        """Stop flow tracker"""
        self.running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        self.logger.info("Flow tracker stopped")

    def process_packet(self, packet: Dict[str, Any]) -> Optional[Flow]:
        """
        Process a packet and update flows.

        Args:
            packet: Packet data dictionary

        Returns:
            The flow this packet belongs to
        """
        try:
            # Create flow ID
            flow_id = self._create_flow_id(packet)
            reverse_flow_id = self._create_reverse_flow_id(packet)

            with self._lock:
                # Check if flow exists (forward or reverse direction)
                if flow_id in self.flows:
                    flow = self.flows[flow_id]
                    is_forward = True
                elif reverse_flow_id in self.flows:
                    flow = self.flows[reverse_flow_id]
                    is_forward = False
                else:
                    # Create new flow
                    flow = Flow(
                        flow_id=flow_id,
                        src_ip=packet.get('src_ip'),
                        dst_ip=packet.get('dst_ip'),
                        protocol=packet.get('protocol_name', 'UNKNOWN'),
                        src_port=packet.get('src_port'),
                        dst_port=packet.get('dst_port')
                    )
                    self.flows[flow_id] = flow
                    is_forward = True

                    self.metrics.increment('flows.created')
                    self.metrics.set_gauge('flows.active', len(self.flows))
                    self.logger.debug(f"New flow created: {flow_id}")

                # Add packet to flow
                flow.add_packet(packet, is_forward)

                return flow

        except Exception as e:
            self.logger.error(f"Error processing packet: {e}", exc_info=True)
            return None

    def _create_flow_id(self, packet: Dict[str, Any]) -> str:
        """Create unique flow identifier"""
        src_ip = packet.get('src_ip', '')
        dst_ip = packet.get('dst_ip', '')
        src_port = packet.get('src_port', 0)
        dst_port = packet.get('dst_port', 0)
        protocol = packet.get('protocol_name', 'UNKNOWN')

        return f"{src_ip}:{src_port}->{dst_ip}:{dst_port}:{protocol}"

    def _create_reverse_flow_id(self, packet: Dict[str, Any]) -> str:
        """Create reverse flow identifier"""
        src_ip = packet.get('src_ip', '')
        dst_ip = packet.get('dst_ip', '')
        src_port = packet.get('src_port', 0)
        dst_port = packet.get('dst_port', 0)
        protocol = packet.get('protocol_name', 'UNKNOWN')

        return f"{dst_ip}:{dst_port}->{src_ip}:{src_port}:{protocol}"

    def _cleanup_loop(self) -> None:
        """Cleanup inactive flows periodically"""
        while self.running:
            time.sleep(self.cleanup_interval)

            try:
                self._cleanup_inactive_flows()
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}", exc_info=True)

    def _cleanup_inactive_flows(self) -> None:
        """Close and remove inactive flows"""
        current_time = time.time()
        flows_to_close = []

        with self._lock:
            for flow_id, flow in list(self.flows.items()):
                # Check if flow is inactive
                if current_time - flow.last_packet_time > self.timeout:
                    flows_to_close.append(flow_id)

            # Close and archive flows
            for flow_id in flows_to_close:
                flow = self.flows.pop(flow_id)
                flow.close()
                self.closed_flows.append(flow)

                self.metrics.increment('flows.closed')
                self.logger.debug(f"Flow closed: {flow_id}")

            # Limit closed flows history
            if len(self.closed_flows) > 10000:
                self.closed_flows = self.closed_flows[-5000:]

            self.metrics.set_gauge('flows.active', len(self.flows))

    def get_flow(self, flow_id: str) -> Optional[Flow]:
        """Get a flow by ID"""
        with self._lock:
            return self.flows.get(flow_id)

    def get_active_flows(self) -> List[Flow]:
        """Get all active flows"""
        with self._lock:
            return list(self.flows.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get flow tracker statistics"""
        with self._lock:
            return {
                'active_flows': len(self.flows),
                'closed_flows': len(self.closed_flows),
                'total_flows': self.metrics.get_counter('flows.created')
            }
