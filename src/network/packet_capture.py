"""
Real-time packet capture and analysis using Scapy
Captures network traffic and extracts relevant features
"""

import threading
import queue
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime
import time

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: Scapy not available. Packet capture will use simulation mode.")

from src.utils.logger import get_logger
from src.utils.metrics import get_metrics


class PacketCapture:
    """
    Handles real-time packet capture from network interfaces.
    Uses Scapy for packet sniffing and analysis.
    """

    def __init__(
        self,
        interfaces: Optional[List[str]] = None,
        buffer_size: int = 10000,
        promiscuous: bool = True
    ):
        """
        Initialize packet capture.

        Args:
            interfaces: List of interfaces to capture on (None for all)
            buffer_size: Maximum packets to buffer
            promiscuous: Enable promiscuous mode
        """
        self.interfaces = interfaces or ['any']
        self.buffer_size = buffer_size
        self.promiscuous = promiscuous

        self.logger = get_logger('blackwall.capture', log_file='logs/capture.log')
        self.metrics = get_metrics()

        self.packet_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread = None

        self.packet_handlers: List[Callable] = []
        self.total_packets = 0

    def add_packet_handler(self, handler: Callable[[Dict], None]) -> None:
        """
        Add a callback function to process packets.

        Args:
            handler: Function that takes a packet dictionary as argument
        """
        self.packet_handlers.append(handler)

    def start(self) -> None:
        """Start packet capture"""
        if self.running:
            self.logger.warning("Packet capture already running")
            return

        self.running = True

        if SCAPY_AVAILABLE:
            self.logger.info(f"Starting packet capture on interfaces: {self.interfaces}")
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
        else:
            self.logger.warning("Scapy not available, starting simulation mode")
            self.capture_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.capture_thread.start()

    def stop(self) -> None:
        """Stop packet capture"""
        self.logger.info("Stopping packet capture")
        self.running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=5)

    def _capture_loop(self) -> None:
        """Main packet capture loop (real mode)"""
        try:
            # Use first interface
            interface = self.interfaces[0] if self.interfaces else None

            self.logger.info(f"Capturing on interface: {interface}")

            sniff(
                iface=interface,
                prn=self._process_packet,
                store=False,
                stop_filter=lambda x: not self.running
            )

        except Exception as e:
            self.logger.error(f"Error in capture loop: {e}", exc_info=True)
            self.running = False

    def _process_packet(self, packet) -> None:
        """
        Process a captured packet.

        Args:
            packet: Scapy packet object
        """
        try:
            packet_data = self._extract_packet_data(packet)

            if packet_data:
                # Update metrics
                self.total_packets += 1
                self.metrics.increment('packets.captured')

                # Add to queue
                if not self.packet_queue.full():
                    self.packet_queue.put(packet_data)
                else:
                    self.metrics.increment('packets.dropped')

                # Call handlers
                for handler in self.packet_handlers:
                    try:
                        handler(packet_data)
                    except Exception as e:
                        self.logger.error(f"Error in packet handler: {e}")

        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")

    def _extract_packet_data(self, packet) -> Optional[Dict[str, Any]]:
        """
        Extract relevant data from packet.

        Args:
            packet: Scapy packet object

        Returns:
            Dictionary with packet information
        """
        try:
            if not packet.haslayer(IP):
                return None

            ip_layer = packet[IP]
            packet_data = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'src_ip': ip_layer.src,
                'dst_ip': ip_layer.dst,
                'protocol': ip_layer.proto,
                'length': len(packet),
                'ttl': ip_layer.ttl,
                'header_length': ip_layer.ihl * 4,
                'flags': []
            }

            # TCP layer
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                packet_data.update({
                    'protocol_name': 'TCP',
                    'src_port': tcp_layer.sport,
                    'dst_port': tcp_layer.dport,
                    'tcp_flags': self._get_tcp_flags(tcp_layer),
                    'seq': tcp_layer.seq,
                    'ack': tcp_layer.ack,
                    'window': tcp_layer.window
                })
                packet_data['flags'] = packet_data['tcp_flags']

                self.metrics.increment('packets.tcp')

            # UDP layer
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                packet_data.update({
                    'protocol_name': 'UDP',
                    'src_port': udp_layer.sport,
                    'dst_port': udp_layer.dport
                })

                self.metrics.increment('packets.udp')

            # ICMP layer
            elif packet.haslayer(ICMP):
                icmp_layer = packet[ICMP]
                packet_data.update({
                    'protocol_name': 'ICMP',
                    'icmp_type': icmp_layer.type,
                    'icmp_code': icmp_layer.code
                })

                self.metrics.increment('packets.icmp')

            else:
                packet_data['protocol_name'] = f'IP_{ip_layer.proto}'
                self.metrics.increment('packets.other')

            # Payload
            if packet.haslayer(Raw):
                payload = packet[Raw].load
                packet_data['payload_size'] = len(payload)
                packet_data['payload'] = payload[:100]  # First 100 bytes only

            return packet_data

        except Exception as e:
            self.logger.error(f"Error extracting packet data: {e}")
            return None

    def _get_tcp_flags(self, tcp_layer) -> List[str]:
        """Extract TCP flags from packet"""
        flags = []
        if tcp_layer.flags.F:
            flags.append('FIN')
        if tcp_layer.flags.S:
            flags.append('SYN')
        if tcp_layer.flags.R:
            flags.append('RST')
        if tcp_layer.flags.P:
            flags.append('PSH')
        if tcp_layer.flags.A:
            flags.append('ACK')
        if tcp_layer.flags.U:
            flags.append('URG')
        if tcp_layer.flags.E:
            flags.append('ECE')
        if tcp_layer.flags.C:
            flags.append('CWR')
        return flags

    def _simulation_loop(self) -> None:
        """Simulation mode for testing without Scapy"""
        import random

        self.logger.info("Running in simulation mode")

        ips = ['192.168.1.100', '192.168.1.101', '10.0.0.50', '172.16.0.10', '8.8.8.8']
        ports = [22, 80, 443, 3389, 445, 23, 8080]
        protocols = ['TCP', 'UDP', 'ICMP']

        while self.running:
            # Simulate packet arrival
            time.sleep(random.uniform(0.001, 0.01))

            packet_data = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'src_ip': random.choice(ips),
                'dst_ip': random.choice(ips),
                'protocol_name': random.choice(protocols),
                'src_port': random.choice(ports),
                'dst_port': random.choice(ports),
                'length': random.randint(60, 1500),
                'ttl': random.randint(32, 128),
                'header_length': 20,
                'flags': random.sample(['SYN', 'ACK', 'PSH', 'FIN'], k=random.randint(0, 2))
            }

            # Update metrics
            self.total_packets += 1
            self.metrics.increment('packets.captured')

            # Add to queue
            if not self.packet_queue.full():
                self.packet_queue.put(packet_data)

            # Call handlers
            for handler in self.packet_handlers:
                try:
                    handler(packet_data)
                except Exception as e:
                    self.logger.error(f"Error in packet handler: {e}")

    def get_packet(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get next packet from queue.

        Args:
            timeout: Timeout in seconds

        Returns:
            Packet data dictionary or None
        """
        try:
            return self.packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics"""
        return {
            'total_packets': self.total_packets,
            'queue_size': self.packet_queue.qsize(),
            'queue_capacity': self.buffer_size,
            'running': self.running
        }
