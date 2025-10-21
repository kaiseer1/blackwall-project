"""
False Positive Protection (FPP) - Advanced Honeypot System
Deploys deception services to attract and analyze attackers
"""

import socket
import threading
from typing import Dict, List, Optional, Any
import time
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.metrics import get_metrics
from src.storage.database import Database


class HoneypotService:
    """Individual honeypot service"""

    def __init__(self, port: int, protocol: str, service_name: str):
        self.port = port
        self.protocol = protocol
        self.service_name = service_name
        self.socket = None
        self.running = False
        self.thread = None
        self.interactions = []

    def start(self) -> bool:
        """Start honeypot service"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.listen(5)
            self.socket.settimeout(1.0)

            self.running = True
            self.thread = threading.Thread(target=self._accept_loop, daemon=True)
            self.thread.start()

            return True

        except Exception as e:
            print(f"Failed to start honeypot on port {self.port}: {e}")
            return False

    def stop(self) -> None:
        """Stop honeypot service"""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join(timeout=2)

    def _accept_loop(self) -> None:
        """Accept incoming connections"""
        while self.running:
            try:
                conn, addr = self.socket.accept()
                threading.Thread(
                    target=self._handle_connection,
                    args=(conn, addr),
                    daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception:
                break

    def _handle_connection(self, conn: socket.socket, addr: tuple) -> None:
        """Handle honeypot connection"""
        try:
            conn.settimeout(10.0)

            # Record interaction
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'src_ip': addr[0],
                'src_port': addr[1],
                'service': self.service_name,
                'port': self.port
            }

            # Send fake banner
            banner = self._get_banner()
            if banner:
                conn.send(banner.encode())

            # Try to receive data
            try:
                data = conn.recv(4096)
                if data:
                    interaction['payload'] = data[:1000]  # First 1KB
            except:
                pass

            self.interactions.append(interaction)

        except Exception:
            pass
        finally:
            conn.close()

    def _get_banner(self) -> Optional[str]:
        """Get service banner"""
        banners = {
            'SSH': 'SSH-2.0-OpenSSH_7.4\r\n',
            'FTP': '220 FTP Server Ready\r\n',
            'Telnet': 'Ubuntu 18.04 LTS\r\nlogin: ',
            'HTTP': 'HTTP/1.1 200 OK\r\nServer: Apache/2.4.41\r\n\r\n',
            'SMB': ''  # Binary protocol
        }
        return banners.get(self.service_name)


class FalsePositiveProtection:
    """
    Advanced honeypot system for attacker detection and analysis.
    Deploys deception services on common attack ports.
    """

    # Common attack ports and services
    HONEYPOT_PORTS = {
        22: 'SSH',
        23: 'Telnet',
        21: 'FTP',
        3389: 'RDP',
        445: 'SMB',
        139: 'NetBIOS',
        8080: 'HTTP-Proxy'
    }

    def __init__(self, database: Database, ports: Optional[List[int]] = None):
        """
        Initialize FPP system.

        Args:
            database: Database instance
            ports: List of ports to deploy honeypots on
        """
        self.database = database
        self.ports = ports or [22, 23, 3389, 445]

        self.logger = get_logger('blackwall.honeypot', log_file='logs/honeypot.log')
        self.metrics = get_metrics()

        self.services: Dict[int, HoneypotService] = {}
        self.active = False

    def start(self) -> None:
        """Start honeypot services"""
        if self.active:
            self.logger.warning("Honeypot already active")
            return

        self.logger.info(f"Starting honeypots on ports: {self.ports}")

        for port in self.ports:
            service_name = self.HONEYPOT_PORTS.get(port, f'Unknown-{port}')

            try:
                service = HoneypotService(port, 'TCP', service_name)

                if service.start():
                    self.services[port] = service
                    self.logger.info(f"Honeypot started: {service_name} on port {port}")
                else:
                    self.logger.warning(f"Failed to start honeypot on port {port}")

            except Exception as e:
                self.logger.error(f"Error starting honeypot on port {port}: {e}")

        self.active = True
        self.logger.info(f"FPP system active with {len(self.services)} honeypots")

    def stop(self) -> None:
        """Stop honeypot services"""
        self.logger.info("Stopping FPP system")

        for port, service in self.services.items():
            try:
                service.stop()
                self.logger.info(f"Stopped honeypot on port {port}")
            except Exception as e:
                self.logger.error(f"Error stopping honeypot on port {port}: {e}")

        self.services.clear()
        self.active = False

    def get_interactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent honeypot interactions"""
        all_interactions = []

        for service in self.services.values():
            all_interactions.extend(service.interactions)

        # Sort by timestamp (newest first)
        all_interactions.sort(key=lambda x: x['timestamp'], reverse=True)

        return all_interactions[:limit]

    def get_attacker_ips(self) -> Dict[str, int]:
        """Get attacker IPs and interaction counts"""
        ip_counts = {}

        for service in self.services.values():
            for interaction in service.interactions:
                ip = interaction['src_ip']
                ip_counts[ip] = ip_counts.get(ip, 0) + 1

        return ip_counts

    def log_interaction_to_db(self, interaction: Dict[str, Any]) -> None:
        """Log honeypot interaction to database"""
        try:
            self.database.get_connection()
            # Could extend Database class with honeypot interaction logging
            self.metrics.increment('honeypot.interactions')

        except Exception as e:
            self.logger.error(f"Error logging interaction: {e}")

    def is_known_attacker(self, ip: str) -> bool:
        """Check if IP has interacted with honeypots"""
        for service in self.services.values():
            for interaction in service.interactions:
                if interaction['src_ip'] == ip:
                    return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get honeypot statistics"""
        total_interactions = sum(
            len(service.interactions)
            for service in self.services.values()
        )

        return {
            'active': self.active,
            'honeypots': len(self.services),
            'ports': list(self.services.keys()),
            'total_interactions': total_interactions,
            'unique_attackers': len(self.get_attacker_ips())
        }
