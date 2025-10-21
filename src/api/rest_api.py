"""
REST API for BlackWall
Provides HTTP endpoints for monitoring and control
"""

from typing import Optional
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from urllib.parse import urlparse, parse_qs

from src.core.blackwall import BlackWall
from src.utils.logger import get_logger


class BlackWallAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for BlackWall API"""

    blackwall: Optional[BlackWall] = None
    api_key: Optional[str] = None

    def _set_headers(self, status: int = 200, content_type: str = 'application/json'):
        """Set HTTP headers"""
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-API-Key')
        self.end_headers()

    def _check_auth(self) -> bool:
        """Check API key authentication"""
        if not self.api_key:
            return True  # No auth required

        provided_key = self.headers.get('X-API-Key')
        return provided_key == self.api_key

    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response"""
        self._set_headers(status)
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        """Handle OPTIONS request (CORS preflight)"""
        self._set_headers()

    def do_GET(self):
        """Handle GET requests"""
        if not self._check_auth():
            self._send_json({'error': 'Unauthorized'}, 401)
            return

        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == '/api/status':
                self._handle_status()
            elif path == '/api/stats':
                self._handle_stats()
            elif path == '/api/threats':
                self._handle_threats()
            elif path == '/api/threats/stats':
                self._handle_threat_stats()
            elif path == '/api/flows':
                self._handle_flows()
            elif path == '/api/honeypot':
                self._handle_honeypot()
            elif path == '/api/blocked':
                self._handle_blocked_ips()
            else:
                self._send_json({'error': 'Not found'}, 404)

        except Exception as e:
            self._send_json({'error': str(e)}, 500)

    def _handle_status(self):
        """GET /api/status - System status"""
        data = {
            'version': BlackWall.VERSION,
            'running': self.blackwall.running if self.blackwall else False,
            'status': 'operational'
        }
        self._send_json(data)

    def _handle_stats(self):
        """GET /api/stats - System statistics"""
        if not self.blackwall:
            self._send_json({'error': 'BlackWall not initialized'}, 500)
            return

        stats = self.blackwall.get_stats()
        self._send_json(stats)

    def _handle_threats(self):
        """GET /api/threats - Recent threats"""
        if not self.blackwall:
            self._send_json({'error': 'BlackWall not initialized'}, 500)
            return

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        limit = int(params.get('limit', ['100'])[0])

        threats = self.blackwall.get_recent_threats(limit=limit)
        self._send_json({'threats': threats, 'count': len(threats)})

    def _handle_threat_stats(self):
        """GET /api/threats/stats - Threat statistics"""
        if not self.blackwall:
            self._send_json({'error': 'BlackWall not initialized'}, 500)
            return

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        hours = int(params.get('hours', ['24'])[0])

        stats = self.blackwall.get_threat_statistics(hours=hours)
        self._send_json(stats)

    def _handle_flows(self):
        """GET /api/flows - Active flows"""
        if not self.blackwall:
            self._send_json({'error': 'BlackWall not initialized'}, 500)
            return

        flows = self.blackwall.flow_tracker.get_active_flows()
        flow_data = [flow.get_statistics() for flow in flows[:100]]

        self._send_json({'flows': flow_data, 'count': len(flow_data)})

    def _handle_honeypot(self):
        """GET /api/honeypot - Honeypot stats"""
        if not self.blackwall:
            self._send_json({'error': 'BlackWall not initialized'}, 500)
            return

        stats = self.blackwall.honeypot.get_stats()
        interactions = self.blackwall.honeypot.get_interactions(limit=50)

        self._send_json({
            'stats': stats,
            'recent_interactions': interactions
        })

    def _handle_blocked_ips(self):
        """GET /api/blocked - Blocked IPs"""
        if not self.blackwall:
            self._send_json({'error': 'BlackWall not initialized'}, 500)
            return

        blocked = self.blackwall.firewall_manager.get_blocked_ips()
        blocked_list = [
            {'ip': ip, 'expires': exp.isoformat()}
            for ip, exp in blocked.items()
        ]

        self._send_json({'blocked_ips': blocked_list, 'count': len(blocked_list)})

    def log_message(self, format, *args):
        """Override to customize logging"""
        pass  # Suppress default logging


class BlackWallAPI:
    """
    REST API server for BlackWall.
    Provides HTTP endpoints for monitoring and management.
    """

    def __init__(
        self,
        blackwall: BlackWall,
        host: str = '127.0.0.1',
        port: int = 8080,
        api_key: Optional[str] = None
    ):
        """
        Initialize API server.

        Args:
            blackwall: BlackWall instance
            host: Host to bind to
            port: Port to listen on
            api_key: Optional API key for authentication
        """
        self.blackwall = blackwall
        self.host = host
        self.port = port
        self.api_key = api_key

        self.logger = get_logger('blackwall.api', log_file='logs/api.log')

        # Set class attributes for handler
        BlackWallAPIHandler.blackwall = blackwall
        BlackWallAPIHandler.api_key = api_key

        self.server = None
        self.server_thread = None

    def start(self) -> None:
        """Start API server"""
        try:
            self.server = HTTPServer((self.host, self.port), BlackWallAPIHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()

            self.logger.info(f"API server started on {self.host}:{self.port}")

        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}", exc_info=True)
            raise

    def stop(self) -> None:
        """Stop API server"""
        if self.server:
            self.server.shutdown()
            self.logger.info("API server stopped")
