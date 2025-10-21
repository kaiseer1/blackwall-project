"""
Firewall integration for automated threat response
Supports iptables (Linux) and simulated mode
"""

import subprocess
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import threading
import time

from src.utils.logger import get_logger


class FirewallManager:
    """
    Manages firewall rules for automated threat response.
    Supports iptables on Linux with automatic rule expiration.
    """

    def __init__(self, enabled: bool = False, auto_block: bool = False):
        """
        Initialize firewall manager.

        Args:
            enabled: Enable firewall integration
            auto_block: Enable automatic blocking
        """
        self.enabled = enabled
        self.auto_block = auto_block

        self.logger = get_logger('blackwall.firewall', log_file='logs/firewall.log')

        # Track blocked IPs with expiration times
        self.blocked_ips: Dict[str, datetime] = {}
        self._lock = threading.Lock()

        # Start cleanup thread
        self.running = False
        self.cleanup_thread = None

        if self.enabled:
            self._check_firewall_support()

    def start(self) -> None:
        """Start firewall manager"""
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        self.logger.info("Firewall manager started")

    def stop(self) -> None:
        """Stop firewall manager"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        self.logger.info("Firewall manager stopped")

    def _check_firewall_support(self) -> bool:
        """Check if firewall tools are available"""
        try:
            result = subprocess.run(
                ['which', 'iptables'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                self.logger.info("iptables found - firewall integration enabled")
                return True
            else:
                self.logger.warning("iptables not found - using simulation mode")
                return False

        except Exception as e:
            self.logger.warning(f"Could not check firewall support: {e}")
            return False

    def block_ip(
        self,
        ip: str,
        duration_minutes: int = 60,
        reason: str = "Threat detected"
    ) -> bool:
        """
        Block an IP address.

        Args:
            ip: IP address to block
            duration_minutes: Block duration in minutes (0 for permanent)
            reason: Reason for blocking

        Returns:
            True if successful
        """
        try:
            with self._lock:
                # Check if already blocked
                if ip in self.blocked_ips:
                    self.logger.info(f"IP {ip} already blocked")
                    return True

                if self.enabled and self._has_iptables():
                    # Add iptables rule
                    success = self._add_iptables_rule(ip)
                else:
                    # Simulation mode
                    success = True

                if success:
                    # Track blocked IP
                    if duration_minutes > 0:
                        expiration = datetime.now() + timedelta(minutes=duration_minutes)
                        self.blocked_ips[ip] = expiration
                    else:
                        self.blocked_ips[ip] = datetime.max

                    self.logger.warning(
                        f"Blocked IP {ip} for {duration_minutes} minutes - {reason}"
                    )
                    return True

                return False

        except Exception as e:
            self.logger.error(f"Error blocking IP {ip}: {e}", exc_info=True)
            return False

    def unblock_ip(self, ip: str) -> bool:
        """
        Unblock an IP address.

        Args:
            ip: IP address to unblock

        Returns:
            True if successful
        """
        try:
            with self._lock:
                if ip not in self.blocked_ips:
                    self.logger.info(f"IP {ip} not blocked")
                    return True

                if self.enabled and self._has_iptables():
                    # Remove iptables rule
                    success = self._remove_iptables_rule(ip)
                else:
                    # Simulation mode
                    success = True

                if success:
                    del self.blocked_ips[ip]
                    self.logger.info(f"Unblocked IP {ip}")
                    return True

                return False

        except Exception as e:
            self.logger.error(f"Error unblocking IP {ip}: {e}", exc_info=True)
            return False

    def _add_iptables_rule(self, ip: str) -> bool:
        """Add iptables rule to block IP"""
        try:
            # Add to INPUT chain
            cmd = [
                'sudo', 'iptables',
                '-I', 'INPUT',
                '-s', ip,
                '-j', 'DROP',
                '-m', 'comment',
                '--comment', 'BlackWall-auto-block'
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                self.logger.debug(f"iptables rule added for {ip}")
                return True
            else:
                self.logger.error(f"iptables command failed: {result.stderr.decode()}")
                return False

        except Exception as e:
            self.logger.error(f"Error adding iptables rule: {e}")
            return False

    def _remove_iptables_rule(self, ip: str) -> bool:
        """Remove iptables rule"""
        try:
            # Remove from INPUT chain
            cmd = [
                'sudo', 'iptables',
                '-D', 'INPUT',
                '-s', ip,
                '-j', 'DROP',
                '-m', 'comment',
                '--comment', 'BlackWall-auto-block'
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                self.logger.debug(f"iptables rule removed for {ip}")
                return True
            else:
                self.logger.error(f"iptables command failed: {result.stderr.decode()}")
                return False

        except Exception as e:
            self.logger.error(f"Error removing iptables rule: {e}")
            return False

    def _has_iptables(self) -> bool:
        """Check if iptables is available"""
        try:
            result = subprocess.run(['which', 'iptables'], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def _cleanup_loop(self) -> None:
        """Cleanup expired blocks"""
        while self.running:
            time.sleep(60)  # Check every minute

            try:
                self._cleanup_expired_blocks()
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    def _cleanup_expired_blocks(self) -> None:
        """Remove expired IP blocks"""
        now = datetime.now()
        expired_ips = []

        with self._lock:
            for ip, expiration in list(self.blocked_ips.items()):
                if expiration <= now:
                    expired_ips.append(ip)

            for ip in expired_ips:
                self.unblock_ip(ip)

        if expired_ips:
            self.logger.info(f"Expired blocks removed: {len(expired_ips)} IPs")

    def is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        with self._lock:
            return ip in self.blocked_ips

    def get_blocked_ips(self) -> Dict[str, datetime]:
        """Get all blocked IPs"""
        with self._lock:
            return self.blocked_ips.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get firewall statistics"""
        with self._lock:
            return {
                'enabled': self.enabled,
                'auto_block': self.auto_block,
                'blocked_ips': len(self.blocked_ips),
                'has_iptables': self._has_iptables() if self.enabled else False
            }
