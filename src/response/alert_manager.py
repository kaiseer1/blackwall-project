"""
Alert management and notification system
Supports multiple notification channels: email, webhook, Slack
"""

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime

from src.utils.logger import get_logger
from src.storage.database import Database


class AlertManager:
    """
    Manages threat alerts and notifications.
    Supports multiple notification channels.
    """

    def __init__(self, database: Database):
        """
        Initialize alert manager.

        Args:
            database: Database instance
        """
        self.database = database
        self.logger = get_logger('blackwall.alerts', log_file='logs/alerts.log')

        # Configuration (would come from config file)
        self.email_config = {
            'enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'from_addr': '',
            'to_addrs': []
        }

        self.webhook_config = {
            'enabled': False,
            'url': '',
            'headers': {}
        }

        self.slack_config = {
            'enabled': False,
            'webhook_url': ''
        }

    def send_alert(self, threat_data: Dict[str, Any]) -> None:
        """
        Send alert through configured channels.

        Args:
            threat_data: Threat information
        """
        try:
            # Format alert message
            alert_message = self._format_alert_message(threat_data)

            # Store in database
            self.database.insert_alert(
                alert_type='threat_detected',
                severity=threat_data.get('severity', 'medium'),
                message=alert_message,
                details=threat_data
            )

            # Send via email
            if self.email_config['enabled']:
                self._send_email_alert(alert_message, threat_data)

            # Send via webhook
            if self.webhook_config['enabled']:
                self._send_webhook_alert(threat_data)

            # Send to Slack
            if self.slack_config['enabled']:
                self._send_slack_alert(alert_message, threat_data)

            self.logger.info(f"Alert sent for threat from {threat_data.get('src_ip')}")

        except Exception as e:
            self.logger.error(f"Error sending alert: {e}", exc_info=True)

    def _format_alert_message(self, threat_data: Dict[str, Any]) -> str:
        """Format alert message"""
        severity = threat_data.get('severity', 'medium').upper()
        threat_types = ', '.join(threat_data.get('threat_types', []))
        confidence = threat_data.get('confidence', 0) * 100

        message = f"""
BLACKWALL SECURITY ALERT - {severity}

Threat Detected: {threat_types}
Confidence: {confidence:.1f}%
Severity: {severity}

Source IP: {threat_data.get('src_ip')}
Destination IP: {threat_data.get('dst_ip')}
Protocol: {threat_data.get('protocol')}
Source Port: {threat_data.get('src_port', 'N/A')}
Destination Port: {threat_data.get('dst_port', 'N/A')}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Recommended Action: {self._get_recommended_action(threat_data)}
"""
        return message.strip()

    def _get_recommended_action(self, threat_data: Dict[str, Any]) -> str:
        """Get recommended action based on threat"""
        severity = threat_data.get('severity', 'medium')

        if severity == 'critical':
            return "IMMEDIATE ACTION REQUIRED: Block source IP and investigate"
        elif severity == 'high':
            return "Review and consider blocking source IP"
        elif severity == 'medium':
            return "Monitor activity and review logs"
        else:
            return "Log for future reference"

    def _send_email_alert(self, message: str, threat_data: Dict[str, Any]) -> None:
        """Send email alert"""
        try:
            if not self.email_config.get('username') or not self.email_config.get('to_addrs'):
                return

            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_addr']
            msg['To'] = ', '.join(self.email_config['to_addrs'])
            msg['Subject'] = f"BlackWall Alert - {threat_data.get('severity', 'medium').upper()}"

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            )
            server.starttls()
            server.login(
                self.email_config['username'],
                self.email_config['password']
            )
            server.send_message(msg)
            server.quit()

            self.logger.info("Email alert sent")

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def _send_webhook_alert(self, threat_data: Dict[str, Any]) -> None:
        """Send webhook alert"""
        try:
            if not self.webhook_config.get('url'):
                return

            payload = {
                'event': 'threat_detected',
                'timestamp': datetime.now().isoformat(),
                'threat': threat_data
            }

            response = requests.post(
                self.webhook_config['url'],
                json=payload,
                headers=self.webhook_config.get('headers', {}),
                timeout=10
            )

            if response.status_code == 200:
                self.logger.info("Webhook alert sent")
            else:
                self.logger.warning(f"Webhook returned status {response.status_code}")

        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")

    def _send_slack_alert(self, message: str, threat_data: Dict[str, Any]) -> None:
        """Send Slack alert"""
        try:
            if not self.slack_config.get('webhook_url'):
                return

            severity = threat_data.get('severity', 'medium')
            color_map = {
                'critical': '#FF0000',
                'high': '#FF6600',
                'medium': '#FFD700',
                'low': '#00FF00'
            }

            payload = {
                'attachments': [{
                    'color': color_map.get(severity, '#FFD700'),
                    'title': f'BlackWall Security Alert - {severity.upper()}',
                    'text': message,
                    'footer': 'BlackWall v4.0',
                    'ts': int(datetime.now().timestamp())
                }]
            }

            response = requests.post(
                self.slack_config['webhook_url'],
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                self.logger.info("Slack alert sent")
            else:
                self.logger.warning(f"Slack returned status {response.status_code}")

        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")

    def configure_email(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str]
    ) -> None:
        """Configure email notifications"""
        self.email_config.update({
            'enabled': True,
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'from_addr': from_addr,
            'to_addrs': to_addrs
        })

    def configure_webhook(self, url: str, headers: Optional[Dict] = None) -> None:
        """Configure webhook notifications"""
        self.webhook_config.update({
            'enabled': True,
            'url': url,
            'headers': headers or {}
        })

    def configure_slack(self, webhook_url: str) -> None:
        """Configure Slack notifications"""
        self.slack_config.update({
            'enabled': True,
            'webhook_url': webhook_url
        })
