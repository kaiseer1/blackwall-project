"""
Enhanced logging system for BlackWall
Provides structured logging with rotation, encryption support, and multiple handlers
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON logs"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


def get_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Get or create a logger with specified configuration.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        json_format: Use JSON format for logs

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if json_format:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (with rotation)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)

        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class ThreatLogger:
    """Specialized logger for threat events"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(
            'blackwall.threats',
            log_file=str(self.log_dir / 'threats.log'),
            json_format=True
        )

    def log_threat(
        self,
        threat_type: str,
        confidence: float,
        src_ip: str,
        dst_ip: str,
        protocol: str,
        details: dict
    ) -> None:
        """Log a detected threat"""
        threat_data = {
            'threat_type': threat_type,
            'confidence': confidence,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': protocol,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Write to dedicated threat log file
        threat_log_path = self.log_dir / 'threats.jsonl'
        with open(threat_log_path, 'a') as f:
            f.write(json.dumps(threat_data) + '\n')

        self.logger.warning(f"Threat detected: {threat_type}", extra={'extra_data': threat_data})

    def get_recent_threats(self, limit: int = 100) -> list:
        """Retrieve recent threat events"""
        threat_log_path = self.log_dir / 'threats.jsonl'

        if not threat_log_path.exists():
            return []

        threats = []
        with open(threat_log_path, 'r') as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    threats.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        return threats
