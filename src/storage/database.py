"""
SQLite database for persistent storage of threats, flows, and system events
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager
import json


class Database:
    """
    Database manager for BlackWall.
    Handles storage of threats, network flows, alerts, and metrics.
    """

    def __init__(self, db_path: str = "blackwall.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_schema()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Threats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    threat_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    src_ip TEXT NOT NULL,
                    dst_ip TEXT NOT NULL,
                    src_port INTEGER,
                    dst_port INTEGER,
                    protocol TEXT NOT NULL,
                    severity TEXT DEFAULT 'medium',
                    details TEXT,
                    mitigated BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Network flows table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS flows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flow_id TEXT UNIQUE NOT NULL,
                    src_ip TEXT NOT NULL,
                    dst_ip TEXT NOT NULL,
                    src_port INTEGER,
                    dst_port INTEGER,
                    protocol TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    packet_count INTEGER DEFAULT 0,
                    byte_count INTEGER DEFAULT 0,
                    flags TEXT,
                    is_malicious BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    acknowledged BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Honeypot interactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS honeypot_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    src_ip TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    protocol TEXT NOT NULL,
                    payload TEXT,
                    interaction_type TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_threats_timestamp ON threats(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_threats_src_ip ON threats(src_ip)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_flows_flow_id ON flows(flow_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')

    def insert_threat(
        self,
        threat_type: str,
        confidence: float,
        src_ip: str,
        dst_ip: str,
        protocol: str,
        src_port: Optional[int] = None,
        dst_port: Optional[int] = None,
        severity: str = 'medium',
        details: Optional[Dict] = None
    ) -> int:
        """Insert a new threat record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO threats (
                    timestamp, threat_type, confidence, src_ip, dst_ip,
                    src_port, dst_port, protocol, severity, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                threat_type,
                confidence,
                src_ip,
                dst_ip,
                src_port,
                dst_port,
                protocol,
                severity,
                json.dumps(details) if details else None
            ))
            return cursor.lastrowid

    def insert_flow(
        self,
        flow_id: str,
        src_ip: str,
        dst_ip: str,
        protocol: str,
        src_port: Optional[int] = None,
        dst_port: Optional[int] = None,
        packet_count: int = 0,
        byte_count: int = 0
    ) -> int:
        """Insert or update a network flow"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO flows (
                    flow_id, src_ip, dst_ip, src_port, dst_port, protocol,
                    start_time, packet_count, byte_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                flow_id,
                src_ip,
                dst_ip,
                src_port,
                dst_port,
                protocol,
                datetime.utcnow().isoformat(),
                packet_count,
                byte_count
            ))
            return cursor.lastrowid

    def insert_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict] = None
    ) -> int:
        """Insert a new alert"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, severity, message, details)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                alert_type,
                severity,
                message,
                json.dumps(details) if details else None
            ))
            return cursor.lastrowid

    def get_recent_threats(self, limit: int = 100, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent threats"""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM threats
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (cutoff, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_threat_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get threat statistics"""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total threats
            cursor.execute('SELECT COUNT(*) FROM threats WHERE timestamp >= ?', (cutoff,))
            total_threats = cursor.fetchone()[0]

            # By type
            cursor.execute('''
                SELECT threat_type, COUNT(*) as count
                FROM threats
                WHERE timestamp >= ?
                GROUP BY threat_type
            ''', (cutoff,))
            by_type = {row['threat_type']: row['count'] for row in cursor.fetchall()}

            # By severity
            cursor.execute('''
                SELECT severity, COUNT(*) as count
                FROM threats
                WHERE timestamp >= ?
                GROUP BY severity
            ''', (cutoff,))
            by_severity = {row['severity']: row['count'] for row in cursor.fetchall()}

            # Top attackers
            cursor.execute('''
                SELECT src_ip, COUNT(*) as count
                FROM threats
                WHERE timestamp >= ?
                GROUP BY src_ip
                ORDER BY count DESC
                LIMIT 10
            ''', (cutoff,))
            top_attackers = [(row['src_ip'], row['count']) for row in cursor.fetchall()]

            return {
                'total_threats': total_threats,
                'by_type': by_type,
                'by_severity': by_severity,
                'top_attackers': top_attackers
            }

    def cleanup_old_records(self, retention_days: int = 30) -> int:
        """Clean up old records"""
        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
        deleted = 0

        with self.get_connection() as conn:
            cursor = conn.cursor()

            for table in ['threats', 'flows', 'alerts', 'metrics', 'honeypot_interactions']:
                cursor.execute(f'DELETE FROM {table} WHERE created_at < ?', (cutoff,))
                deleted += cursor.rowcount

            # Vacuum database
            cursor.execute('VACUUM')

        return deleted
