"""
Configuration management for BlackWall
Supports YAML configuration files with validation and defaults
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class NetworkConfig:
    """Network monitoring configuration"""
    interfaces: list = field(default_factory=lambda: ['any'])
    promiscuous_mode: bool = True
    packet_buffer_size: int = 10000
    capture_filter: str = ""
    max_packet_size: int = 65535


@dataclass
class MLConfig:
    """Machine learning configuration"""
    model_type: str = "RandomForest"
    model_path: str = "models/blackwall_model.joblib"
    dataset_path: str = "datasets/Sampled_Dataset_Example_cleaned.csv"
    confidence_threshold: float = 0.75
    retrain_interval_hours: int = 24
    enable_online_learning: bool = False


@dataclass
class DetectionConfig:
    """Threat detection configuration"""
    enable_ml_detection: bool = True
    enable_signature_detection: bool = True
    enable_anomaly_detection: bool = True
    enable_behavioral_analysis: bool = True
    alert_threshold: float = 0.7


@dataclass
class HoneypotConfig:
    """Honeypot/FPP configuration"""
    enabled: bool = True
    ports: list = field(default_factory=lambda: [22, 23, 3389, 445])
    deception_level: str = "medium"
    log_interactions: bool = True


@dataclass
class ResponseConfig:
    """Automated response configuration"""
    enabled: bool = True
    auto_block: bool = False
    auto_block_threshold: float = 0.9
    block_duration_minutes: int = 60
    enable_firewall_integration: bool = False
    enable_email_alerts: bool = False
    enable_webhook_alerts: bool = False


@dataclass
class APIConfig:
    """REST API configuration"""
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: Optional[str] = None
    enable_cors: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_path: str = "blackwall.db"
    retention_days: int = 30
    auto_vacuum: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_dir: str = "logs"
    json_format: bool = False
    max_file_size_mb: int = 100
    backup_count: int = 5


class Config:
    """
    Main configuration class for BlackWall.
    Loads and validates configuration from YAML file.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or "config/config.yaml"

        # Initialize with defaults
        self.network = NetworkConfig()
        self.ml = MLConfig()
        self.detection = DetectionConfig()
        self.honeypot = HoneypotConfig()
        self.response = ResponseConfig()
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.logging = LoggingConfig()

        # Load from file if exists
        if Path(self.config_path).exists():
            self.load()

    def load(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}

            # Update configurations
            if 'network' in config_data:
                self._update_dataclass(self.network, config_data['network'])

            if 'ml' in config_data:
                self._update_dataclass(self.ml, config_data['ml'])

            if 'detection' in config_data:
                self._update_dataclass(self.detection, config_data['detection'])

            if 'honeypot' in config_data:
                self._update_dataclass(self.honeypot, config_data['honeypot'])

            if 'response' in config_data:
                self._update_dataclass(self.response, config_data['response'])

            if 'api' in config_data:
                self._update_dataclass(self.api, config_data['api'])

            if 'database' in config_data:
                self._update_dataclass(self.database, config_data['database'])

            if 'logging' in config_data:
                self._update_dataclass(self.logging, config_data['logging'])

        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            print("Using default configuration")

    def save(self) -> None:
        """Save current configuration to YAML file"""
        config_data = {
            'network': self._dataclass_to_dict(self.network),
            'ml': self._dataclass_to_dict(self.ml),
            'detection': self._dataclass_to_dict(self.detection),
            'honeypot': self._dataclass_to_dict(self.honeypot),
            'response': self._dataclass_to_dict(self.response),
            'api': self._dataclass_to_dict(self.api),
            'database': self._dataclass_to_dict(self.database),
            'logging': self._dataclass_to_dict(self.logging)
        }

        # Create config directory if needed
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _update_dataclass(obj, data: dict) -> None:
        """Update dataclass fields from dictionary"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    @staticmethod
    def _dataclass_to_dict(obj) -> dict:
        """Convert dataclass to dictionary"""
        return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}

    def get_bool(self, section: str, key: str, default: bool = False) -> bool:
        """
        Get boolean configuration value (legacy method for compatibility).

        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found

        Returns:
            Boolean configuration value
        """
        section_map = {
            'General': self.__dict__,
            'Network': self.network,
            'ML': self.ml,
            'Detection': self.detection,
            'Honeypot': self.honeypot,
            'Response': self.response,
            'API': self.api,
            'Database': self.database,
            'Logging': self.logging
        }

        if section in section_map:
            obj = section_map[section]
            if hasattr(obj, key.lower()):
                return getattr(obj, key.lower())

        return default
