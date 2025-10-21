"""
Feature extraction from network packets for ML models
Converts raw packet data into feature vectors for threat detection
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime


class FeatureExtractor:
    """
    Extracts features from network packets and flows for ML analysis.
    """

    # Standard feature columns (matching training dataset)
    FEATURE_COLUMNS = [
        'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
        'total_length_fwd_packets', 'total_length_bwd_packets',
        'fwd_packet_length_max', 'fwd_packet_length_min', 'fwd_packet_length_mean',
        'fwd_packet_length_std', 'bwd_packet_length_max', 'bwd_packet_length_min',
        'bwd_packet_length_mean', 'bwd_packet_length_std', 'flow_bytes_per_sec',
        'flow_packets_per_sec', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max',
        'flow_iat_min', 'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std',
        'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_total', 'bwd_iat_mean',
        'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags',
        'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fwd_header_length',
        'bwd_header_length', 'fwd_packets_per_sec', 'bwd_packets_per_sec',
        'min_packet_length', 'max_packet_length', 'packet_length_mean',
        'packet_length_std', 'packet_length_variance', 'fin_flag_count',
        'syn_flag_count', 'rst_flag_count', 'psh_flag_count', 'ack_flag_count',
        'urg_flag_count', 'cwe_flag_count', 'ece_flag_count', 'down_up_ratio',
        'average_packet_size', 'avg_fwd_segment_size', 'avg_bwd_segment_size',
        'fwd_header_length_total', 'fwd_avg_bytes_per_bulk', 'fwd_avg_packets_per_bulk',
        'fwd_avg_bulk_rate', 'bwd_avg_bytes_per_bulk', 'bwd_avg_packets_per_bulk',
        'bwd_avg_bulk_rate', 'subflow_fwd_packets', 'subflow_fwd_bytes',
        'subflow_bwd_packets', 'subflow_bwd_bytes', 'init_win_bytes_forward',
        'init_win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
        'active_mean', 'active_std', 'active_max', 'active_min',
        'idle_mean', 'idle_std', 'idle_max', 'idle_min'
    ]

    def __init__(self):
        """Initialize feature extractor"""
        self.flow_cache: Dict[str, Dict] = {}

    def extract_from_flow(self, flow_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract features from a network flow.

        Args:
            flow_data: Flow statistics dictionary

        Returns:
            Feature vector as numpy array, or None if insufficient data
        """
        try:
            features = {}

            # Basic flow statistics
            features['flow_duration'] = flow_data.get('duration', 0)
            features['total_fwd_packets'] = flow_data.get('fwd_packets', 0)
            features['total_bwd_packets'] = flow_data.get('bwd_packets', 0)
            features['total_length_fwd_packets'] = flow_data.get('fwd_bytes', 0)
            features['total_length_bwd_packets'] = flow_data.get('bwd_bytes', 0)

            # Packet length statistics
            fwd_lengths = flow_data.get('fwd_packet_lengths', [])
            bwd_lengths = flow_data.get('bwd_packet_lengths', [])
            all_lengths = fwd_lengths + bwd_lengths

            features['fwd_packet_length_max'] = max(fwd_lengths) if fwd_lengths else 0
            features['fwd_packet_length_min'] = min(fwd_lengths) if fwd_lengths else 0
            features['fwd_packet_length_mean'] = np.mean(fwd_lengths) if fwd_lengths else 0
            features['fwd_packet_length_std'] = np.std(fwd_lengths) if fwd_lengths else 0

            features['bwd_packet_length_max'] = max(bwd_lengths) if bwd_lengths else 0
            features['bwd_packet_length_min'] = min(bwd_lengths) if bwd_lengths else 0
            features['bwd_packet_length_mean'] = np.mean(bwd_lengths) if bwd_lengths else 0
            features['bwd_packet_length_std'] = np.std(bwd_lengths) if bwd_lengths else 0

            features['min_packet_length'] = min(all_lengths) if all_lengths else 0
            features['max_packet_length'] = max(all_lengths) if all_lengths else 0
            features['packet_length_mean'] = np.mean(all_lengths) if all_lengths else 0
            features['packet_length_std'] = np.std(all_lengths) if all_lengths else 0
            features['packet_length_variance'] = np.var(all_lengths) if all_lengths else 0

            # Flow rates
            duration = max(features['flow_duration'], 0.000001)  # Avoid division by zero
            total_bytes = features['total_length_fwd_packets'] + features['total_length_bwd_packets']
            total_packets = features['total_fwd_packets'] + features['total_bwd_packets']

            features['flow_bytes_per_sec'] = total_bytes / duration
            features['flow_packets_per_sec'] = total_packets / duration
            features['fwd_packets_per_sec'] = features['total_fwd_packets'] / duration
            features['bwd_packets_per_sec'] = features['total_bwd_packets'] / duration

            # Inter-arrival times
            fwd_iats = flow_data.get('fwd_iats', [])
            bwd_iats = flow_data.get('bwd_iats', [])
            all_iats = fwd_iats + bwd_iats

            features['flow_iat_mean'] = np.mean(all_iats) if all_iats else 0
            features['flow_iat_std'] = np.std(all_iats) if all_iats else 0
            features['flow_iat_max'] = max(all_iats) if all_iats else 0
            features['flow_iat_min'] = min(all_iats) if all_iats else 0

            features['fwd_iat_total'] = sum(fwd_iats) if fwd_iats else 0
            features['fwd_iat_mean'] = np.mean(fwd_iats) if fwd_iats else 0
            features['fwd_iat_std'] = np.std(fwd_iats) if fwd_iats else 0
            features['fwd_iat_max'] = max(fwd_iats) if fwd_iats else 0
            features['fwd_iat_min'] = min(fwd_iats) if fwd_iats else 0

            features['bwd_iat_total'] = sum(bwd_iats) if bwd_iats else 0
            features['bwd_iat_mean'] = np.mean(bwd_iats) if bwd_iats else 0
            features['bwd_iat_std'] = np.std(bwd_iats) if bwd_iats else 0
            features['bwd_iat_max'] = max(bwd_iats) if bwd_iats else 0
            features['bwd_iat_min'] = min(bwd_iats) if bwd_iats else 0

            # TCP flags
            flags = flow_data.get('flags', {})
            features['fin_flag_count'] = flags.get('FIN', 0)
            features['syn_flag_count'] = flags.get('SYN', 0)
            features['rst_flag_count'] = flags.get('RST', 0)
            features['psh_flag_count'] = flags.get('PSH', 0)
            features['ack_flag_count'] = flags.get('ACK', 0)
            features['urg_flag_count'] = flags.get('URG', 0)
            features['cwe_flag_count'] = flags.get('CWE', 0)
            features['ece_flag_count'] = flags.get('ECE', 0)

            features['fwd_psh_flags'] = flags.get('FWD_PSH', 0)
            features['bwd_psh_flags'] = flags.get('BWD_PSH', 0)
            features['fwd_urg_flags'] = flags.get('FWD_URG', 0)
            features['bwd_urg_flags'] = flags.get('BWD_URG', 0)

            # Header lengths
            features['fwd_header_length'] = flow_data.get('fwd_header_length', 0)
            features['bwd_header_length'] = flow_data.get('bwd_header_length', 0)
            features['fwd_header_length_total'] = features['fwd_header_length'] * features['total_fwd_packets']

            # Packet size statistics
            total_packets_nonzero = max(total_packets, 1)
            features['average_packet_size'] = total_bytes / total_packets_nonzero
            features['avg_fwd_segment_size'] = features['total_length_fwd_packets'] / max(features['total_fwd_packets'], 1)
            features['avg_bwd_segment_size'] = features['total_length_bwd_packets'] / max(features['total_bwd_packets'], 1)

            # Ratios
            features['down_up_ratio'] = features['total_bwd_packets'] / max(features['total_fwd_packets'], 1)

            # Bulk transfer features (simplified)
            features['fwd_avg_bytes_per_bulk'] = 0
            features['fwd_avg_packets_per_bulk'] = 0
            features['fwd_avg_bulk_rate'] = 0
            features['bwd_avg_bytes_per_bulk'] = 0
            features['bwd_avg_packets_per_bulk'] = 0
            features['bwd_avg_bulk_rate'] = 0

            # Subflow features
            features['subflow_fwd_packets'] = features['total_fwd_packets']
            features['subflow_fwd_bytes'] = features['total_length_fwd_packets']
            features['subflow_bwd_packets'] = features['total_bwd_packets']
            features['subflow_bwd_bytes'] = features['total_length_bwd_packets']

            # Window sizes
            features['init_win_bytes_forward'] = flow_data.get('init_win_fwd', 0)
            features['init_win_bytes_backward'] = flow_data.get('init_win_bwd', 0)

            # Active data
            features['act_data_pkt_fwd'] = flow_data.get('act_data_pkt_fwd', 0)
            features['min_seg_size_forward'] = min(fwd_lengths) if fwd_lengths else 0

            # Active/Idle times
            active_times = flow_data.get('active_times', [])
            idle_times = flow_data.get('idle_times', [])

            features['active_mean'] = np.mean(active_times) if active_times else 0
            features['active_std'] = np.std(active_times) if active_times else 0
            features['active_max'] = max(active_times) if active_times else 0
            features['active_min'] = min(active_times) if active_times else 0

            features['idle_mean'] = np.mean(idle_times) if idle_times else 0
            features['idle_std'] = np.std(idle_times) if idle_times else 0
            features['idle_max'] = max(idle_times) if idle_times else 0
            features['idle_min'] = min(idle_times) if idle_times else 0

            # Convert to numpy array in correct order
            feature_vector = np.array([
                features.get(col, 0) for col in self.FEATURE_COLUMNS
            ], dtype=np.float32)

            # Replace any NaN or inf values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

            return feature_vector.reshape(1, -1)

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def create_flow_summary(self, packets: List[Dict]) -> Dict[str, Any]:
        """
        Create flow summary from list of packets.

        Args:
            packets: List of packet dictionaries

        Returns:
            Flow summary dictionary
        """
        if not packets:
            return {}

        flow_summary = {
            'fwd_packets': 0,
            'bwd_packets': 0,
            'fwd_bytes': 0,
            'bwd_bytes': 0,
            'fwd_packet_lengths': [],
            'bwd_packet_lengths': [],
            'fwd_iats': [],
            'bwd_iats': [],
            'flags': defaultdict(int),
            'fwd_header_length': 0,
            'bwd_header_length': 0,
            'init_win_fwd': 0,
            'init_win_bwd': 0,
            'act_data_pkt_fwd': 0,
            'active_times': [],
            'idle_times': [],
            'duration': 0
        }

        # Determine flow direction based on first packet
        first_packet = packets[0]
        src_ip = first_packet.get('src_ip')
        src_port = first_packet.get('src_port')

        start_time = packets[0].get('timestamp', 0)
        end_time = packets[-1].get('timestamp', 0)
        flow_summary['duration'] = end_time - start_time

        last_fwd_time = None
        last_bwd_time = None

        for packet in packets:
            is_forward = (packet.get('src_ip') == src_ip and packet.get('src_port') == src_port)
            packet_len = packet.get('length', 0)
            timestamp = packet.get('timestamp', 0)

            if is_forward:
                flow_summary['fwd_packets'] += 1
                flow_summary['fwd_bytes'] += packet_len
                flow_summary['fwd_packet_lengths'].append(packet_len)

                if last_fwd_time is not None:
                    flow_summary['fwd_iats'].append(timestamp - last_fwd_time)
                last_fwd_time = timestamp
            else:
                flow_summary['bwd_packets'] += 1
                flow_summary['bwd_bytes'] += packet_len
                flow_summary['bwd_packet_lengths'].append(packet_len)

                if last_bwd_time is not None:
                    flow_summary['bwd_iats'].append(timestamp - last_bwd_time)
                last_bwd_time = timestamp

            # TCP flags
            if 'flags' in packet:
                for flag in packet['flags']:
                    flow_summary['flags'][flag] += 1

            # Headers
            header_len = packet.get('header_length', 20)
            if is_forward:
                flow_summary['fwd_header_length'] = header_len
            else:
                flow_summary['bwd_header_length'] = header_len

        return flow_summary
