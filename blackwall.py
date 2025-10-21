#!/usr/bin/env python3
"""
BlackWall v4.0.0 - Main CLI Entry Point
AI-Driven Cybersecurity Defense System

Copyright (c) 2025 Basil Abdullah
Enhanced by Claude AI

This is a complete refactor with:
- Modular architecture
- Real packet capture (Scapy)
- Advanced ML detection
- REST API
- Database persistence
- Honeypot system
- Automated response
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.blackwall import BlackWall
from src.api.rest_api import BlackWallAPI
from src.utils.logger import get_logger


def print_banner():
    """Print BlackWall banner"""
    banner = """
╔══════════════════════════════════════════════════════════╗
║             BlackWall v4.0.0 Security System             ║
║          AI-Driven Cybersecurity Defense Platform        ║
║                                                          ║
║  Enhanced with:                                          ║
║    • Real-time Packet Capture & Analysis                ║
║    • Advanced ML Threat Detection                       ║
║    • Behavioral Anomaly Detection                       ║
║    • Adaptive Honeypot System                           ║
║    • Automated Threat Response                          ║
║    • REST API & Web Dashboard                           ║
║    • Comprehensive Logging & Metrics                    ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="BlackWall - AI-Driven Cybersecurity Defense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python blackwall_new.py --monitor              # Start network monitoring
  python blackwall_new.py --train                # Train ML model
  python blackwall_new.py --stats                # Show statistics
  python blackwall_new.py --threats              # Show recent threats
  python blackwall_new.py --api                  # Start with REST API
  python blackwall_new.py --config custom.yaml   # Use custom config
"""
    )

    # Commands
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument('--monitor', action='store_true',
                         help='Start network monitoring')
    commands.add_argument('--train', action='store_true',
                         help='Train machine learning model')
    commands.add_argument('--stats', action='store_true',
                         help='Show system statistics')
    commands.add_argument('--threats', action='store_true',
                         help='Show recent threats')
    commands.add_argument('--version', action='store_true',
                         help='Show version information')

    # Options
    parser.add_argument('--config', '-c', type=str,
                       help='Path to configuration file')
    parser.add_argument('--dataset', '-d', type=str,
                       help='Path to dataset for training')
    parser.add_argument('--model', '-m', type=str,
                       choices=['RandomForest', 'GradientBoosting', 'Ensemble'],
                       default='RandomForest',
                       help='ML model type')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force operation (e.g., retrain model)')
    parser.add_argument('--api', action='store_true',
                       help='Enable REST API server')
    parser.add_argument('--api-port', type=int, default=8080,
                       help='API server port (default: 8080)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--no-banner', action='store_true',
                       help='Suppress banner')

    args = parser.parse_args()

    # Print banner
    if not args.no_banner:
        print_banner()

    try:
        # Initialize BlackWall
        blackwall = BlackWall(config_path=args.config)
        logger = get_logger('blackwall.cli')

        # Handle commands
        if args.version:
            print(f"BlackWall v{BlackWall.VERSION}")
            print("Enhanced AI-Driven Cybersecurity Defense Platform")
            return 0

        elif args.train:
            print(f"\nTraining ML model...")
            print(f"Dataset: {args.dataset or 'default'}")
            print(f"Model Type: {args.model}")
            print("-" * 60)

            success = blackwall.train_model(
                dataset_path=args.dataset,
                model_type=args.model,
                force=args.force
            )

            if success:
                model_info = blackwall.model_manager.get_model_info()
                print("\n✓ Model training completed successfully!")
                print(f"\nPerformance Metrics:")
                print(f"  Accuracy:  {model_info.get('accuracy', 0):.4f}")
                print(f"  Precision: {model_info.get('precision', 0):.4f}")
                print(f"  Recall:    {model_info.get('recall', 0):.4f}")
                print(f"  F1 Score:  {model_info.get('f1_score', 0):.4f}")
                return 0
            else:
                print("\n✗ Model training failed. Check logs for details.")
                return 1

        elif args.stats:
            print("\nBlackWall System Statistics")
            print("=" * 60)

            stats = blackwall.get_stats()

            print(f"\nSystem:")
            print(f"  Version: {stats['system']['version']}")
            print(f"  Status: {'Running' if stats['system']['running'] else 'Stopped'}")
            print(f"  Uptime: {stats['system']['uptime_formatted']}")

            print(f"\nPacket Capture:")
            print(f"  Total Packets: {stats['capture']['total_packets']}")
            print(f"  Queue Size: {stats['capture']['queue_size']}/{stats['capture']['queue_capacity']}")

            print(f"\nNetwork Flows:")
            print(f"  Active Flows: {stats['flows']['active_flows']}")
            print(f"  Total Flows: {stats['flows']['total_flows']}")

            print(f"\nThreat Detection:")
            print(f"  Threats Detected: {stats['detection']['threats_detected']}")
            print(f"  Threats/Minute: {stats['detection']['threats_per_minute']:.2f}")

            if stats['honeypot']['active']:
                print(f"\nHoneypot:")
                print(f"  Active Honeypots: {stats['honeypot']['honeypots']}")
                print(f"  Interactions: {stats['honeypot']['total_interactions']}")

            print(f"\nML Model:")
            model_info = stats['model']
            if model_info['status'] == 'loaded':
                print(f"  Status: Loaded")
                print(f"  Algorithm: {model_info.get('algorithm', 'N/A')}")
                print(f"  Accuracy: {model_info.get('accuracy', 0):.4f}")

            return 0

        elif args.threats:
            print("\nRecent Threats")
            print("=" * 60)

            threats = blackwall.get_recent_threats(limit=20)

            if not threats:
                print("No threats detected recently.")
                return 0

            for threat in threats:
                print(f"\n[{threat['timestamp']}] {threat['severity'].upper()}")
                print(f"  Type: {threat['threat_type']}")
                print(f"  Source: {threat['src_ip']}:{threat.get('src_port', 'N/A')}")
                print(f"  Destination: {threat['dst_ip']}:{threat.get('dst_port', 'N/A')}")
                print(f"  Confidence: {threat['confidence']*100:.1f}%")

            return 0

        elif args.monitor:
            print("\nStarting BlackWall network monitoring...")
            print("Press Ctrl+C to stop\n")

            # Start BlackWall
            blackwall.start()

            # Start API if requested
            api_server = None
            if args.api:
                api_server = BlackWallAPI(
                    blackwall=blackwall,
                    host='0.0.0.0',
                    port=args.api_port
                )
                api_server.start()
                print(f"✓ REST API started on port {args.api_port}")

            print("✓ Network monitoring started")
            print("✓ Threat detection active")

            if blackwall.config.honeypot.enabled:
                print("✓ Honeypot system active")

            print("\nMonitoring... (Ctrl+C to stop)")

            try:
                # Monitor and display stats
                while True:
                    time.sleep(5)

                    stats = blackwall.get_stats()
                    metrics = stats['metrics']

                    status_line = (
                        f"Packets: {stats['capture']['total_packets']} | "
                        f"Flows: {stats['flows']['active_flows']} | "
                        f"Threats: {stats['detection']['threats_detected']}"
                    )

                    if blackwall.config.honeypot.enabled:
                        status_line += f" | Honeypot Hits: {stats['honeypot']['total_interactions']}"

                    print(f"\r{status_line}", end='', flush=True)

            except KeyboardInterrupt:
                print("\n\nStopping BlackWall...")
                blackwall.stop()
                if api_server:
                    api_server.stop()
                print("✓ BlackWall stopped successfully")
                return 0

        else:
            # No command specified, show help
            parser.print_help()
            return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130

    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
