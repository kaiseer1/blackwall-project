# BlackWall v4.0 - Quick Start Guide

Get up and running with BlackWall in 5 minutes!

## Prerequisites

- Linux or macOS system
- Python 3.8 or higher
- sudo/root access (for packet capture)

## Installation

### Option 1: Quick Install (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/kaiseer1/blackwall-project.git
cd blackwall-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the ML model
python blackwall.py --train

# 4. Start monitoring
sudo python blackwall.py --monitor
```

### Option 2: Docker Install

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f blackwall
```

## First Run

### 1. Train Your Model

Before monitoring, train the ML model:

```bash
python blackwall.py --train
```

This will:
- Load the sample dataset
- Train a RandomForest model
- Save it to `models/blackwall_model.joblib`
- Display performance metrics

**Expected output:**
```
Training RandomForest model on 4000 samples...
✓ Model training completed successfully!

Performance Metrics:
  Accuracy:  0.9542
  Precision: 0.9301
  Recall:    0.9687
  F1 Score:  0.9490
```

### 2. Start Monitoring

Run BlackWall with monitoring enabled:

```bash
sudo python blackwall.py --monitor
```

**Why sudo?** Packet capture requires raw socket access.

**Expected output:**
```
╔══════════════════════════════════════════════════════════╗
║             BlackWall v4.0.0 Security System             ║
║          AI-Driven Cybersecurity Defense Platform        ║
╚══════════════════════════════════════════════════════════╝

Starting BlackWall network monitoring...
✓ Network monitoring started
✓ Threat detection active

Monitoring... (Ctrl+C to stop)
Packets: 1245 | Flows: 12 | Threats: 0
```

### 3. Access the Dashboard

Start with API enabled:

```bash
sudo python blackwall.py --monitor --api
```

Then open in your browser:
```
http://localhost:8080/dashboard.html
```

## Basic Commands

### View Statistics

```bash
python blackwall.py --stats
```

Shows:
- System status
- Packet capture stats
- Active flows
- Threats detected
- Model performance

### View Recent Threats

```bash
python blackwall.py --threats
```

Shows the last 20 detected threats with details.

### Custom Configuration

```bash
python blackwall.py --monitor --config config/my_config.yaml
```

### Enable Features

```bash
# With REST API
sudo python blackwall.py --monitor --api

# With verbose logging
sudo python blackwall.py --monitor --verbose

# Force retrain model
python blackwall.py --train --force

# Use ensemble model
python blackwall.py --train --model Ensemble
```

## Configuration

Edit `config/config.yaml` to customize:

### Enable Honeypot

```yaml
honeypot:
  enabled: true
  ports: [22, 23, 3389, 445]
```

### Enable Auto-Blocking (Caution!)

```yaml
response:
  auto_block: true
  auto_block_threshold: 0.9
  enable_firewall_integration: true
```

### Change Network Interface

```yaml
network:
  interfaces:
    - eth0  # Your interface name
```

### Adjust Detection Sensitivity

```yaml
detection:
  alert_threshold: 0.7  # Lower = more sensitive
ml:
  confidence_threshold: 0.75
```

## Testing

### Generate Test Traffic

From another machine on your network:

```bash
# Port scan (should be detected)
nmap -sS your_blackwall_ip

# SYN flood (should be detected)
hping3 -S -p 80 --flood your_blackwall_ip
```

**Warning**: Only test in controlled environments!

### Check Detection

```bash
# View threats
python blackwall.py --threats

# Check database
sqlite3 blackwall.db "SELECT * FROM threats ORDER BY timestamp DESC LIMIT 5;"
```

## REST API Usage

With API enabled, access:

```bash
# System status
curl http://localhost:8080/api/status

# Statistics
curl http://localhost:8080/api/stats | jq

# Recent threats
curl http://localhost:8080/api/threats?limit=10 | jq

# Active flows
curl http://localhost:8080/api/flows | jq
```

## Troubleshooting

### "Permission denied"

Run with sudo:
```bash
sudo python blackwall.py --monitor
```

### "Scapy not available"

Install Scapy:
```bash
pip install scapy
```

### "Model not found"

Train the model first:
```bash
python blackwall.py --train
```

### "Dataset not found"

Ensure the dataset is in the correct location:
```bash
ls datasets/Sampled_Dataset_Example_cleaned.csv
```

If missing, check `archive/` or re-download.

### "Port already in use" (API)

Change the API port:
```bash
sudo python blackwall.py --monitor --api --api-port 8081
```

## Directory Structure

After installation:

```
blackwall-project/
├── blackwall.py           # Main entry point
├── config/
│   └── config.yaml        # Configuration
├── src/                   # Source code
├── models/                # Trained models
│   └── blackwall_model.joblib
├── logs/                  # Log files (created on first run)
├── datasets/              # Training data
└── blackwall.db          # Threat database (created on first run)
```

## Next Steps

1. ✅ **Customize Configuration**: Edit `config/config.yaml`
2. ✅ **Enable Features**: Try honeypot, API, auto-response
3. ✅ **Set Up Alerts**: Configure email/webhook notifications
4. ✅ **Deploy**: Use Docker for production deployment
5. ✅ **Read Full Docs**: See `README.md` for detailed information

## Getting Help

- **Full Documentation**: [README.md](README.md)
- **Migration Guide**: [MIGRATION.md](MIGRATION.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Issues**: GitHub Issues page

## Security Warning

⚠️ **Important Security Notes:**

1. **Testing First**: Always test in an isolated environment
2. **Auto-Block Risk**: Can cause false positive blocks
3. **Privilege Escalation**: Requires root access
4. **Network Impact**: Packet capture adds overhead
5. **Honeypot Exposure**: May attract attackers

## Quick Reference Card

```bash
# Training
python blackwall.py --train                    # Train model
python blackwall.py --train --model Ensemble   # Use ensemble

# Monitoring
sudo python blackwall.py --monitor             # Start monitoring
sudo python blackwall.py --monitor --api       # With API
sudo python blackwall.py --monitor --verbose   # Verbose mode

# Information
python blackwall.py --stats                    # View statistics
python blackwall.py --threats                  # View threats
python blackwall.py --version                  # Version info

# Configuration
python blackwall.py --monitor --config custom.yaml
```

---

**You're all set!** BlackWall is now protecting your network. 🛡️

For advanced features and deployment options, see the full [README.md](README.md).
