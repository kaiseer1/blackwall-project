# Migration Guide: v3.4.0 → v4.0.0

This guide helps you migrate from BlackWall v3.4.0 to v4.0.0.

## What Changed?

BlackWall v4.0.0 is a **complete refactor** with a new architecture. While the core mission remains the same (AI-driven threat detection), the implementation has been modernized significantly.

## Breaking Changes

### 1. File Structure
**Before:**
```
blackwall-project/
├── blackwall.py (monolithic)
├── fix_dataset.py
├── requirements.txt
└── datasets/
```

**After:**
```
blackwall-project/
├── blackwall.py (new CLI)
├── src/ (modular codebase)
├── config/config.yaml
├── requirements.txt (updated)
└── datasets/
```

### 2. Command Line Interface

**v3.4.0:**
```bash
python blackwall.py --monitor
python blackwall.py --train
```

**v4.0.0:**
```bash
python blackwall.py --monitor  # Same command
python blackwall.py --train    # Same command
python blackwall.py --api      # New: Enable REST API
```

The basic commands remain the same for compatibility!

### 3. Configuration

**v3.4.0:**
- Hardcoded configuration in the script

**v4.0.0:**
- YAML configuration file: `config/config.yaml`
- Allows customization without code changes

## Migration Steps

### Step 1: Backup Your Data

```bash
# Backup your old trained models
cp -r models/ models_backup/

# Backup your datasets
cp -r datasets/ datasets_backup/
```

### Step 2: Update Dependencies

```bash
# Install new requirements
pip install -r requirements.txt
```

New dependencies include:
- `scapy` - For packet capture
- `imbalanced-learn` - For SMOTE
- Additional utilities

### Step 3: Configuration Setup

Create your configuration file:

```bash
# Copy default config
cp config/config.yaml config/my_config.yaml

# Edit as needed
nano config/my_config.yaml
```

### Step 4: Retrain Models (Recommended)

The new version uses enhanced feature extraction:

```bash
# Train with new architecture
python blackwall.py --train

# Or specify model type
python blackwall.py --train --model Ensemble
```

Your old models may still work, but retraining is recommended for best performance.

### Step 5: Test Run

```bash
# Test basic monitoring
sudo python blackwall.py --monitor

# Test with API
sudo python blackwall.py --monitor --api
```

## Feature Mapping

| v3.4.0 Feature | v4.0.0 Equivalent | Notes |
|----------------|-------------------|-------|
| Model training | `--train` | Enhanced with ensemble methods |
| Network monitoring | `--monitor` | Now with real packet capture |
| Statistics | `--stats` | More detailed metrics |
| Log viewing | `--threats` | Database-backed with history |
| - | `--api` | New REST API feature |
| - | Web Dashboard | New visualization |
| - | Honeypot | New deception system |
| - | Auto-response | New firewall integration |

## API Integration

If you were using BlackWall programmatically:

**v3.4.0:**
```python
from blackwall import BlackWall
bw = BlackWall()
bw.start_monitoring()
```

**v4.0.0:**
```python
from src.core.blackwall import BlackWall
bw = BlackWall()
bw.start()
```

The core API is similar but enhanced with new methods.

## Configuration Migration

If you had custom modifications in v3.4.0, here's how to migrate them:

### Model Configuration
**v3.4.0:** Edit code directly
```python
model = RandomForestClassifier(n_estimators=100, ...)
```

**v4.0.0:** Edit `config/config.yaml`
```yaml
ml:
  model_type: RandomForest
  confidence_threshold: 0.75
```

### Network Settings
**v3.4.0:** Hardcoded
```python
interfaces = ["eth0"]
```

**v4.0.0:** In config
```yaml
network:
  interfaces:
    - eth0
```

## New Features to Explore

### 1. REST API
```bash
# Start with API enabled
sudo python blackwall.py --monitor --api

# Access endpoints
curl http://localhost:8080/api/stats
```

### 2. Web Dashboard
Open `http://localhost:8080/dashboard.html` in your browser for real-time monitoring.

### 3. Honeypot System
Enable in `config/config.yaml`:
```yaml
honeypot:
  enabled: true
  ports: [22, 23, 3389, 445]
```

### 4. Automated Alerts
Configure email/webhook/Slack notifications:
```yaml
response:
  enable_email_alerts: true
  enable_webhook_alerts: true
```

### 5. Database Persistence
All threats are now stored in SQLite:
```bash
sqlite3 blackwall.db "SELECT * FROM threats LIMIT 10;"
```

## Troubleshooting

### Issue: "Scapy not available"
**Solution:** Install scapy
```bash
pip install scapy
```

### Issue: "Permission denied"
**Solution:** Run with sudo
```bash
sudo python blackwall.py --monitor
```

### Issue: "Model not found"
**Solution:** Retrain the model
```bash
python blackwall.py --train
```

### Issue: "Config file not found"
**Solution:** The default config is in `config/config.yaml`. Create it if missing:
```bash
mkdir -p config
cp config/config.yaml.example config/config.yaml
```

## Rollback to v3.4.0

If you need to rollback:

```bash
cd archive/v3.4.0
python blackwall.py --help
```

The old version is preserved in the `archive/` directory.

## Getting Help

- **Documentation**: See `README.md`
- **Configuration**: See `config/config.yaml` for all options
- **API Reference**: See REST API section in README
- **Issues**: Check the GitHub issues page

## Performance Considerations

v4.0.0 has different resource requirements:

| Resource | v3.4.0 | v4.0.0 |
|----------|--------|--------|
| Memory | ~50MB | ~200MB (with packet capture) |
| CPU | Low | Medium (multi-threaded) |
| Disk | Minimal | ~100MB (with database) |
| Network | None | Packet capture overhead |

## Security Notes

v4.0.0 introduces new security features:

1. **Firewall Integration**: Can auto-block IPs (disabled by default)
2. **API Authentication**: Supports API keys
3. **Honeypots**: May attract more attention to your system

**Recommendation**: Test in an isolated environment first.

## Next Steps

1. ✅ Complete migration steps above
2. ✅ Test basic functionality
3. ✅ Explore new features
4. ✅ Configure for your environment
5. ✅ Deploy to production

---

Welcome to BlackWall v4.0.0! 🛡️
