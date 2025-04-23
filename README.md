# BlackWall v3.4.0

## AI-Driven Cybersecurity Defense System

### ⚠️ License Information (IMPORTANT)
BlackWall is dual-licensed under **Apache 2.0** and **AGPLv3**:

**1️⃣ Apache 2.0 (For open-source and non-commercial use)**
- Free to use, modify, and distribute.
- No requirement to open-source modifications.
- Ideal for individual developers, researchers, and educational use.

**2️⃣ AGPLv3 (For companies, enterprises, and SaaS/cloud providers)**
- If you modify and distribute BlackWall (including hosting it as a cloud service),
  you **must open-source** your modifications under the same license.
- Prevents companies from privatizing improvements without contributing back.
- If you wish to retain modifications privately, contact us for a **commercial license**.

**📩 Commercial License:**
For enterprise usage and closed-source extensions, contact **444019967@stu.bu.edu.sa**.

---

## Overview
BlackWall is a next-generation, AI-powered cybersecurity system that proactively monitors, detects, and neutralizes cyber threats in real-time. It leverages machine learning, deception-based defenses, and autonomous threat responses to adapt to emerging cyber risks.

---

## Key Features
- **AI-Based Intrusion Detection**
  - RandomForest and Gradient Boosting-based ML models
  - Anomaly detection, including zero-day exploit identification

- **Comprehensive Network Monitoring**
  - Deep packet inspection
  - Flow-based traffic analysis
  - Protocol behavior profiling

- **False Positive Protocol (FPP)**
  - Adaptive honeypot deployment
  - Attacker behavior analysis
  - Threat intelligence generation

- **Automated Threat Response**
  - Firewall rule generation
  - Process containment and isolation
  - SOC notification integration

- **Scalable Architecture**
  - Lightweight footprint
  - Modular component deployment
  - On-premise, cloud, or hybrid compatibility

---

## Project Structure
```
blackwall/
├── blackwall.py                  # Main application logic
├── fix_dataset.py                # Dataset preprocessing script (v3.4.0 patch)
├── datasets/
│   ├── Sampled_Dataset_Example_cleaned.csv  # Cleaned and updated dataset
│   └── Final_Preprocessed_Dataset_Sample.csv
├── models/
│   └── blackwall_model.joblib    # Trained AI model
├── LICENSE-Apache-2.0
├── LICENSE-AGPL-3.0
├── README.md
├── requirements.txt
└── other_source_files...
```

---

## Installation
See OS-specific instructions above for Windows and Linux/Mac setup, including:
- Creating virtual environments
- Installing dependencies
- Preparing datasets
- Training the model with `--train`

---

## Command Line Usage
- `python blackwall.py --monitor` — Start live monitoring
- `python blackwall.py --train` — Train ML model
- `python blackwall.py --logs` — View logs
- `python blackwall.py --version` — Check version

---

## Integration API (For Developers)
```python
from blackwall import BlackWall
bw = BlackWall()
bw.start_monitoring(interfaces=["eth0"])
bw.on_threat_detected(alert_handler)
stats = bw.get_stats()
```

---

## Machine Learning
- **Supervised Learning:**
  - Trained on labeled datasets
  - Detects known threats

- **Unsupervised Learning:**
  - Identifies anomalies
  - Critical for detecting unknown attack vectors

---

## Patch Notes — v3.4.0
- Added: `fix_dataset.py` script to clean and preprocess corrupted/missing values
- Added: `Sampled_Dataset_Example_cleaned.csv` to replace broken dataset
- Improved: Dataset preprocessing now handles NaN and infinite values
- Improved: Fallbacks added for missing label columns

---

## Future Roadmap
- Q2 2025: SIEM integration, enhanced cloud protection
- Q3 2025: Reinforcement learning integration, threat hunting AI
- Q4 2025: IoT support, forensic analytics tools

---

## About the Author
BlackWall is developed by **Basil Abdullah** at **Al-Baha University**.

> "The best defense is not just a good offense, but an intelligent, adaptive, and deceptive one."

---

## Final Notes
This version marks a major milestone in the BlackWall journey. While development will be paused, the foundation has been laid for future evolution through deeper research, advanced AI integration, and enterprise applications.

For serious inquiries or commercial use, contact: **444019967@stu.bu.edu.sa**




