BlackWall: AI-Driven Cybersecurity Defense
Author: Basil Abdullah
Version: 1.0
Date: 2025
Affiliation: Al-Baha University

Overview
BlackWall is an AI-powered cybersecurity framework designed to autonomously detect, contain, and neutralize cyber threats in real-time. Unlike traditional security solutions that rely on static rule-based detection, BlackWall leverages:

- Machine Learning for adaptive threat detection
- Deception-Based Security to mislead attackers
- Deep System Monitoring for proactive protection

BlackWall is modular, scalable, and efficient — integrating multiple cybersecurity layers to predict, deceive, and neutralize cyber threats before they escalate.

Key Features
- Network Traffic Monitoring – Continuously scans and analyzes real-time network packets for malicious behavior.
- AI-Powered Threat Detection – Uses a RandomForest-based ML model to detect anomalous patterns.
- Automated Threat Containment – Dynamically blocks and isolates threats via firewall adjustments.
- Deception-Based Security – Implements honeypots and False Positive Protocol (FPP) to mislead attackers.
- Kernel-Level Security Enforcement – Monitors system calls (Ring 0) for unauthorized activities.
- Global Threat Intelligence Integration – Uses external security feeds to stay updated on new threats.

Installation

Prerequisites
Ensure you have the following installed before running BlackWall:
- Python 3.9+
- pip (Python package manager)

Installing Dependencies
Install the required dependencies:
pip install -r requirements.txt

Dataset Installation
To train the AI model effectively, download the CIC-IDS-2017 dataset:
Download Dataset: http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip

1. Download the following CSV files:
   - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
2. Extract these files into the following path:
blackwall_project/datasets/MachineLearningCSV/

Folder Structure
blackwall_project/
│── datasets/                  # Contains network traffic datasets
│── models/                    # Trained ML models
│── blackwall.py               # Main execution script
│── requirements.txt           # Required Python libraries
│── README.md                  # Project documentation
│── LICENSE                    # Licensing details

Usage

Training the AI Model
To train the AI model using real-world cybersecurity datasets:
python blackwall.py --train

This will:
- Load the network traffic datasets
- Train the RandomForest-based intrusion detection model
- Generate a blackwall_model.pkl and blackwall_scaler.pkl in the models/ folder

Running the Cyber Defense System
Once trained, BlackWall can autonomously monitor and defend against cyber threats.

To start monitoring your network:
python blackwall.py --monitor

- The system will log all processed packets and detected intrusions inside blackwall.log.
- The monitor runs for 30 seconds by default.

How BlackWall Works
1. Loads Network Traffic Data – Supports PCAP and CSV format.
2. Feature Extraction – Analyzes packet behavior, flags, and timing for key insights.
3. Data Preprocessing – Cleans missing values, normalizes data, and removes outliers.
4. Machine Learning Model – Uses a RandomForest classifier to predict attack traffic.
5. Deception Techniques – Sends attackers into honeypots instead of allowing access.
6. Automated Containment – Blocks malicious traffic with dynamic firewall adjustments.

Logs & Results

Training Logs
- Training logs are saved in blackwall.log.
- Key outputs include:
  - Dataset loading
  - Feature extraction
  - Model performance details

Monitoring Logs
- While monitoring, all packets are logged in blackwall.log.
- Detected intrusions are logged as:
Potential Intrusion Detected! Packet Summary: <details>

To filter detected threats only:
cat blackwall.log | grep "Potential Intrusion Detected"

Troubleshooting

No Detected Intrusions in Logs?
- Ensure you have trained the model correctly with:
  python blackwall.py --train
- Simulate attack traffic with tools like nmap, hping3, or Metasploit to test detection.
- If needed, try adjusting the model's parameters for improved sensitivity.

Dataset Not Found Error?
- Confirm that your dataset files are correctly placed inside:
blackwall_project/datasets/MachineLearningCSV/

Future Enhancements
- Reinforcement Learning Integration – Adaptive AI that evolves with attack patterns.
- Cloud Security Deployment – Expanded support for AWS, Azure, and GCP.
- Blockchain Security Logs – Immutable logs to prevent tampering.
- Real-time EDR/NDR Integration – Support for enterprise-grade security solutions.

Contributions
Contributions are welcome. To contribute:
1. Fork the repository.
2. Submit a pull request with new features, bug fixes, or improvements.
3. Report issues and suggest enhancements.

Contact
For inquiries, collaborations, or commercial licensing, contact:

Email: 444019967@stu.bu.edu.sa
Discord: awaitingg

"BlackWall is the future of AI-driven cybersecurity. The goal is clear: deceive, neutralize, and outsmart attackers before they strike."

Glory to God.

