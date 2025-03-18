import pandas as pd
import numpy as np
import joblib
import os
import gc
import logging
import argparse
from datetime import datetime
from pathlib import Path
from threading import Timer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scapy.all import sniff, IP, TCP, UDP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("blackwall.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model and dataset paths
MODEL_DIR = Path("models")
DATA_DIR = Path("C:/Users/cyber/OneDrive/Documents/blackwall project/datasets/MachineLearningCSV")
MODEL_FILE = MODEL_DIR / "blackwall_model.pkl"
SCALER_FILE = MODEL_DIR / "blackwall_scaler.pkl"

# Features used in training
FEATURE_COLUMNS = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "SYN Flag Count", "FIN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count"
]
LABEL_COLUMN = "Label"


def train_model():
    """Train the intrusion detection model and save it."""
    try:
        logger.info("Starting BlackWall Model Training...")
        dataset_files = [
            DATA_DIR / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            DATA_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
        ]
        
        dfs = []
        for file_path in dataset_files:
            if file_path.exists():
                logger.info(f"Processing: {file_path.name}")
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(lambda x: 1 if "Attack" in str(x) else 0)
                df.fillna(0, inplace=True)
                dfs.append(df)
        
        if not dfs:
            raise ValueError("No valid dataset files found.")
        
        df = pd.concat(dfs, ignore_index=True)

        # 🔥 Fix Inf/NaN Issues Before Training
        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
        df.dropna(inplace=True)  # Remove rows with NaN values
        
        X = df[FEATURE_COLUMNS]
        y = df[LABEL_COLUMN].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if min(y_train.value_counts()) < 1000:
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        
        model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight="balanced")
        model.fit(X_train_scaled, y_train)
        
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        
        logger.info("Model Training Completed Successfully.")
    except Exception as e:
        logger.error(f"Error in training: {e}", exc_info=True)


def extract_packet_features(packet):
    """Extracts relevant features from a live network packet."""
    features = {
        "Flow Duration": 1,
        "Total Fwd Packets": 1,
        "Total Backward Packets": 0,
        "Fwd Packet Length Max": len(packet) if packet.haslayer(IP) else 0,
        "Fwd Packet Length Min": len(packet) if packet.haslayer(IP) else 0,
        "SYN Flag Count": 1 if TCP in packet and packet[TCP].flags & 0x02 else 0,
        "FIN Flag Count": 1 if TCP in packet and packet[TCP].flags & 0x01 else 0,
        "RST Flag Count": 1 if TCP in packet and packet[TCP].flags & 0x04 else 0,
        "PSH Flag Count": 1 if TCP in packet and packet[TCP].flags & 0x08 else 0,
        "ACK Flag Count": 1 if TCP in packet and packet[TCP].flags & 0x10 else 0,
        "URG Flag Count": 1 if TCP in packet and packet[TCP].flags & 0x20 else 0
    }
    return np.array([features.get(col, 0) for col in FEATURE_COLUMNS])


def monitor_network():
    """Run real-time intrusion detection with enhanced logging."""
    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        logger.error("No trained model found! Please run `python blackwall.py --train` first.")
        return
    
    logger.info("Starting BlackWall Live Threat Detection...")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    def process_packet(packet):
        logger.info(f"Processing packet: {packet.summary()}")  # Log every processed packet
        features = extract_packet_features(packet)
        features_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)
        if prediction == 1:
            alert_message = f"⚠️ Potential Intrusion Detected! Packet Summary: {packet.summary()}"
            logger.warning(alert_message)
            logger.info("Logging detected intrusion into blackwall.log")
            with open("blackwall.log", "a") as log_file:
                log_file.write(alert_message + "\n")
    
    def stop_sniffing():
        logger.info("Stopping network monitoring...")
        os._exit(0)
    
    t = Timer(30, stop_sniffing)
    t.start()
    sniff(prn=process_packet, store=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlackWall: AI-Powered Cybersecurity Defense")
    parser.add_argument("--train", action="store_true", help="Train the model on cybersecurity dataset")
    parser.add_argument("--monitor", action="store_true", help="Run real-time intrusion detection")
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.monitor:
        monitor_network()
    else:
        print("Usage: python blackwall.py --train OR python blackwall.py --monitor")
