import pandas as pd
import numpy as np
import joblib
import os
import gc
import logging
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("blackwall_training.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    try:
        start_time = datetime.now()
        logger.info("BlackWall AI Model Training Started")
        
        # Create directories if they don't exist
        model_dir = Path("models")
        data_dir = Path.cwd() / "datasets" / "MachineLearningCSV"
        model_dir.mkdir(exist_ok=True)
        
        # Dataset files
        dataset_files = [
            data_dir / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            data_dir / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
        ]
        
        # Features selection
        feature_columns = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
            'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
            'Flow IAT Min', 'SYN Flag Count', 'FIN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count'
        ]
        label_column = 'Label'
        
        # Load data with chunking to manage memory
        logger.info("Loading datasets...")
        dfs = []
        for file_path in dataset_files:
            if file_path.exists():
                logger.info(f"Processing: {file_path.name}")
                chunks = pd.read_csv(file_path, chunksize=100000)
                for chunk in chunks:
                    chunk.columns = chunk.columns.str.strip()
                    if label_column not in chunk.columns:
                        logger.warning(f"Label column missing in {file_path.name}, skipping...")
                        continue
                    available_features = list(set(feature_columns) & set(chunk.columns))
                    missing_features = set(feature_columns) - set(chunk.columns)
                    for feature in missing_features:
                        chunk[feature] = 0  # Fill missing features with 0
                    chunk = chunk[available_features + [label_column]]
                    attack_labels = ["DDoS", "PortScan", "Infilteration", "Web Attack", "Brute Force"]
                    chunk[label_column] = chunk[label_column].apply(lambda x: 1 if any(label in str(x) for label in attack_labels) else 0)
                    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                    chunk.fillna(0, inplace=True)
                    dfs.append(chunk)
                    gc.collect()
            else:
                logger.warning(f"File not found: {file_path}")
        
        if not dfs:
            raise ValueError("No valid dataset files found!")
        
        df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Class distribution: {df[label_column].value_counts()}")
        
        if len(df[label_column].unique()) < 2:
            raise ValueError("Dataset contains only one class. SMOTE cannot be applied!")
        
        X = df[feature_columns]
        y = df[label_column].astype(int)
        del df
        gc.collect()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        del X, y
        gc.collect()
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        del X_train
        gc.collect()
        
        if min(y_train.value_counts()) < 1000:
            logger.info("Applying SMOTE for class balance...")
            smote = SMOTE(random_state=42, n_jobs=-1)
            X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
        else:
            X_resampled, y_resampled = X_train_scaled, y_train
        del X_train_scaled, y_train
        gc.collect()
        
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', bootstrap=True,
            random_state=42, n_jobs=-1, class_weight='balanced'
        )
        model.fit(X_resampled, y_resampled)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(model, model_dir / f"blackwall_model_{timestamp}.pkl", compress=3)
        joblib.dump(scaler, model_dir / "blackwall_scaler.pkl", compress=3)
        
        logger.info("BlackWall AI Model Training Completed Successfully")
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
    finally:
        gc.collect()

if __name__ == "__main__":
    main()
