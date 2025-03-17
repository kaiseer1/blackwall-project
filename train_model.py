import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Dataset files
dataset_files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
]

# Features & Labels
feature_columns = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'SYN Flag Count', 'ACK Flag Count'
]
label_column = 'Label'

# Load Data
df_list = []
for file in dataset_files:
    if os.path.exists(file):
        try:
            chunk = pd.read_csv(file, low_memory=False)
            df_list.append(chunk)
        except Exception as e:
            print(f"[!] Error reading {file}: {e}")

df = pd.concat(df_list, ignore_index=True) if df_list else None

if df is None:
    raise ValueError("No valid dataset files found!")

# ✅ Remove spaces from column names
df.columns = df.columns.str.strip()

# ✅ Ensure the label column exists
if label_column not in df.columns:
    raise ValueError(f"[!] Label column '{label_column}' not found in dataset!")

# ✅ Convert label to binary (1 = attack, 0 = normal)
df[label_column] = df[label_column].apply(lambda x: 1 if 'attack' in str(x).lower() else 0)

# ✅ Check Class Distribution
print("[+] Class Distribution:")
print(df[label_column].value_counts())

# ✅ Ensure at least 2 classes exist before applying SMOTE
if len(df[label_column].unique()) < 2:
    print("[!] Only one class detected. Using original dataset without SMOTE.")

    # ✅ Fix Invalid Values (handle NaNs and inf before training)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
    df.fillna(df.mean(), inplace=True)  # Replace NaN with column mean

    # ✅ Cap extreme values (avoiding overflow issues)
    df[feature_columns] = np.clip(df[feature_columns], -1e6, 1e6)

    # ✅ Scale Data
    scaler = StandardScaler()
    X_resampled = scaler.fit_transform(df[feature_columns])
    y_resampled = df[label_column].values

else:
    # ✅ Fix Invalid Values (handle NaNs and inf before training)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
    df.fillna(df.mean(), inplace=True)  # Replace NaN with column mean

    # ✅ Cap extreme values
    df[feature_columns] = np.clip(df[feature_columns], -1e6, 1e6)

    X = df[feature_columns]
    y = df[label_column]

    # ✅ Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ✅ Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# ✅ Final Check: Ensure X has valid numbers
if np.any(np.isnan(X_resampled)) or np.any(np.isinf(X_resampled)):
    print("[!] Warning: X contains invalid numbers. Replacing NaNs with 0.")
    X_resampled = np.nan_to_num(X_resampled)  # Replace NaN with 0

# ✅ Train Model
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_resampled, y_resampled)

# ✅ Save Model & Scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("[+] AI Model Trained and Saved Successfully!")
