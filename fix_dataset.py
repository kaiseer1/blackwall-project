import pandas as pd
import numpy as np

# Load the cleaned dataset
print("Loading dataset...")
df = pd.read_csv('datasets/Sampled_Dataset_Example_cleaned.csv')

# Find which column contains 'BENIGN' values
for col in df.columns:
    try:
        unique_values = df[col].astype(str).unique()
        if 'BENIGN' in unique_values:
            print(f"Found 'BENIGN' in column: '{col}'")
            # This is our label column
            label_col = col
            break
    except:
        continue

# Verify what we found
print(f"Label column: '{label_col}'")
print(f"Unique values in label column: {df[label_col].unique()}")

# Move the label column to be named 'Label'
df_fixed = df.drop(columns=[label_col])
df_fixed['Label'] = df[label_col]

# Save the modified dataset
print("Saving modified dataset...")
df_fixed.to_csv('datasets/Sampled_Dataset_Example_fixed.csv', index=False)
print("Done!")