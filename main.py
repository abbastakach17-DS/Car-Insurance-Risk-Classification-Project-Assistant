import pandas as pd
import numpy as np          # <-- Add this
import matplotlib.pyplot as plt # <-- Add this
import seaborn as sns
import os
import sklearn
# =============================================================================
# STEP 1: LOAD DATASET AND SELECT RELEVANT COLUMNS
# =============================================================================

print("\n" + "="*60)
print("STEP 1: LOADING DATA")
print("="*60)

# Load the dataset
df = pd.read_csv(r'C:\Users\Admin\Desktop\data_analysis_project\Car_Insurance_Claim.csv\Car_Insurance_Claim.csv')

# Show what we have
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Select only the columns the professor wants
relevant_columns = [
    'AGE',
    'DRIVING_EXPERIENCE',
    'VEHICLE_TYPE',
    'ANNUAL_MILEAGE',
    'PAST_ACCIDENTS',
    'DUIS',
    'SPEEDING_VIOLATIONS',
    'OUTCOME'
]

# Keep only columns that exist
available_cols = [col for col in relevant_columns if col in df.columns]
df_selected = df[available_cols].copy()

print(f"\nSelected {len(available_cols)} columns:")
print(df_selected.columns.tolist())
print(f"\nFirst 5 rows:")
print(df_selected.head())