import pandas as pd

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

# ---------- SAVE THE FILE ----------
print("\n" + "-"*40)
print("SAVING SELECTED DATA")
print("-"*40)

df_selected.to_csv('selected_data.csv', index=False)
print("[SUCCESS] File saved as 'selected_data.csv'")
print(f"[SUCCESS] Saved {df_selected.shape[0]} rows and {df_selected.shape[1]} columns")
