import pandas as pd

# Load the data from Step 1
df = pd.read_csv('selected_data.csv')

print(f"Original shape: {df.shape}")

# ---------- PART 1: HANDLE MISSING VALUES ----------
print("\n" + "-"*40)
print("1. HANDLING MISSING VALUES")
print("-"*40)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# FIX 1: Fill missing ANNUAL_MILEAGE with median
mileage_median = df['ANNUAL_MILEAGE'].median()
df['ANNUAL_MILEAGE'] = df['ANNUAL_MILEAGE'].fillna(mileage_median)
print(f"\n[SUCCESS] Filled missing ANNUAL_MILEAGE with median: {mileage_median}")

# FIX 2: Fill missing VEHICLE_TYPE with the most common type
mode_vehicle = df['VEHICLE_TYPE'].mode()[0]
df['VEHICLE_TYPE'] = df['VEHICLE_TYPE'].fillna(mode_vehicle)
print(f"[SUCCESS] Filled missing VEHICLE_TYPE with mode: {mode_vehicle}")

# ---------- PART 2: CONVERT TEXT TO NUMBERS ----------
print("\n" + "-"*40)
print("2. CONVERTING CATEGORICAL DATA TO NUMBERS")
print("-"*40)

# Convert AGE (e.g., "65+" → 65)
df['AGE_NUMERIC'] = df['AGE'].str.extract(r'(\d+)').astype(float)
print("Converted AGE to numbers")

# Convert DRIVING_EXPERIENCE (e.g., "0-9y" → 4.5)
exp_map = {
    '0-9y': 4.5,
    '10-19y': 14.5, 
    '20-29y': 24.5,
    '30y+': 35
}
df['EXP_NUMERIC'] = df['DRIVING_EXPERIENCE'].map(exp_map)
print("Converted DRIVING_EXPERIENCE to numbers")

# Convert VEHICLE_TYPE (e.g., "sedan" → 0)
vehicle_map = {
    'sedan': 0,
    'suv': 1,
    'truck': 2,
    'sports car': 3,
    'van': 4
}
df['VEHICLE_TYPE_NUMERIC'] = df['VEHICLE_TYPE'].map(vehicle_map)
print("Converted VEHICLE_TYPE to numbers")

# ---------- PART 3: CREATE CLEAN DATASET ----------
print("\n" + "-"*40)
print("3. CREATING FINAL CLEAN DATASET")
print("-"*40)

# Keep only the columns we need for modeling
clean_columns = [
    'AGE_NUMERIC',
    'EXP_NUMERIC',
    'VEHICLE_TYPE_NUMERIC',
    'ANNUAL_MILEAGE',
    'PAST_ACCIDENTS',
    'DUIS',
    'SPEEDING_VIOLATIONS',
    'OUTCOME'
]

df_clean = df[clean_columns].copy()

# Rename columns for clarity
df_clean.columns = [
    'Age',
    'Driving_Experience',
    'Vehicle_Type',
    'Annual_Mileage',
    'Past_Accidents',
    'DUIS',
    'Speeding_Violations',
    'Outcome'
]

print(f"Clean dataset shape: {df_clean.shape}")
print(f"Columns: {df_clean.columns.tolist()}")

# ---------- PART 4: VERIFY CLEANING ----------
print("\n" + "-"*40)
print("4. VERIFY CLEANING")
print("-"*40)

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# Check if any missing values remain
if df_clean.isnull().sum().sum() == 0:
    print("\n[SUCCESS] No missing values remain!")
else:
    print("\n[WARNING] Some missing values still exist!")

print("\nData types:")
print(df_clean.dtypes)

print("\nFirst 5 rows of cleaned data:")
print(df_clean.head())

# ---------- PART 5: SAVE CLEANED DATA ----------
print("\n" + "-"*40)
print("5. SAVING CLEANED DATA")
print("-"*40)

df_clean.to_csv('cleaned_data.csv', index=False)
print("[SUCCESS] Cleaned data saved to 'cleaned_data.csv'")
print(f"[SUCCESS] Saved {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")

print("\n" + "="*60)
print("STEP 3 COMPLETE! Data is now clean and ready for modeling!")
print("="*60)