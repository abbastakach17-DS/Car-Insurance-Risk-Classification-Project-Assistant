import pandas as pd
# Load the data
df = pd.read_csv(r'C:\Users\Admin\Desktop\data_analysis_project\Car_Insurance_Claim.csv\Car_Insurance_Claim.csv')

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
print(rf"\n✅ Filled missing ANNUAL_MILEAGE with median: {mileage_median}")

# FIX 2: Drop CREDIT_SCORE (we don't need it, and it has many missing values)
df = df.drop('CREDIT_SCORE', axis=1)
print(rf"✅ Dropped CREDIT_SCORE column (has many missing values)")

# ---------- PART 2: KEEP ONLY RELEVANT COLUMNS ----------
print("\n" + "-"*40)
print("2. SELECTING RELEVANT COLUMNS")
print("-"*40)

# Keep only the columns we need
relevant_columns = [
    'AGE', 'DRIVING_EXPERIENCE', 'VEHICLE_TYPE', 
    'ANNUAL_MILEAGE', 'PAST_ACCIDENTS', 'DUIS', 
    'SPEEDING_VIOLATIONS', 'OUTCOME'
]

df = df[relevant_columns]
print(rf"✅ Kept {len(relevant_columns)} columns: {relevant_columns}")

# ---------- PART 3: CONVERT TEXT TO NUMBERS ----------
print("\n" + "-"*40)
print("3. CONVERTING CATEGORICAL DATA TO NUMBERS")
print("-"*40)

# Convert AGE (e.g., "65+" → 65)
df['AGE_NUMERIC'] = df['AGE'].str.extract(r'(\d+)').astype(float)
print("✅ Converted AGE to numbers")

# Convert DRIVING_EXPERIENCE (e.g., "0-9y" → 4.5)
exp_map = {
    '0-9y': 4.5,
    '10-19y': 14.5, 
    '20-29y': 24.5,
    '30y+': 35
}
df['EXP_NUMERIC'] = df['DRIVING_EXPERIENCE'].map(exp_map)
print("✅ Converted DRIVING_EXPERIENCE to numbers")

# Convert VEHICLE_TYPE (e.g., "sedan" → 0)
vehicle_map = {
    'sedan': 0,
    'suv': 1,
    'truck': 2,
    'sports': 3,
    'van': 4
}
df['VEHICLE_TYPE_NUMERIC'] = df['VEHICLE_TYPE'].map(vehicle_map)
print("✅ Converted VEHICLE_TYPE to numbers")

# ---------- PART 4: VERIFY CLEANING ----------
print("\n" + "-"*40)
print("4. VERIFY CLEANING")
print("-"*40)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print(f"\nShape after cleaning: {df.shape}")

# Show data types
print("\nData types after cleaning:")
print(df.dtypes)

# ---------- PART 5: SAVE CLEANED DATA ----------
print("\n" + "-"*40)
print("5. SAVING CLEANED DATA")
print("-"*40)

df.to_csv('cleaned_data.csv', index=False)
print("✅ Cleaned data saved to 'cleaned_data.csv'")

# Show first few rows
print("\nFirst 5 rows of cleaned data:")
print(df.head())

print("\n" + "="*60)
print("✅ STEP 3 COMPLETE! Data is now clean!")
print("="*60)