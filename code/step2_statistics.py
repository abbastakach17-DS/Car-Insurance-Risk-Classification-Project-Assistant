
import pandas as pd

# Load the data we saved from Step 1
df = pd.read_csv(r"C:\Users\Admin\Desktop\data_analysis_project\selected_data.csv")

print("Data loaded successfully!")
print(f"Shape: {df.shape}")

# Check for missing values
print("\n" + "-"*40)
print("MISSING VALUES:")
print("-"*40)
print(df.isnull().sum())

# Convert AGE to numeric (extract the number)
# AGE is like "65+", "16-25", "26-39"
# We'll take the first number
df['AGE_NUMERIC'] = df['AGE'].str.extract(r'(\d+)').astype(float)

# Convert DRIVING_EXPERIENCE to numeric
exp_map = {
    '0-9y': 4.5,
    '10-19y': 14.5, 
    '20-29y': 24.5,
    '30y+': 35
}
df['EXP_NUMERIC'] = df['DRIVING_EXPERIENCE'].map(exp_map)

# Compute statistics
print("\n" + "-"*40)
print("REQUIRED STATISTICS (from professor):")
print("-"*40)

print(f"Mean Age: {df['AGE_NUMERIC'].mean():.2f} years")
print(f"Mean Annual Mileage: {df['ANNUAL_MILEAGE'].mean():.2f}")

print("\n" + "-"*40)
print("ADDITIONAL STATISTICS:")
print("-"*40)

print(f"Average Past Accidents: {df['PAST_ACCIDENTS'].mean():.2f}")
print(f"Average Speeding Violations: {df['SPEEDING_VIOLATIONS'].mean():.2f}")

# Target variable distribution
print("\n" + "-"*40)
print("TARGET VARIABLE (OUTCOME) DISTRIBUTION:")
print("-"*40)
print(df['OUTCOME'].value_counts())

risk_percentage = df['OUTCOME'].mean() * 100
print(f"\nHigh Risk (1): {risk_percentage:.2f}%")
print(f"Low Risk (0): {100 - risk_percentage:.2f}%")

# Save the statistics to a file
print("\nStatistics computed successfully!")