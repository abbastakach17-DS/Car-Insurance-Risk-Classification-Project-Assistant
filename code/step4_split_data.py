# step4_split_data.py
# STEP 4: Split data into training and testing sets

import pandas as pd
from sklearn.model_selection import train_test_split



# Load the cleaned data from Step 3
df = pd.read_csv(r'C:\Users\Admin\Desktop\data_analysis_project\cleaned_data.csv')

print(f"Loaded cleaned data: {df.shape}")

# ---------- SEPARATE FEATURES AND TARGET ----------
print("\n" + "-"*40)
print("1. SEPARATING FEATURES AND TARGET")
print("-"*40)

# Define features (X) and target (y)
# X = All columns except 'Outcome'
X = df.drop('Outcome', axis=1)

# y = The target column 'Outcome'
y = df['Outcome']

print(f"Features (X) shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")
print(f"Target (y) shape: {y.shape}")

# ---------- SPLIT THE DATA ----------
print("\n" + "-"*40)
print("2. SPLITTING INTO TRAINING AND TEST SETS")
print("-"*40)

# Split 75% training, 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,          # 25% for testing
    random_state=42,         # For reproducibility
    stratify=y              # Maintain class distribution
)

print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ---------- CHECK CLASS DISTRIBUTION ----------
print("\n" + "-"*40)
print("3. CHECKING CLASS DISTRIBUTION")
print("-"*40)

print("\nOriginal Data:")
print(df['Outcome'].value_counts())
print(f"High Risk: {df['Outcome'].mean()*100:.2f}%")

print("\nTraining Set:")
print(y_train.value_counts())
print(f"High Risk: {y_train.mean()*100:.2f}%")

print("\nTest Set:")
print(y_test.value_counts())
print(f"High Risk: {y_test.mean()*100:.2f}%")

# ---------- VERIFY SPLIT ----------
print("\n" + "-"*40)
print("4. VERIFYING SPLIT")
print("-"*40)

print(f"Original: {len(X)} samples")
print(f"Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"Total:    {len(X_train) + len(X_test)} samples")

# Check if class distribution is maintained
print(f"\nTraining High Risk %: {y_train.mean()*100:.2f}%")
print(f"Test High Risk %: {y_test.mean()*100:.2f}%")
print(f"Original High Risk %: {df['Outcome'].mean()*100:.2f}%")

# ---------- SAVE THE SPLIT DATA ----------
print("\n" + "-"*40)
print("5. SAVING SPLIT DATA")
print("-"*40)

# Save each set to a CSV file
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("[SUCCESS] Saved X_train.csv")
print("[SUCCESS] Saved X_test.csv")
print("[SUCCESS] Saved y_train.csv")
print("[SUCCESS] Saved y_test.csv")

