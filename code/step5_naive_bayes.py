# step5_naive_bayes.py
# STEP 5: Train and evaluate Naive Bayes model
#1-Load the training data
#2-Train a Naive Bayes classifier
#3-Make predictions on the test data
#4-Evaluate performance (accuracy, precision, recall, F1)
#5-Create a confusion matrix

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- LOAD THE SPLIT DATA ----------
print("\n" + "-"*40)
print("1. LOADING DATA")
print("-"*40)

X_train = pd.read_csv(r'C:\Users\Admin\Desktop\data_analysis_project\X_train.csv')
X_test = pd.read_csv(r'C:\Users\Admin\Desktop\data_analysis_project\X_test.csv')
y_train = pd.read_csv(r'C:\Users\Admin\Desktop\data_analysis_project\y_train.csv').values.ravel()
y_test = pd.read_csv(r'C:\Users\Admin\Desktop\data_analysis_project\y_test.csv').values.ravel()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# ---------- SCALE THE DATA ----------
print("\n" + "-"*40)
print("2. SCALING FEATURES")
print("-"*40)

# Naive Bayes works better with scaled data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled to have mean=0 and std=1")

# ---------- TRAIN NAIVE BAYES ----------
print("\n" + "-"*40)
print("3. TRAINING NAIVE BAYES")
print("-"*40)

# Create and train the model
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

print("Naive Bayes model trained successfully!")

# ---------- MAKE PREDICTIONS ----------
print("\n" + "-"*40)
print("4. MAKING PREDICTIONS")
print("-"*40)

# Predict on test data
y_pred = nb_model.predict(X_test_scaled)
y_pred_proba = nb_model.predict_proba(X_test_scaled)[:, 1]

print("Predictions complete!")

# ---------- EVALUATE MODEL ----------
print("\n" + "-"*40)
print("5. EVALUATING MODEL")
print("-"*40)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")

# ---------- CONFUSION MATRIX ----------
print("\n" + "-"*40)
print("6. CONFUSION MATRIX")
print("-"*40)

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:")
print(f"TN: {cm[0,0]:>5}  FP: {cm[0,1]:>5}")
print(f"FN: {cm[1,0]:>5}  TP: {cm[1,1]:>5}")

# ---------- CLASSIFICATION REPORT ----------
print("\n" + "-"*40)
print("7. CLASSIFICATION REPORT")
print("-"*40)

print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

# ---------- VISUALIZATIONS ----------
print("\n" + "-"*40)
print("8. CREATING VISUALIZATIONS")
print("-"*40)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
plt.title('Naive Bayes - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('nb_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved confusion matrix as 'nb_confusion_matrix.png'")
plt.show()

# Plot ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Naive Bayes (AUC = {auc_roc:.3f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Naive Bayes - ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('nb_roc_curve.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved ROC curve as 'nb_roc_curve.png'")
plt.show()

# ---------- SUMMARY ----------
print("\n" + "="*60)
print("NAIVE BAYES MODEL COMPLETE!")
print("="*60)

print("\nModel Performance Summary:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc_roc:.4f}")

# Save results
results = {
    'model': 'Naive Bayes',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'auc_roc': auc_roc,
    'confusion_matrix': cm
}

# Save to file for later comparison
import pickle
with open('nb_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\n[SUCCESS] Results saved to 'nb_results.pkl'")

print("\nNext Step: Step 6 - Decision Tree Model")