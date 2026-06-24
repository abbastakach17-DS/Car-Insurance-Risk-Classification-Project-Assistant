# step6_decision_tree.py
# STEP 6: Train and evaluate Decision Tree model

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
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
import pickle


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

# ---------- HYPERPARAMETER TUNING ----------
print("\n" + "-"*40)
print("2. HYPERPARAMETER TUNING")
print("-"*40)

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}

print("Searching for best parameters...")
print("This may take a moment...")

# Grid search with cross-validation
dt_model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    dt_model, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\nBest parameters: {best_params}")
print(f"Best cross-validation score: {best_score:.4f}")

# Use the best model
dt_best = grid_search.best_estimator_

# ---------- TRAIN DECISION TREE ----------
print("\n" + "-"*40)
print("3. TRAINING DECISION TREE")
print("-"*40)

# Train the model (already trained in grid search)
print("Decision Tree model trained successfully!")
print(f"Tree depth: {dt_best.get_depth()}")
print(f"Number of leaves: {dt_best.get_n_leaves()}")

# ---------- MAKE PREDICTIONS ----------
print("\n" + "-"*40)
print("4. MAKING PREDICTIONS")
print("-"*40)

y_pred = dt_best.predict(X_test)
y_pred_proba = dt_best.predict_proba(X_test)[:, 1]

print("Predictions complete!")

# ---------- FEATURE IMPORTANCE ----------
print("\n" + "-"*40)
print("5. FEATURE IMPORTANCE")
print("-"*40)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': dt_best.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features:")
print(feature_importance.head(5))

print("\nAll feature importances:")
print(feature_importance)

# ---------- EVALUATE MODEL ----------
print("\n" + "-"*40)
print("6. EVALUATING MODEL")
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
print("7. CONFUSION MATRIX")
print("-"*40)

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:")
print(f"TN: {cm[0,0]:>5}  FP: {cm[0,1]:>5}")
print(f"FN: {cm[1,0]:>5}  TP: {cm[1,1]:>5}")

# ---------- CLASSIFICATION REPORT ----------
print("\n" + "-"*40)
print("8. CLASSIFICATION REPORT")
print("-"*40)

print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

# ---------- VISUALIZATIONS ----------
print("\n" + "-"*40)
print("9. CREATING VISUALIZATIONS")
print("-"*40)

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
plt.title('Decision Tree - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('dt_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved confusion matrix as 'dt_confusion_matrix.png'")
plt.show()

# 2. ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='green', lw=2, label=f'Decision Tree (AUC = {auc_roc:.3f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Decision Tree - ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dt_roc_curve.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved ROC curve as 'dt_roc_curve.png'")
plt.show()

# 3. Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], 
         color='green', edgecolor='black')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Decision Tree - Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('dt_feature_importance.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved feature importance as 'dt_feature_importance.png'")
plt.show()

# 4. Decision Tree Visualization (if tree is small enough)
if dt_best.get_depth() <= 5:
    plt.figure(figsize=(20, 10))
    plot_tree(dt_best, 
              feature_names=X_train.columns,
              class_names=['Low Risk', 'High Risk'],
              filled=True, 
              rounded=True,
              fontsize=10,
              max_depth=3)
    plt.title('Decision Tree Structure (First 3 Levels)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dt_tree_structure.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Saved tree structure as 'dt_tree_structure.png'")
    plt.show()
else:
    print(f"\n[INFO] Tree depth ({dt_best.get_depth()}) > 5, skipping tree visualization")
    print("Tree is too large to display clearly")

# ---------- SUMMARY ----------
print("\n" + "="*60)
print("DECISION TREE MODEL COMPLETE!")
print("="*60)

print("\nModel Performance Summary:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc_roc:.4f}")

# Save results
results = {
    'model': 'Decision Tree',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'auc_roc': auc_roc,
    'confusion_matrix': cm,
    'feature_importance': feature_importance.to_dict(),
    'best_params': best_params
}

with open('dt_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\n[SUCCESS] Results saved to 'dt_results.pkl'")

print("\nNext Step: Step 7 - Compare Models")