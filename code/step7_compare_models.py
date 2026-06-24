# step7_compare_models_fixed.py
# STEP 7: Model Comparison & Conclusions (FIXED)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STEP 7: MODEL COMPARISON & CONCLUSIONS")
print("="*60)

# ---------- LOAD RESULTS ----------
print("\n" + "-"*40)
print("1. LOADING MODEL RESULTS")
print("-"*40)

# Load Naive Bayes results
with open(r'C:\Users\Admin\Desktop\data_analysis_project\Naïve Bayes\nb_results.pkl', 'rb') as f:
    nb_results = pickle.load(f)

# Load Decision Tree results
with open(r'C:\Users\Admin\Desktop\data_analysis_project\Decision Tree\dt_results.pkl', 'rb') as f:
    dt_results = pickle.load(f)

print("Naive Bayes results loaded:")
print(f"  Accuracy: {nb_results['accuracy']:.4f}")
print(f"  Precision: {nb_results['precision']:.4f}")
print(f"  Recall: {nb_results['recall']:.4f}")
print(f"  F1-Score: {nb_results['f1_score']:.4f}")
print(f"  AUC-ROC: {nb_results['auc_roc']:.4f}")

print("\nDecision Tree results loaded:")
print(f"  Accuracy: {dt_results['accuracy']:.4f}")
print(f"  Precision: {dt_results['precision']:.4f}")
print(f"  Recall: {dt_results['recall']:.4f}")
print(f"  F1-Score: {dt_results['f1_score']:.4f}")
print(f"  AUC-ROC: {dt_results['auc_roc']:.4f}")

# ---------- CREATE COMPARISON TABLE ----------
print("\n" + "-"*40)
print("2. MODEL COMPARISON TABLE")
print("-"*40)

comparison_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Naive Bayes': [
        nb_results['accuracy'],
        nb_results['precision'],
        nb_results['recall'],
        nb_results['f1_score'],
        nb_results['auc_roc']
    ],
    'Decision Tree': [
        dt_results['accuracy'],
        dt_results['precision'],
        dt_results['recall'],
        dt_results['f1_score'],
        dt_results['auc_roc']
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Difference'] = comparison_df['Decision Tree'] - comparison_df['Naive Bayes']

print("\nModel Performance Comparison:")
print(comparison_df.to_string(index=False))

# ---------- DETERMINE BEST MODEL ----------
print("\n" + "-"*40)
print("3. DETERMINING BEST MODEL")
print("-"*40)

best_accuracy = "Decision Tree" if dt_results['accuracy'] > nb_results['accuracy'] else "Naive Bayes"
best_precision = "Decision Tree" if dt_results['precision'] > nb_results['precision'] else "Naive Bayes"
best_recall = "Naive Bayes" if nb_results['recall'] > dt_results['recall'] else "Decision Tree"
best_f1 = "Naive Bayes" if nb_results['f1_score'] > dt_results['f1_score'] else "Decision Tree"
best_auc = "Decision Tree" if dt_results['auc_roc'] > nb_results['auc_roc'] else "Naive Bayes"

print(f"Best Accuracy:  {best_accuracy}")
print(f"Best Precision: {best_precision}")
print(f"Best Recall:    {best_recall}")
print(f"Best F1-Score:  {best_f1}")
print(f"Best AUC-ROC:   {best_auc}")

# ---------- CREATE VISUALIZATIONS ----------
print("\n" + "-"*40)
print("4. CREATING COMPARISON VISUALIZATIONS")
print("-"*40)

# Set style and figure size for better resolution
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# FIGURE 1: METRICS COMPARISON
fig1 = plt.figure(figsize=(12, 8))
ax1 = plt.subplot(111)

metrics = comparison_df['Metric']
nb_values = comparison_df['Naive Bayes']
dt_values = comparison_df['Decision Tree']
x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, nb_values, width, label='Naive Bayes', color='cornflowerblue', edgecolor='black')
bars2 = ax1.bar(x + width/2, dt_values, width, label='Decision Tree', color='lightgreen', edgecolor='black')

ax1.set_xlabel('Metrics', fontsize=14)
ax1.set_ylabel('Score', fontsize=14)
ax1.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=12)
ax1.legend(fontsize=12)
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison_metrics.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved metrics comparison as 'model_comparison_metrics.png'")
plt.show()

# FIGURE 2: CONFUSION MATRICES SIDE BY SIDE
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6))

# Naive Bayes Confusion Matrix
cm_nb = nb_results['confusion_matrix']
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'],
            annot_kws={'size': 16})
ax2.set_title(f'Naive Bayes\nAccuracy: {nb_results["accuracy"]:.3f}', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted', fontsize=12)
ax2.set_ylabel('Actual', fontsize=12)

# Decision Tree Confusion Matrix
cm_dt = dt_results['confusion_matrix']
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=ax3,
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'],
            annot_kws={'size': 16})
ax3.set_title(f'Decision Tree\nAccuracy: {dt_results["accuracy"]:.3f}', fontsize=14, fontweight='bold')
ax3.set_xlabel('Predicted', fontsize=12)
ax3.set_ylabel('Actual', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved confusion matrices as 'confusion_matrices_comparison.png'")
plt.show()

# FIGURE 3: FEATURE IMPORTANCE
if 'feature_importance' in dt_results:
    fig3, ax4 = plt.subplots(figsize=(10, 6))
    
    importance_dict = dt_results['feature_importance']
    features = list(importance_dict['feature'].values())
    importance = list(importance_dict['importance'].values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    features_sorted = [features[i] for i in sorted_idx]
    importance_sorted = [importance[i] for i in sorted_idx]
    
    bars = ax4.barh(features_sorted, importance_sorted, color='green', edgecolor='black')
    ax4.set_xlabel('Importance', fontsize=14)
    ax4.set_ylabel('Feature', fontsize=14)
    ax4.set_title('Decision Tree - Feature Importance', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, v) in enumerate(zip(bars, importance_sorted)):
        ax4.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Saved feature importance as 'feature_importance.png'")
    plt.show()

# FIGURE 4: SUMMARY TABLE
fig4, ax5 = plt.subplots(figsize=(10, 4))
ax5.axis('tight')
ax5.axis('off')

# Prepare table data
table_data = [
    ['Metric', 'Naive Bayes', 'Decision Tree', 'Best Model'],
    ['Accuracy', f'{nb_results["accuracy"]:.3f}', f'{dt_results["accuracy"]:.3f}', best_accuracy],
    ['Precision', f'{nb_results["precision"]:.3f}', f'{dt_results["precision"]:.3f}', best_precision],
    ['Recall', f'{nb_results["recall"]:.3f}', f'{dt_results["recall"]:.3f}', best_recall],
    ['F1-Score', f'{nb_results["f1_score"]:.3f}', f'{dt_results["f1_score"]:.3f}', best_f1],
    ['AUC-ROC', f'{nb_results["auc_roc"]:.3f}', f'{dt_results["auc_roc"]:.3f}', best_auc]
]

table = ax5.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Color the header
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('model_summary_table.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved summary table as 'model_summary_table.png'")
plt.show()

# ---------- GENERATE FINAL REPORT (WITHOUT UNICODE) ----------
print("\n" + "-"*40)
print("5. GENERATING FINAL REPORT")
print("-"*40)

# Use simple ASCII characters instead of Unicode
report = """
================================================================================
                    CAR INSURANCE RISK CLASSIFICATION
                          FINAL PROJECT REPORT
================================================================================

1. PROJECT OVERVIEW
================================================================================
This project developed machine learning models to classify car insurance 
applicants as High Risk or Low Risk based on their driving history and 
demographic information. The goal was to help insurance companies identify 
the best clients for coverage.

2. DATASET SUMMARY
================================================================================
- Total samples: 10,000
- Features used: 7 (Age, Driving Experience, Vehicle Type, Annual Mileage, 
                 Past Accidents, DUIS, Speeding Violations)
- Target distribution: 68.67% Low Risk, 31.33% High Risk
- Data split: 75% training (7,500 samples), 25% testing (2,500 samples)

3. MODELS IMPLEMENTED
================================================================================
Two classification models were trained and evaluated:
1. Naive Bayes - Probabilistic classifier assuming feature independence
2. Decision Tree - Tree-based classifier with hyperparameter tuning

4. PERFORMANCE COMPARISON
================================================================================
"""
report += comparison_df.to_string(index=False)
report += f"""

5. KEY FINDINGS
================================================================================
"""

if nb_results['recall'] > dt_results['recall']:
    report += f"[OK] Naive Bayes achieved higher RECALL ({nb_results['recall']:.3f} vs {dt_results['recall']:.3f})\n"
    report += f"     -> Catches {nb_results['recall']*100:.1f}% of High Risk cases\n"
else:
    report += f"[OK] Decision Tree achieved higher RECALL ({dt_results['recall']:.3f} vs {nb_results['recall']:.3f})\n"

if dt_results['precision'] > nb_results['precision']:
    report += f"[OK] Decision Tree achieved higher PRECISION ({dt_results['precision']:.3f} vs {nb_results['precision']:.3f})\n"
    report += f"     -> {dt_results['precision']*100:.1f}% of High Risk predictions are correct\n"
else:
    report += f"[OK] Naive Bayes achieved higher PRECISION ({nb_results['precision']:.3f} vs {dt_results['precision']:.3f})\n"

report += f"""
6. FEATURE IMPORTANCE (Decision Tree)
================================================================================
"""
if 'feature_importance' in dt_results:
    importance_dict = dt_results['feature_importance']
    features = list(importance_dict['feature'].values())
    importance = list(importance_dict['importance'].values())
    
    for f, imp in zip(features[:5], importance[:5]):
        report += f"  - {f}: {imp*100:.1f}%\n"

report += f"""
7. CONFUSION MATRIX COMPARISON
================================================================================
Naive Bayes Confusion Matrix:
  Low Risk:  {cm_nb[0,0]}   High Risk: {cm_nb[0,1]}
  Low Risk:  {cm_nb[1,0]}   High Risk: {cm_nb[1,1]}

Decision Tree Confusion Matrix:
  Low Risk:  {cm_dt[0,0]}   High Risk: {cm_dt[0,1]}
  Low Risk:  {cm_dt[1,0]}   High Risk: {cm_dt[1,1]}

8. BUSINESS RECOMMENDATIONS
================================================================================
1. PRIMARY SCREENING: Use Naive Bayes to catch most High Risk cases
   - Recall: {nb_results['recall']*100:.1f}% of High Risk identified
   - Good for initial screening of all applicants

2. FINAL DECISIONS: Use Decision Tree for final approval
   - Precision: {dt_results['precision']*100:.1f}% accuracy in High Risk flags
   - More interpretable for explaining decisions

3. ENSEMBLE APPROACH: Combine both models
   - Use Naive Bayes for high recall + Decision Tree for high precision
   - Only approve applications where both models agree

9. CONCLUSIONS
================================================================================
This project successfully developed two machine learning models for car 
insurance risk classification. The Decision Tree model achieved higher 
overall accuracy ({dt_results['accuracy']:.3f}) while the Naive Bayes model 
demonstrated better recall ({nb_results['recall']:.3f}) for identifying 
High Risk cases.

For the insurance company, the best approach depends on priorities:
- If catching High Risk cases is critical -> Use Naive Bayes
- If minimizing false positives is more important -> Use Decision Tree
- For best overall performance -> Use an ensemble of both

The models can help insurance companies make data-driven decisions about
premiums and coverage, reducing financial losses from high-risk clients.

================================================================================
                    PROJECT COMPLETED SUCCESSFULLY!
================================================================================
"""

# Print report without Unicode errors
print(report)

# Save report to file
with open('final_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\n[SUCCESS] Final report saved as 'final_report.txt'")

# ---------- FINAL SUMMARY ----------
print("\n" + "="*60)
print("PROJECT COMPLETE!")
print("="*60)

print("\nFiles Created:")
print("  - selected_data.csv (Step 1)")
print("  - cleaned_data.csv (Step 3)")
print("  - X_train.csv, X_test.csv, y_train.csv, y_test.csv (Step 4)")
print("  - nb_confusion_matrix.png, nb_roc_curve.png (Step 5)")
print("  - dt_confusion_matrix.png, dt_roc_curve.png, dt_feature_importance.png (Step 6)")
print("  - model_comparison_metrics.png (Step 7)")
print("  - confusion_matrices_comparison.png (Step 7)")
print("  - feature_importance.png (Step 7)")
print("  - model_summary_table.png (Step 7)")
print("  - final_report.txt (Step 7)")
print("  - nb_results.pkl, dt_results.pkl (Results saved)")

print("\n" + "="*60)
print("THANK YOU FOR COMPLETING THIS PROJECT!")
print("="*60)