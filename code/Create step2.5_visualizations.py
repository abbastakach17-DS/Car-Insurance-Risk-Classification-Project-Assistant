#Plot one or two relationships (e.g., Past Accidents vs Driving experience, speeding violations vs outcome...). 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r'C:\Users\Admin\Desktop\data_analysis_project\selected_data.csv')

# Convert AGE to numeric for plotting
df['AGE_NUMERIC'] = df['AGE'].str.extract(r'(\d+)').astype(float)

# ---------- PLOT 1: Past Accidents vs Driving Experience ----------
print("\n" + "-"*40)
print("1. Past Accidents vs Driving Experience")
print("-"*40)

plt.figure(figsize=(10, 6))

# Create scatter plot colored by OUTCOME
scatter = plt.scatter(
    df['DRIVING_EXPERIENCE'], 
    df['PAST_ACCIDENTS'],
    c=df['OUTCOME'],
    cmap='RdYlGn_r',
    alpha=0.6,
    edgecolors='black',
    linewidth=0.5
)

plt.xlabel('Driving Experience (years)', fontsize=12)
plt.ylabel('Past Accidents', fontsize=12)
plt.title('Past Accidents vs Driving Experience (colored by Risk)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Risk Level (0=Low, 1=High)')
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('past_accidents_vs_experience.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Plot saved as 'past_accidents_vs_experience.png'")

plt.show()

# ---------- PLOT 2: Speeding Violations vs Outcome ----------
print("\n" + "-"*40)
print("2. Speeding Violations vs Outcome")
print("-"*40)

plt.figure(figsize=(10, 6))

# Create a boxplot
df.boxplot(column='SPEEDING_VIOLATIONS', by='OUTCOME', grid=True)
plt.xlabel('Risk Level (0=Low, 1=High)', fontsize=12)
plt.ylabel('Speeding Violations', fontsize=12)
plt.title('Speeding Violations by Risk Level', fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove automatic title

# Add mean line
mean_speeding = df['SPEEDING_VIOLATIONS'].mean()
plt.axhline(y=mean_speeding, color='red', linestyle='--', 
           label=f'Overall Mean: {mean_speeding:.2f}')
plt.legend()

# Save the plot
plt.savefig('speeding_vs_outcome.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Plot saved as 'speeding_vs_outcome.png'")

plt.show()

# ---------- PLOT 3: Additional - Age vs Speeding Violations ----------
print("\n" + "-"*40)
print("3. Additional: Age vs Speeding Violations")
print("-"*40)

plt.figure(figsize=(10, 6))

# Create scatter plot
for risk in [0, 1]:
    subset = df[df['OUTCOME'] == risk]
    plt.scatter(
        subset['AGE_NUMERIC'], 
        subset['SPEEDING_VIOLATIONS'],
        label=f'Risk {risk}',
        alpha=0.5,
        s=30
    )

plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Speeding Violations', fontsize=12)
plt.title('Age vs Speeding Violations by Risk Level', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('age_vs_speeding.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Plot saved as 'age_vs_speeding.png'")

plt.show()

# ---------- PLOT 4: Additional - Correlation Heatmap ----------
print("\n" + "-"*40)
print("4. Correlation Heatmap")
print("-"*40)

plt.figure(figsize=(10, 8))

# Select numeric columns
numeric_cols = ['AGE_NUMERIC', 'ANNUAL_MILEAGE', 'PAST_ACCIDENTS', 
                'DUIS', 'SPEEDING_VIOLATIONS', 'OUTCOME']

correlation = df[numeric_cols].corr()

# Create heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', 
            square=True, linewidths=0.5)

plt.title('Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold')

# Save the plot
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Plot saved as 'correlation_heatmap.png'")

plt.show()

print("\n" + "="*60)
print("STEP 2.5 COMPLETE! All visualizations saved.")
print("="*60)