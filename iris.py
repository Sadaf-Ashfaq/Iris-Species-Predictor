import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
df = sns.load_dataset('iris')

# PART 1: Data Loading and Inspection

print("DATASET SHAPE")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print()

print("COLUMN NAMES")
print(df.columns.tolist())
print()

print("FIRST FEW ROWS")
print(df.head())
print()

print("DATASET INFORMATION")
print(df.info())
print()

print("SUMMARY STATISTICS")
print(df.describe())
print()

# PART 2: Data Visualization

# Set the style for better looking plots
sns.set_style("whitegrid")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Scatter plots showing relationships between features
plt.subplot(2, 3, 1)
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', s=100, alpha=0.7)
plt.title('Sepal Length vs Sepal Width', fontsize=12, fontweight='bold')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

plt.subplot(2, 3, 2)
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', s=100, alpha=0.7)
plt.title('Petal Length vs Petal Width', fontsize=12, fontweight='bold')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

plt.subplot(2, 3, 3)
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species', s=100, alpha=0.7)
plt.title('Sepal Length vs Petal Length', fontsize=12, fontweight='bold')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')

# 2. Histograms showing value distributions
plt.subplot(2, 3, 4)
df['sepal_length'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length', fontsize=12, fontweight='bold')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')

plt.subplot(2, 3, 5)
df['petal_length'].hist(bins=20, color='lightcoral', edgecolor='black')
plt.title('Distribution of Petal Length', fontsize=12, fontweight='bold')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')

plt.subplot(2, 3, 6)
df['petal_width'].hist(bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Petal Width', fontsize=12, fontweight='bold')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('iris_visualizations_1.png', dpi=300, bbox_inches='tight')
print("Saved: iris_visualizations_1.png")
plt.show()

# 3. Box plots to identify outliers
fig2 = plt.figure(figsize=(14, 5))

plt.subplot(1, 4, 1)
sns.boxplot(y=df['sepal_length'], color='skyblue')
plt.title('Sepal Length Outliers', fontsize=12, fontweight='bold')
plt.ylabel('Sepal Length (cm)')

plt.subplot(1, 4, 2)
sns.boxplot(y=df['sepal_width'], color='lightcoral')
plt.title('Sepal Width Outliers', fontsize=12, fontweight='bold')
plt.ylabel('Sepal Width (cm)')

plt.subplot(1, 4, 3)
sns.boxplot(y=df['petal_length'], color='lightgreen')
plt.title('Petal Length Outliers', fontsize=12, fontweight='bold')
plt.ylabel('Petal Length (cm)')

plt.subplot(1, 4, 4)
sns.boxplot(y=df['petal_width'], color='plum')
plt.title('Petal Width Outliers', fontsize=12, fontweight='bold')
plt.ylabel('Petal Width (cm)')

plt.tight_layout()
plt.savefig('iris_boxplots.png', dpi=300, bbox_inches='tight')
print("Saved: iris_boxplots.png")
plt.show()

# Additional visualization: Box plots by species
fig3 = plt.figure(figsize=(14, 5))

plt.subplot(1, 4, 1)
sns.boxplot(data=df, x='species', y='sepal_length', palette='Set2')
plt.title('Sepal Length by Species', fontsize=12, fontweight='bold')
plt.ylabel('Sepal Length (cm)')
plt.xlabel('Species')

plt.subplot(1, 4, 2)
sns.boxplot(data=df, x='species', y='sepal_width', palette='Set2')
plt.title('Sepal Width by Species', fontsize=12, fontweight='bold')
plt.ylabel('Sepal Width (cm)')
plt.xlabel('Species')

plt.subplot(1, 4, 3)
sns.boxplot(data=df, x='species', y='petal_length', palette='Set2')
plt.title('Petal Length by Species', fontsize=12, fontweight='bold')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')

plt.subplot(1, 4, 4)
sns.boxplot(data=df, x='species', y='petal_width', palette='Set2')
plt.title('Petal Width by Species', fontsize=12, fontweight='bold')
plt.ylabel('Petal Width (cm)')
plt.xlabel('Species')

plt.tight_layout()
plt.savefig('iris_boxplots_by_species.png', dpi=300, bbox_inches='tight')
print("Saved: iris_boxplots_by_species.png")
plt.show()

print("\nAnalysis complete! All visualizations have been saved.")