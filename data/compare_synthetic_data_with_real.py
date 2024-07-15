import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV files
file1 = '/Users/rachana/tanpfn/data/dataset/titanic.csv'
file2 = '/Users/rachana/tanpfn/data/dataset/synthetic_data_with_20000_epochs.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# List of features/columns to compare
features = df1.columns.intersection(df2.columns)

# Function to plot distribution of a feature
def plot_feature_distribution(df1, df2, feature):
    plt.figure(figsize=(10, 6))
    sns.histplot(df1[feature], label='Real Titanic data', kde=True, color='blue', stat='density', bins=30, alpha=0.5)
    sns.histplot(df2[feature], label='Synthetically generated titanic dataset', kde=True, color='red', stat='density', bins=30, alpha=0.5)
    plt.title(f'Distribution Comparison for Feature: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Loop through each feature and plot its distribution
for feature in features:
    plot_feature_distribution(df1, df2, feature)
