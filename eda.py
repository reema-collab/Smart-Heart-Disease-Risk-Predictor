import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("heart.csv")

print("\n📌 Dataset first 5 rows:")
print(df.head())

print("\n📌 Dataset shape (rows, columns):")
print(df.shape)

print("\n📌 Column names:")
print(df.columns)

print("\n📌 Missing values in each column:")
print(df.isnull().sum())

print("\n📌 Statistical summary:")
print(df.describe())

# Visualizations
plt.figure(figsize=(10, 4))
sns.countplot(x='target', data=df)
plt.title("Heart Disease Distribution (0=No, 1=Yes)")
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
