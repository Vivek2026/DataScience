import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Survival Count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived')
plt.title("Survival Count (0 = Died, 1 = Survived)")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2. Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 3. Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 4. Age Distribution of Survivors
plt.figure(figsize=(6, 4))
sns.histplot(data=df[df['Survived'] == 1], x='Age', bins=30, kde=True)
plt.title("Age Distribution of Survivors")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 5. Fare Distribution by Class
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Pclass', y='Fare')
plt.title("Fare Distribution by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.show()

# 6. Heatmap of Correlation
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
