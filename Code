import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats

# Load the dataset
file_path = "/content/BANKCHURN.csv"
df = pd.read_csv(file_path)

# Print the dimensions of the DataFrame and display the first 3 rows
print("DataFrame shape:", df.shape)
print(df.head(3))

# Rename columns
df.columns = [
    'RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',
    'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary', 'Exited', 'Complain',
    'SatisfactionScore', 'CardType', 'PointEarned'
]

# Descriptive Statistics
print("\nDescriptive Statistics:\n", df.describe())
# Drop non-numeric columns before calculating correlation matrix
numeric_df = df.select_dtypes(include=[np.number])

# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
