import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r'C:\Users\tcong\OneDrive\Documents\GitHub\tests\newML\Real_Estate.csv'
df = pd.read_csv(file_path)
df.head()

df.info()

# Check for missing values
df.isnull().sum()


# Convert 'Transaction date' to datetime format
df['Transaction date'] = pd.to_datetime(df['Transaction date'], format='ISO8601')
df.head()

# Plotting pairplot to see relationships
#sns.pairplot(df)
#plt.show()

# Correlation heatmap
#numeric_df = df.select_dtypes(include=[np.number])
#plt.figure(figsize=(10, 8))
#sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
#plt.show()


# Define features and target variable
X = df.drop(columns=['House price of unit area', 'Transaction date'])
y = df['House price of unit area']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse, r2
