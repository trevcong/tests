import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Read the CSV file
df = pd.read_csv(r'C:\Users\tcong\OneDrive\Documents\GitHub\tests\newML\Real_Estate.csv')

# Display the first few rows of the dataframe
print(df.head())

# Check for null values
print(df.isnull().sum())

# Describe the dataset
print(df.describe())

# Display info about the dataset
print(df.info())

# Convert 'Transaction date' to datetime
df['Transaction date'] = pd.to_datetime(df['Transaction date'])

# Insert new columns at specific positions
df.insert(1, 'Transaction year', df['Transaction date'].dt.year)
df.insert(2, 'Transaction month', df['Transaction date'].dt.month)
df.insert(3, 'Transaction day', df['Transaction date'].dt.day)
df.insert(4, 'Transaction day of week', df['Transaction date'].dt.dayofweek)

# Drop the original 'Transaction date' column
df = df.drop(columns=['Transaction date'])

# Display the first few rows of the updated dataframe
print(df.head())

# Calculate correlation matrix
correlation_matrix = df.corr()

# Sort correlations with respect to 'House price of unit area' in descending order
sorted_correlations = correlation_matrix['House price of unit area'].sort_values(ascending=False)
print(sorted_correlations)

# Scaling
std = StandardScaler()
x = df.drop('House price of unit area', axis=1)
y = df['House price of unit area']
x_scaled = std.fit_transform(x)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=1)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict on the test set
y_pred = lr.predict(x_test)

# Print mean squared error and r2 score
print('mean_squared_error:', mean_squared_error(y_test, y_pred))
print('r2_score:', r2_score(y_test, y_pred))

# Get the weights (coefficients) and bias (intercept)
weights = lr.coef_
bias = lr.intercept_

print('Weights (coefficients):', weights)
print('Bias (intercept):', bias)

# Visualize true values vs predicted values
plt.figure(figsize=(10, 6))

# Plotting the true values
plt.scatter(y_test, y_pred, color='blue', label='True vs Predicted')

# Plotting the line where true = predicted
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect prediction')

plt.title('True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()