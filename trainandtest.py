#!/usr/bin/env python3

# prepare_data.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Load your cleaned data
df = pd.read_csv('cleaned_rideshare_data.csv')

# Separate features (X) and target variable (y)
X = df.drop(columns=['price'])  # 'price' is the target variable
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize the data
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Save each set to separate CSV files
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Data has been split into training and testing sets and saved as CSV files.")
