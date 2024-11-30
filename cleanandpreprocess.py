#!/usr/bin/env python3
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data from a CSV file (change the filename accordingly)
# Example: df = pd.read_csv('rideshare_kaggle.csv')
df = pd.read_csv('rideshare_kaggle.csv')

# Step 1: Keep only relevant columns
relevant_columns = [
    'price', 'distance', 'surge_multiplier', 
    'latitude', 'longitude', 'source', 'destination', 
    'hour', 'day', 'month', 'timestamp', 
    'temperature', 'humidity', 'precipProbability', 'windSpeed', 
    'short_summary', 'cab_type'
]
df = df[relevant_columns]

# Step 2: Handle Missing Data
# Fill missing numerical values with the median of each column
numerical_columns = ['price', 'distance', 'surge_multiplier', 'latitude', 'longitude', 
                     'temperature', 'humidity', 'precipProbability', 'windSpeed']
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Fill missing categorical values in 'short_summary' with a placeholder
df['short_summary'].fillna('Unknown', inplace=True)

# Step 3: Feature Engineering
# Convert 'timestamp' from Unix time to datetime and extract new time-based features
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df['weekday'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)  # 1 for Saturday/Sunday

# Simplify weather conditions for easier analysis (e.g., combine all types of rain)
df['weather_condition'] = df['short_summary'].apply(lambda x: 'Rainy' if 'Rain' in x else 'Clear')

# Drop 'timestamp' and 'short_summary' as they have been used to create new features
df.drop(columns=['timestamp', 'short_summary'], inplace=True)

# Step 4: Scaling Numerical Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['distance', 'temperature', 'humidity', 'windSpeed']])
df[['distance', 'temperature', 'humidity', 'windSpeed']] = scaled_features

# Step 5: Encoding Categorical Features
encoder = LabelEncoder()
df['source'] = encoder.fit_transform(df['source'])
df['destination'] = encoder.fit_transform(df['destination'])
df['weather_condition'] = encoder.fit_transform(df['weather_condition'])

# Step 6: Save the cleaned and preprocessed data to a new CSV file
df.to_csv('cleaned_rideshare_data.csv', index=False)

print("Data cleaning and preprocessing completed. Saved to 'cleaned_rideshare_data.csv'.")
