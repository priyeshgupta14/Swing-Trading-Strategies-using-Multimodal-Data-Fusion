import os
import pandas as pd
import numpy as np

# Folder path where CSV files are stored
data_folder = "daily_data_stocks"

# Load all CSV files from the folder
def load_data(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            # Ensure the date column is in datetime format
            df['date'] = pd.to_datetime(df['date'])
            df['stock'] = filename.split('.')[0]  # Add stock identifier from filename
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Load all data into a single DataFrame
data = load_data(data_folder)

# Calculate daily percentage change
data['pct_change'] = data['close'].pct_change() * 100

# Add additional date information columns
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['quarter'] = data['date'].dt.quarter
data['day_of_week'] = data['date'].dt.day_name()  # Monday, Tuesday, etc.

# Drop any rows with missing percentage change values (e.g., from stock market holidays)
data = data.dropna(subset=['pct_change'])

# 1. Day-wise Analysis (Seasonal Analysis) Matrix
day_wise_matrix = data.groupby(['stock', 'day_of_week'])['pct_change'].sum().unstack().reindex(
    columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
)
print("Day-wise Percentage Gain/Loss Matrix (Seasonal Analysis):")
print(day_wise_matrix)

# 2. Monthly Analysis Matrix
monthly_matrix = data.groupby(['stock', 'month'])['pct_change'].sum().unstack().reindex(
    columns=range(1, 13)
)
monthly_matrix.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print("\nMonthly Analysis Matrix (Average Percentage Gain/Loss):")
print(monthly_matrix)

# 3. Quarterly Analysis Matrix
quarterly_matrix = data.groupby(['stock', 'quarter'])['pct_change'].sum().unstack().reindex(
    columns=range(1, 5)
)
quarterly_matrix.columns = ['Q1', 'Q2', 'Q3', 'Q4']
print("\nQuarterly Analysis Matrix (Average Percentage Gain/Loss):")
print(quarterly_matrix)

# Optional: Save the matrices to CSV files for easy access
day_wise_matrix.to_csv("day_wise_matrix.csv")
monthly_matrix.to_csv("monthly_matrix.csv")
quarterly_matrix.to_csv("quarterly_matrix.csv")