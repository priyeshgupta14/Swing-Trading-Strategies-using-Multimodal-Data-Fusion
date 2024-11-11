import os
import pandas as pd

# Folder path where CSV files are stored
data_folder = "daily_data_stocks"

# Load all CSV files from the folder
def load_data(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])  # Ensure the date column is in datetime format
            df['stock'] = filename.split('.')[0]  # Add stock identifier from filename
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Load all data into a single DataFrame
data = load_data(data_folder)

# Add additional date information columns
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['quarter'] = data['date'].dt.quarter
data['day_of_week'] = data['date'].dt.day_name()  # Monday, Tuesday, etc.

# Sort data for cumulative calculations
data = data.sort_values(by=['stock', 'date'])

# Define a helper function to calculate overall percentage gain/loss
def calculate_overall_pct_change(group):
    first_close = group.iloc[0]['close']
    last_close = group.iloc[-1]['close']
    return ((last_close - first_close) / first_close) * 100

# 1. Day-wise (Seasonal) Analysis Matrix - Overall percentage gain/loss
day_wise_matrix = data.groupby(['stock', 'year', 'day_of_week']).apply(calculate_overall_pct_change).unstack().reindex(
    columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
)
print("Day-wise Overall Percentage Gain/Loss Matrix (Seasonal Analysis):")
print(day_wise_matrix)

# 2. Monthly Analysis Matrix - Overall percentage gain/loss
monthly_matrix = data.groupby(['stock', 'year', 'month']).apply(calculate_overall_pct_change).unstack().reindex(
    columns=range(1, 13)
)
monthly_matrix.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print("\nMonthly Overall Percentage Gain/Loss Matrix:")
print(monthly_matrix)

# 3. Quarterly Analysis Matrix - Overall percentage gain/loss
quarterly_matrix = data.groupby(['stock', 'year', 'quarter']).apply(calculate_overall_pct_change).unstack().reindex(
    columns=range(1, 5)
)
quarterly_matrix.columns = ['Q1', 'Q2', 'Q3', 'Q4']
print("\nQuarterly Overall Percentage Gain/Loss Matrix:")
print(quarterly_matrix)

# 4. Year-wise Analysis Matrix - Overall percentage gain/loss
yearly_matrix = data.groupby(['stock', 'year']).apply(calculate_overall_pct_change).unstack()
print("\nYear-wise Overall Percentage Gain/Loss Matrix:")
print(yearly_matrix)

# 5. Overall Analysis - Overall percentage gain/loss across all years
overall_matrix = data.groupby(['stock']).apply(calculate_overall_pct_change)
print("\nOverall Percentage Gain/Loss Matrix across all years:")
print(overall_matrix)

# Optional: Save the matrices to CSV files for easy access
day_wise_matrix.to_csv("day_wise_overall_matrix.csv")
monthly_matrix.to_csv("monthly_overall_matrix.csv")
quarterly_matrix.to_csv("quarterly_overall_matrix.csv")
yearly_matrix.to_csv("yearly_overall_matrix.csv")
overall_matrix.to_csv("overall_matrix.csv")