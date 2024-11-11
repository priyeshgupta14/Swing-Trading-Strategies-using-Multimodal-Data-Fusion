import pandas as pd
import numpy as np
import os
from datetime import datetime
from glob import glob

# Function to calculate moving average manually
def moving_average(values, window):
    return values.rolling(window=window).mean()

# Example function to integrate into your main strategy
def simulate_strategy(data, hold_periods, stock_name, available_capital, active_trades):
    trades = []
    
    for i in range(len(data) - max(hold_periods)):
        df_slice = data.iloc[:i + 1]

        # Only buy if there is capital available and less than 20 active trades
        if buy_condition(df_slice) and len(active_trades) < 300 and available_capital >= 2_00_000:
            for hold in hold_periods:
                sell_index = min(i + hold, len(data) - 1)
                sell_slice = data.iloc[:sell_index + 1]
                if sell_condition(sell_slice):
                    buy_price = data.iloc[i]['close']
                    sell_price = data.iloc[sell_index]['close']
                    profit_pct = (sell_price - buy_price) / buy_price * 100
                    trade_profit = 2_00_000 * (profit_pct / 100)  # Profit in INR for 1 lakh trade

                    trades.append({
                        'stock': stock_name,
                        'buy_date': data.iloc[i].name,
                        'sell_date': data.iloc[sell_index].name,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'hold_days': hold,
                        'profit_loss_pct': profit_pct,
                        'trade_profit': trade_profit
                    })

                    # Update active trades and adjust available capital
                    active_trades.append({
                        'stock': stock_name,
                        'buy_date': data.iloc[i].name,
                        'buy_price': buy_price,
                        'capital_allocated': 2_00_000,
                        'sell_date': data.iloc[sell_index].name,
                        'sell_price': sell_price,
                        'profit_pct': profit_pct,
                        'trade_profit': trade_profit
                    })
                    available_capital -= 2_00_000
                    break

    return trades, available_capital

# Function to calculate performance statistics
def calculate_overall_performance(trades, available_capital):
    total_trades = len(trades)
    profitable_trades = [t for t in trades if t['profit_loss_pct'] > 0]
    losing_trades = [t for t in trades if t['profit_loss_pct'] <= 0]

    total_profit = sum(t['trade_profit'] for t in trades)
    total_profit_pct = (total_profit / (300 * 2_00_000)) * 100
    avg_return_per_trade = total_profit / total_trades if total_trades > 0 else 0
    avg_profit = np.mean([t['trade_profit'] for t in profitable_trades]) if profitable_trades else 0
    avg_loss = np.mean([t['trade_profit'] for t in losing_trades]) if losing_trades else 0
    hit_ratio = len(profitable_trades) / total_trades if total_trades > 0 else 0
    risk_reward_ratio = avg_profit / abs(avg_loss) if avg_loss != 0 else 0

    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {len(profitable_trades)}")
    print(f"Losing Trades: {len(losing_trades)}")
    print(f"Total Profit/Loss in INR: {total_profit}")
    print(f"Total Profit/Loss Percentage: {total_profit_pct}")
    print(f"Average Return per Trade in INR: {avg_return_per_trade}")
    print(f"Average Profit per Trade in INR: {avg_profit}")
    print(f"Average Loss per Trade in INR: {avg_loss}")
    print(f"Hit Ratio (Success Rate): {hit_ratio}")
    print(f"Risk/Reward Ratio: {risk_reward_ratio}")

def buy_condition(df_slice):
    if len(df_slice) < 6:  # Ensure at least 6 rows for comparison (latest day + previous 5 days)
        return False
    
    latest_row = df_slice.iloc[-1]
    one_day_ago = df_slice.iloc[-2]
    two_days_ago = df_slice.iloc[-3]
    three_days_ago = df_slice.iloc[-4]
    four_days_ago = df_slice.iloc[-5]
    five_days_ago = df_slice.iloc[-6]

    return (
        latest_row['close'] > one_day_ago['close'] and
        one_day_ago['high'] > two_days_ago['close'] and
        two_days_ago['close'] > three_days_ago['low'] and
        three_days_ago['low'] > four_days_ago['close'] and
        four_days_ago['low'] > five_days_ago['close']
    )

# Function to check the sell condition
def sell_condition(df_slice):
    if len(df_slice) < 3:  # Ensure at least three rows for comparison
        return False

    latest_row = df_slice.iloc[-1]
    one_day_ago = df_slice.iloc[-2]
    two_days_ago = df_slice.iloc[-3]

    return (
        one_day_ago['low'] > two_days_ago['high'] and
        latest_row['close'] < one_day_ago['close'] and
        latest_row['volume'] > df_slice.iloc[-3]['volume'] and
        latest_row['high'] < one_day_ago['high']
    )

# Main loop to process multiple stocks
def process_stocks(folder_path):
    all_files = glob(os.path.join(folder_path, "*.csv"))
    hold_periods = range(2, 151)  # Hold period grid search from 2 to 60 days
    all_trades = []
    active_trades = []
    available_capital = 8_00_00_000  # Starting capital of 20 lakh

    for file in all_files:
        stock_name = os.path.basename(file).replace('.csv', '')
        data = pd.read_csv(file, index_col='date', parse_dates=True)

        # Filter last 3 years of data
        three_years_ago = datetime.now() - pd.DateOffset(years=5)
        data = data[data.index >= three_years_ago]

        if len(data) < 200:  # Ensure there's enough data for 200-day SMA
            continue

        trades, available_capital = simulate_strategy(data, hold_periods, stock_name, available_capital, active_trades)
        all_trades.extend(trades)

    # Save trade details to CSV
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv('strategy6_trades_with_capital.csv', index=False,
                     columns=['stock', 'buy_date', 'sell_date', 'buy_price', 'sell_price', 'hold_days', 'profit_loss_pct', 'trade_profit'])

    # Print overall performance
    calculate_overall_performance(all_trades, available_capital)

# Folder path for stock data
folder_path = 'daily_data_stocks'

# Run the processing function
process_stocks(folder_path)
