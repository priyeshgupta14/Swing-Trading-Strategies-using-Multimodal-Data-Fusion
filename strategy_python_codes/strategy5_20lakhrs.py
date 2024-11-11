import pandas as pd
import numpy as np
import os
from glob import glob
from datetime import datetime

# Function to calculate EMA
def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

# Function to calculate SMA
def calculate_sma(data, period):
    return data.rolling(window=period).mean()

# Function to implement the buy condition
def buy_condition(df):
    if len(df) < 2:  # Ensure there are at least two rows
        return False
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    condition = (latest['open'] < latest['close'] and
                 prev['open'] > prev['close'] and
                 latest['open'] < prev['close'] and
                 latest['close'] > prev['open'] and
                 max(df['close'].tail(5)) < calculate_ema(df['close'], 20).iloc[-1] and
                 calculate_sma(df['volume'], 5).iloc[-1] < latest['volume'])
    return condition

# Function to implement the sell condition
def sell_condition(df):
    if len(df) < 2:  # Ensure there are at least two rows
        return False
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    sma_200 = calculate_sma(df['close'], 200).iloc[-1]
    prev_sma_200 = calculate_sma(df['close'], 200).iloc[-2]
    
    condition = (latest['close'] < sma_200 and 
                 prev['close'] >= prev_sma_200 and
                 latest['volume'] > 100000)
    return condition

# Function to simulate the strategy and grid search for hold days
def simulate_strategy(data, hold_periods, stock_name, available_capital, active_trades):
    trades = []
    for i in range(len(data) - max(hold_periods)):
        df_slice = data.iloc[:i + 1]
        
        # Ensure that you can only buy if you have capital left and less than 20 active trades
        if buy_condition(df_slice) and len(active_trades) < 50:
            for hold in hold_periods:
                sell_index = min(i + hold, len(data) - 1)
                sell_slice = data.iloc[:sell_index + 1]
                if sell_condition(sell_slice):
                    buy_price = data.iloc[i]['close']
                    sell_price = data.iloc[sell_index]['close']
                    profit_pct = (sell_price - buy_price) / buy_price * 100
                    trade_profit = 2_00_000 * (profit_pct / 100)  # Profit in INR based on 1 lakh per trade
                    
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
                    
                    # Add the trade to active trades
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
                    break
    return trades

# Function to calculate overall performance statistics
def calculate_overall_performance(trades, available_capital):
    total_trades = len(trades)
    profitable_trades = [t for t in trades if t['profit_loss_pct'] > 0]
    losing_trades = [t for t in trades if t['profit_loss_pct'] <= 0]
    
    total_profit = sum(t['trade_profit'] for t in trades)  # Total in INR
    total_profit_pct = (total_profit / (50 * 2_00_000)) * 100  # Profit as percentage of total capital
    avg_return_per_trade = total_profit / total_trades if total_trades > 0 else 0
    avg_profit = np.mean([t['trade_profit'] for t in profitable_trades]) if profitable_trades else 0
    avg_loss = np.mean([t['trade_profit'] for t in losing_trades]) if losing_trades else 0
    hit_ratio = len(profitable_trades) / total_trades if total_trades > 0 else 0
    risk_reward_ratio = avg_profit / abs(avg_loss) if avg_loss != 0 else 0
    
    print("Total Trades:", total_trades)
    print("Profitable Trades:", len(profitable_trades))
    print("Losing Trades:", len(losing_trades))
    print("Total Profit/Loss in INR:", total_profit)
    print("Total Profit/Loss Percentage:", total_profit_pct)
    print("Average Return per Trade in INR:", avg_return_per_trade)
    print("Average Profit per Trade in INR:", avg_profit)
    print("Average Loss per Trade in INR:", avg_loss)
    print("Hit Ratio (Success Rate):", hit_ratio)
    print("Risk/Reward Ratio:", risk_reward_ratio)

# Main processing loop for all stocks in the folder
def process_stocks(folder_path):
    all_files = glob(os.path.join(folder_path, "*.csv"))
    hold_periods = range(2, 76)  # Grid search for hold period between 2 to 60 days
    all_trades = []
    active_trades = []
    available_capital = 1_00_00_000  # Starting with 20 lakh rupees

    for file in all_files:
        stock_name = os.path.basename(file).replace('.csv', '')
        data = pd.read_csv(file, index_col='date', parse_dates=True)
        
        # Filter for last 3 years of data
        three_years_ago = datetime.now() - pd.DateOffset(years=5)
        data = data[data.index >= three_years_ago]

        if len(data) < 200:  # Ensure there's enough data for 200-day SMA
            continue

        trades = simulate_strategy(data, hold_periods, stock_name, available_capital, active_trades)
        all_trades.extend(trades)

    # Save specific trade details to CSV
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv('strategy_trades_with_capital.csv', index=False,
                     columns=['stock', 'buy_date', 'sell_date', 'buy_price', 'sell_price', 'hold_days', 'profit_loss_pct', 'trade_profit'])

    # Print overall performance
    calculate_overall_performance(all_trades, available_capital)

# Folder containing all stock CSV files
folder_path = 'daily_data_stocks'

# Run the stock processing
process_stocks(folder_path)