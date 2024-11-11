import pandas as pd
import numpy as np
import os
from datetime import timedelta

# Function to calculate moving average manually
def moving_average(values, window):
    return values.rolling(window=window).mean()

# Function to calculate the strategy performance for a single stock with grid search
def strategy_performance(df, stock_name, max_hold_days=60):
    # Ensure column names are in lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Filter the data to the last two years
    df['date'] = pd.to_datetime(df['date'])
    df_last_two_years = df[df['date'] > (df['date'].max() - pd.DateOffset(years=5))].copy()

    # Calculate moving averages manually
    df_last_two_years['10_sma_volume'] = moving_average(df_last_two_years['volume'], window=10)
    df_last_two_years['200_sma_close'] = moving_average(df_last_two_years['close'], window=200)

    best_trades = []
    best_profit = -np.inf
    best_hold_period = 0

    # Loop through possible holding periods
    for hold_period in range(1, max_hold_days + 1):
        trades = []
        
        # Loop through the data and apply the strategy
        for i in range(1, len(df_last_two_years)):
            # Buying strategy
            if (df_last_two_years['volume'].iloc[i] > df_last_two_years['10_sma_volume'].iloc[i] * 5 and 
                (df_last_two_years['close'].iloc[i] > df_last_two_years['close'].iloc[i-1] * 1.05 or
                 df_last_two_years['close'].iloc[i] < df_last_two_years['close'].iloc[i-1] * 0.95)):

                buy_date = df_last_two_years['date'].iloc[i]
                buy_price = df_last_two_years['close'].iloc[i]

                # Check sell conditions within the hold period
                sell_date = None
                sell_price = None
                
                for j in range(i + 1, min(i + hold_period + 1, len(df_last_two_years))):
                    if (df_last_two_years['close'].iloc[j] < df_last_two_years['200_sma_close'].iloc[j] and 
                        df_last_two_years['close'].iloc[j-1] >= df_last_two_years['200_sma_close'].iloc[j-1] and
                        df_last_two_years['volume'].iloc[j] > 100000):
                        sell_date = df_last_two_years['date'].iloc[j]
                        sell_price = df_last_two_years['close'].iloc[j]
                        break
                
                if sell_price is None:
                    # Sell on the last available date if no sell condition met
                    sell_price = df_last_two_years['close'].iloc[min(i + hold_period, len(df_last_two_years) - 1)]
                    sell_date = df_last_two_years['date'].iloc[min(i + hold_period, len(df_last_two_years) - 1)]

                # Calculate profit/loss percentage
                profit_loss_pct = ((sell_price - buy_price) / buy_price) * 100

                # Record trade details
                trades.append({
                    'stock': stock_name,
                    'buy_date': buy_date,
                    'sell_date': sell_date,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'hold_days': (sell_date - buy_date).days,
                    'profit_loss_pct': profit_loss_pct
                })

        total_profit = sum([trade['profit_loss_pct'] for trade in trades])
        
        # Update best trades if total profit is better than previous
        if total_profit > best_profit:
            best_profit = total_profit
            best_hold_period = hold_period
            best_trades = trades

    # Convert best trades to DataFrame for further analysis
    best_trades_df = pd.DataFrame(best_trades)
    return best_trades_df, best_profit, best_hold_period

# Function to evaluate strategy performance on all stocks
def evaluate_strategy(stocks_folder, max_hold_days=60):
    result_trades = []

    # Loop through all stock files in the folder
    for stock_file in os.listdir(stocks_folder):
        if stock_file.endswith('.csv'):
            stock_name = stock_file.replace('.csv', '')
            stock_data = pd.read_csv(os.path.join(stocks_folder, stock_file))

            # Get performance data for each stock
            trades_df, total_profit, hold_period = strategy_performance(stock_data, stock_name, max_hold_days)
            result_trades.append(trades_df)

    # Combine results for all stocks
    all_trades_df = pd.concat(result_trades, ignore_index=True)
    return all_trades_df

# Summary statistics calculation
def calculate_summary_stats(trades_df):
    total_trades = len(trades_df)
    profitable_trades = len(trades_df[trades_df['profit_loss_pct'] > 0])
    losing_trades = total_trades - profitable_trades
    avg_trades_per_month = total_trades / 24  # for 2 years
    max_profitable_streak = (trades_df['profit_loss_pct'] > 0).astype(int).groupby(trades_df['profit_loss_pct'] < 0).cumsum().max()
    max_losing_streak = (trades_df['profit_loss_pct'] < 0).astype(int).groupby(trades_df['profit_loss_pct'] > 0).cumsum().max()
    hit_ratio = profitable_trades / total_trades if total_trades > 0 else 0
    avg_return_per_trade = trades_df['profit_loss_pct'].mean() if total_trades > 0 else 0
    avg_profit_per_trade = trades_df[trades_df['profit_loss_pct'] > 0]['profit_loss_pct'].mean() if profitable_trades > 0 else 0
    avg_loss_per_trade = trades_df[trades_df['profit_loss_pct'] < 0]['profit_loss_pct'].mean() if losing_trades > 0 else 0

    # Risk-Reward ratio (avg profit per trade / abs(avg loss per trade))
    risk_reward_ratio = abs(avg_profit_per_trade / avg_loss_per_trade) if avg_loss_per_trade != 0 else float('inf')

    # Calculate total profit/loss percentage across all trades
    total_profit_loss_pct = trades_df['profit_loss_pct'].sum()

    return {
        'Total Trades': total_trades,
        'Profitable Trades': profitable_trades,
        'Losing Trades': losing_trades,
        'Average Trades per Month': avg_trades_per_month,
        'Max Profitable Streak': max_profitable_streak,
        'Max Losing Streak': max_losing_streak,
        'Hit Ratio': hit_ratio,
        'Risk Reward Ratio': risk_reward_ratio,
        'Average Return per Trade': avg_return_per_trade,
        'Average Profit per Trade': avg_profit_per_trade,
        'Average Loss per Trade': avg_loss_per_trade,
        'Total Profit/Loss Percentage': total_profit_loss_pct
    }

# Define your stocks daily data folder
stocks_folder = 'daily_data_stocks'

# Get all trades across stocks with grid search for holding period
all_trades = evaluate_strategy(stocks_folder, max_hold_days=60)

# Calculate summary stats
summary_stats = calculate_summary_stats(all_trades)
print(summary_stats)

# Save all trades data to a CSV
all_trades.to_csv('strategy2_trades.csv', index=False)