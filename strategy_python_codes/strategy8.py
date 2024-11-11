import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

def calculate_true_range(df):
    """Calculate True Range manually"""
    high_low = df['high'] - df['low']
    high_cp = abs(df['high'] - df['close'].shift(1))
    low_cp = abs(df['low'] - df['close'].shift(1))
    
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    return tr

def calculate_sma(series, window):
    """Calculate Simple Moving Average"""
    return series.rolling(window=window).mean()

def calculate_rsi(df, period=14):
    """Calculate RSI manually"""
    # Calculate price changes
    delta = df['close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def analyze_stock(df, max_hold_days=120, min_hold_days=2):
    """Analyze a single stock based on the given strategy"""
    df = df.copy()
    
    # Calculate required indicators
    df['TR'] = calculate_true_range(df)
    df['volume_SMA20'] = calculate_sma(df['volume'], 20)
    df['high_20_Max'] = df['high'].rolling(20).max()
    df['low_20_Min'] = df['low'].rolling(20).min()
    df['RSI'] = calculate_rsi(df)
    
    # Initialize signals
    buy_signals = []
    trade_results = []
    
    # Keep track of open positions
    in_position = False
    buy_price = 0
    buy_date = None
    position_days = 0
    buy_day_low = 0  # Store the low price of the buy day
    
    for i in range(20, len(df)-1):  # Need 20 days for indicators
        current_date = df.index[i]
        
        if not in_position:
            # Buy conditions
            buy_condition = (
                df['TR'].iloc[i] > df['close'].iloc[i] * 0.03 and
                (df['high_20_Max'].iloc[i] - df['low_20_Min'].iloc[i]) < df['close'].iloc[i] * 0.1 and
                df['volume'].iloc[i] > df['volume_SMA20'].iloc[i]
            )
            
            if buy_condition:
                buy_price = df['close'].iloc[i]
                buy_date = current_date
                buy_day_low = df['low'].iloc[i]  # Store the low of the buy day
                in_position = True
                position_days = 0
                
        else:
            position_days += 1
            
            # Updated sell conditions:
            # 1. Price closes below buy day's low
            # 2. Price closes 5% below buy price
            # 3. RSI below 45
            sell_condition = ( # Below buy day's low
                df['close'].iloc[i] < (buy_price * 0.90) or  # 15% below buy price
                df['RSI'].iloc[i] < 35  # RSI below 35
            )
            
            # Force sell if max hold days reached
            if sell_condition or position_days >= max_hold_days:
                sell_price = df['close'].iloc[i]
                profit_pct = ((sell_price - buy_price) / buy_price) * 100
                
                if position_days >= min_hold_days:  # Only record trades that meet minimum hold time
                    trade_results.append({
                        'stock': df.index.name,
                        'buy_date': buy_date,
                        'sell_date': current_date,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'hold_days': position_days,
                        'profit_loss_pct': profit_pct
                    })
                
                in_position = False
    
    return trade_results

def calculate_performance_metrics(trades_df):
    """Calculate performance metrics for a set of trades"""
    if len(trades_df) == 0:
        return None
    
    metrics = {
        'total_trades': len(trades_df),
        'profitable_trades': len(trades_df[trades_df['profit_loss_pct'] > 0]),
        'losing_trades': len(trades_df[trades_df['profit_loss_pct'] <= 0]),
        'avg_trades_per_month': len(trades_df) / ((trades_df['sell_date'].max() - trades_df['buy_date'].min()).days / 30),
        'hit_ratio': len(trades_df[trades_df['profit_loss_pct'] > 0]) / len(trades_df),
        'avg_return_per_trade': trades_df['profit_loss_pct'].mean(),
        'avg_profit_per_trade': trades_df[trades_df['profit_loss_pct'] > 0]['profit_loss_pct'].mean() if len(trades_df[trades_df['profit_loss_pct'] > 0]) > 0 else 0,
        'avg_loss_per_trade': trades_df[trades_df['profit_loss_pct'] <= 0]['profit_loss_pct'].mean() if len(trades_df[trades_df['profit_loss_pct'] <= 0]) > 0 else 0,
        'total_profit_loss_pct': trades_df['profit_loss_pct'].sum()
    }
    
    # Calculate streaks
    profits = (trades_df['profit_loss_pct'] > 0).astype(int)
    streak = profits.groupby((profits != profits.shift()).cumsum()).cumcount() + 1
    metrics['max_profitable_streak'] = streak[profits == 1].max() if len(streak[profits == 1]) > 0 else 0
    metrics['max_losing_streak'] = streak[profits == 0].max() if len(streak[profits == 0]) > 0 else 0
    
    # Calculate risk-reward ratio
    avg_profit = metrics['avg_profit_per_trade']
    avg_loss = abs(metrics['avg_loss_per_trade']) if metrics['avg_loss_per_trade'] != 0 else 1
    metrics['risk_reward_ratio'] = avg_profit / avg_loss
    
    return metrics

def main():
    # Parameters
    data_folder = "daily_data_stocks"  # Replace with your folder path
    output_file = "strategy8_results.csv"
    max_positions = 5
    capital_per_stock = 200000  # 2 lacs
    total_capital = 2000000  # 20 lacs
    
    all_trades = []
    
    # Process each stock file
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_folder, file))
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index.name = file.replace('.csv', '')
            
            # Get trades for this stock
            trades = analyze_stock(df)
            all_trades.extend(trades)
    
    # Convert to DataFrame and sort by date
    trades_df = pd.DataFrame(all_trades)
    if len(trades_df) > 0:
        trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date'])
        trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date'])
        trades_df = trades_df.sort_values('buy_date')
        
        # Save trades to CSV
        trades_df.to_csv(output_file, index=False)
        
        # Calculate and print performance metrics
        metrics = calculate_performance_metrics(trades_df)
        if metrics:
            print("\nPerformance Metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value:.2f}")
    else:
        print("No trades found matching the strategy criteria.")

if __name__ == "__main__":
    main()