import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(folder_path):
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            symbol = filename.split('.')[0]
            df = pd.read_csv(os.path.join(folder_path, filename))
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            data[symbol] = df
    return data

def calculate_indicators(df):
    # Calculate SMA
    for period in [10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()

    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['%K'] = (df['close'] - low_14) / (high_14 - low_14) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Calculate Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Std'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_lower'] = df['BB_Middle'] - 2 * df['BB_Std']

    # Calculate SuperTrend
    def calculate_supertrend(df, period=10, multiplier=1):
        atr = pd.DataFrame()
        atr['tr'] = np.maximum(
            (df['high'] - df['low']),
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr['atr'] = atr['tr'].rolling(window=period).mean()
        
        df['upperband'] = ((df['high'] + df['low']) / 2) + (multiplier * atr['atr'])
        df['lowerband'] = ((df['high'] + df['low']) / 2) - (multiplier * atr['atr'])
        df['supertrend'] = np.nan
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['upperband'].iloc[i-1]:
                df.loc[df.index[i], 'supertrend'] = df['lowerband'].iloc[i]
            elif df['close'].iloc[i] < df['lowerband'].iloc[i-1]:
                df.loc[df.index[i], 'supertrend'] = df['upperband'].iloc[i]
            else:
                df.loc[df.index[i], 'supertrend'] = df['supertrend'].iloc[i-1]
        
        return df['supertrend']

    df['SuperTrend_10_1'] = calculate_supertrend(df, 10, 1)
    df['SuperTrend_7_3'] = calculate_supertrend(df, 7, 3)

    return df

def check_buy_condition(df):
    return (
        (df['SuperTrend_10_1'] < df['close']) &
        (df['SuperTrend_10_1'].shift(1) >= df['close'].shift(1)) &
        (df['close'] > 50) &
        (df['SuperTrend_7_3'] < df['close']) &
        (df['RSI'] >= 50) &
        (df['close'] > df['SMA_10']) &
        (df['close'] > df['SMA_50']) &
        (df['close'] > df['SMA_200']) &
        (df['%K'] > df['%D']) &
        (df['SMA_10'] > df['SMA_20']) &
        (df['SMA_20'] > df['SMA_50']) &
        (df['SMA_50'] > df['SMA_100']) &
        (df['SMA_100'] > df['SMA_200']) &
        (df['MACD'] > df['Signal']) &
        (df['RSI'] < 70) &
        (df['close'] / df['close'].shift(1) < 1.07) &
        (df['volume'].shift(1) * 1.8 < df['volume'])
    )

def check_sell_condition(df, current_index, buy_price, buy_date):
    row = df.loc[current_index]
    prev_row = df.loc[df.index[df.index.get_loc(current_index) - 1]]
    days_held = (current_index - buy_date).days
    percent_change = (row['close'] - buy_price) / buy_price * 100

    condition1 = (
        percent_change < 0 and
        row['close'] < prev_row['close'] + (prev_row['close'] * 0.01) and
        row['close'] < (row['BB_Upper'] + row['BB_lower']) / 2 and
        prev_row['close'] >= (prev_row['BB_Upper'] + prev_row['BB_lower']) / 2
    )

    condition2 = (
        row['close'] < row['SMA_200'] and
        prev_row['close'] >= prev_row['SMA_200'] and
        row['volume'] > 100000
    )

    condition3 = (
        row['close'] > 50 and
        row['RSI'] < 50 and
        row['close'] < prev_row['low'] and
        row['close'] < row['SMA_50']
    )

    return condition1 or condition2 or condition3 or days_held >= 120

def backtest_strategy(data, max_hold_days=120, max_positions=10, capital_per_stock=200000):
    results = []
    current_positions = {}
    available_capital = capital_per_stock * max_positions

    for symbol, df in data.items():
        df = calculate_indicators(df)
        buy_signals = df[check_buy_condition(df)].index

        for buy_date in buy_signals:
            if len(current_positions) >= max_positions or available_capital < capital_per_stock:
                continue

            buy_price = df.loc[buy_date, 'close']
            shares = int(capital_per_stock / buy_price)
            cost = shares * buy_price

            for sell_date in df.loc[buy_date:].index[1:]:
                if check_sell_condition(df, sell_date, buy_price, buy_date):
                    sell_price = df.loc[sell_date, 'close']
                    profit = (sell_price - buy_price) * shares
                    profit_percentage = (sell_price - buy_price) / buy_price * 100
                    hold_days = (sell_date - buy_date).days

                    results.append({
                        'Symbol': symbol,
                        'Buy date': buy_date,
                        'Sell date': sell_date,
                        'Buy Price': buy_price,
                        'Sell Price': sell_price,
                        'Shares': shares,
                        'Profit': profit,
                        'Profit Percentage': profit_percentage,
                        'Hold Days': hold_days
                    })

                    available_capital += cost + profit
                    break

                if (sell_date - buy_date).days >= max_hold_days:
                    sell_price = df.loc[sell_date, 'close']
                    profit = (sell_price - buy_price) * shares
                    profit_percentage = (sell_price - buy_price) / buy_price * 100

                    results.append({
                        'Symbol': symbol,
                        'Buy date': buy_date,
                        'Sell date': sell_date,
                        'Buy Price': buy_price,
                        'Sell Price': sell_price,
                        'Shares': shares,
                        'Profit': profit,
                        'Profit Percentage': profit_percentage,
                        'Hold Days': max_hold_days
                    })

                    available_capital += cost + profit
                    break

    return pd.DataFrame(results)

def optimize_hold_period(data, max_hold_days=120, step=5):
    best_profit = float('-inf')
    best_hold_days = 0

    for hold_days in range(2, max_hold_days + 1, step):
        results = backtest_strategy(data, max_hold_days=hold_days)
        total_profit = results['Profit'].sum()

        if total_profit > best_profit:
            best_profit = total_profit
            best_hold_days = hold_days

    return best_hold_days

def analyze_results(results):
    max_positions=10
    capital_per_stock=200000
    total_trades = len(results)
    profitable_trades = len(results[results['Profit'] > 0])
    losing_trades = len(results[results['Profit'] <= 0])
    
    avg_trades_per_month = total_trades / (results['Sell date'].max() - results['Buy date'].min()).days * 30
    
    profit_streak = (results['Profit'] > 0).astype(int)
    loss_streak = (results['Profit'] <= 0).astype(int)
    max_profitable_streak = profit_streak.groupby((profit_streak != profit_streak.shift()).cumsum()).sum().max()
    max_losing_streak = loss_streak.groupby((loss_streak != loss_streak.shift()).cumsum()).sum().max()
    
    hit_ratio = profitable_trades / total_trades if total_trades > 0 else 0
    
    avg_profit = results[results['Profit'] > 0]['Profit'].mean()
    avg_loss = abs(results[results['Profit'] <= 0]['Profit'].mean())
    risk_reward_ratio = avg_profit / avg_loss if avg_loss != 0 else float('inf')
    
    avg_return_per_trade = results['Profit'].mean()
    total_profit_loss = results['Profit'].sum()
    total_profit_loss_percentage = (total_profit_loss / (capital_per_stock * max_positions)) * 100

    return {
        'Total Trades': total_trades,
        'Profitable Trades': profitable_trades,
        'Losing Trades': losing_trades,
        'Avg Trades per Month': avg_trades_per_month,
        'Max Profitable Streak': max_profitable_streak,
        'Max Losing Streak': max_losing_streak,
        'Hit Ratio': hit_ratio,
        'Risk Reward Ratio': risk_reward_ratio,
        'Average Return per Trade': avg_return_per_trade,
        'Avg Profit per Trade': avg_profit,
        'Avg Loss per Trade': avg_loss,
        'Total Profit/Loss': total_profit_loss,
        'Total Profit/Loss Percentage': total_profit_loss_percentage
    }

# Main execution
folder_path = 'sector_wise_daily_data/NIFTY CPSE'
data = load_data(folder_path)

# Optimize hold period
best_hold_days = optimize_hold_period(data)
print(f"Optimal hold period: {best_hold_days} days")

# Run backtest with optimal hold period
results = backtest_strategy(data, max_hold_days=best_hold_days)

# Analyze results
analysis = analyze_results(results)

# Save results to CSV
results.to_csv('strategy3_results_cpse.csv', index=False)

# Print analysis
for key, value in analysis.items():
    print(f"{key}: {value}")
    
# for best performing stocks: Optimal hold period: 42 days
# Total Trades: 19
# Profitable Trades: 9
# Losing Trades: 10
# Avg Trades per Month: 0.38961038961038963
# Max Profitable Streak: 4
# Max Losing Streak: 5
# Hit Ratio: 0.47368421052631576
# Risk Reward Ratio: 11.361885821838706
# Average Return per Trade: 22739.45
# Avg Profit per Trade: 53208.961111111115
# Avg Loss per Trade: 4683.110000000004
# Total Profit/Loss: 432049.55000000005
# Total Profit/Loss Percentage: 21.602477500000003