import pandas as pd
import numpy as np
import os

# Function to calculate moving average manually
def moving_average(values, window):
    return values.rolling(window=window).mean()

# Function to calculate the ADX (average directional index)
def calculate_adx(high, low, close, window=14):
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})
    df['TR'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    df['+DM'] = df['high'].diff().apply(lambda x: x if x > 0 else 0)
    df['-DM'] = df['low'].diff().apply(lambda x: -x if x < 0 else 0)
    df['+DI'] = 100 * df['+DM'].rolling(window).sum() / df['TR'].rolling(window).sum()
    df['-DI'] = 100 * df['-DM'].rolling(window).sum() / df['TR'].rolling(window).sum()
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(window).mean()
    return df['ADX']

# Function to calculate the strategy performance for a single stock and save results
def strategy_performance(df, stock_name, max_hold_days=30):
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df_last_two_years = df[df['date'] > (df['date'].max() - pd.DateOffset(years=2))].copy()

    df_last_two_years['10_sma_volume'] = moving_average(df_last_two_years['volume'], window=10)
    df_last_two_years['50_ema_close'] = moving_average(df_last_two_years['close'], window=50)
    df_last_two_years['200_ema_close'] = moving_average(df_last_two_years['close'], window=200)
    df_last_two_years['adx_14'] = calculate_adx(df_last_two_years['high'], df_last_two_years['low'], df_last_two_years['close'], window=14)

    # Calculate weekly and monthly closes
    df_last_two_years['weekly_close'] = df_last_two_years['close'].rolling(window=5).mean()
    df_last_two_years['monthly_close'] = df_last_two_years['close'].rolling(window=22).mean()

    best_trades = []
    best_profit = -np.inf
    best_hold_period = 0

    for hold_period in range(1, max_hold_days + 1):
        trades = []
        
        for i in range(2, len(df_last_two_years)):
            # Buying conditions
            latest_close = df_last_two_years['close'].iloc[i]
            one_day_ago_high = df_last_two_years['high'].iloc[i - 1]
            two_day_ago_high = df_last_two_years['high'].iloc[i - 2]
            one_day_ago_close = df_last_two_years['close'].iloc[i - 1]
            one_day_ago_max_high = df_last_two_years['high'].iloc[i - 1: i - 1 + 10].max()
            two_day_ago_max_high = df_last_two_years['high'].iloc[i - 2: i - 2 + 10].max()
            adx_14 = df_last_two_years['adx_14'].iloc[i]

            # First buying condition
            if (latest_close > one_day_ago_max_high and
                one_day_ago_close <= two_day_ago_max_high and
                two_day_ago_high < one_day_ago_max_high and
                latest_close > 50 and
                df_last_two_years['volume'].iloc[i] > df_last_two_years['10_sma_volume'].iloc[i] and
                df_last_two_years['volume'].iloc[i] > df_last_two_years['volume'].iloc[i - 1] and
                adx_14 > 20 and
                df_last_two_years['weekly_close'].iloc[i] > df_last_two_years['50_ema_close'].iloc[i]):

                buy_date = df_last_two_years['date'].iloc[i]
                buy_price = latest_close

                # Selling conditions within the hold period
                sell_date = None
                sell_price = None
                
                for j in range(i + 1, min(i + hold_period + 1, len(df_last_two_years))):
                    one_day_ago_ema50 = moving_average(df_last_two_years['close'], window=50).iloc[j - 1]
                    one_day_ago_ema200 = moving_average(df_last_two_years['close'], window=200).iloc[j - 1]

                    if ((one_day_ago_ema50 < one_day_ago_ema200 and
                         df_last_two_years['close'].iloc[j - 1] < one_day_ago_ema50 and
                         df_last_two_years['rsi'].iloc[j - 1] < 20 and
                         df_last_two_years['close'].iloc[j - 1] < df_last_two_years['high'].iloc[j - 1] and
                         df_last_two_years['close'].iloc[j - 1] < df_last_two_years['open'].iloc[j - 1]) or
                        (df_last_two_years['open'].iloc[j] - df_last_two_years['close'].iloc[j] > 
                         (df_last_two_years['high'].iloc[j] - df_last_two_years['low'].iloc[j]) * 0.5 and
                         df_last_two_years['close'].iloc[j] - df_last_two_years['low'].iloc[j] < 
                         (df_last_two_years['high'].iloc[j] - df_last_two_years['low'].iloc[j]) * 0.15) or
                        (df_last_two_years['high'].iloc[j - 1] > df_last_two_years['high'].iloc[j - 2] and
                         df_last_two_years['close'].iloc[j - 1] < df_last_two_years['high'].iloc[j - 2] and
                         df_last_two_years['volume'].iloc[j - 1] > df_last_two_years['volume'].iloc[j - 3] and
                         df_last_two_years['high'].iloc[j] < df_last_two_years['high'].iloc[j - 1])):

                        sell_date = df_last_two_years['date'].iloc[j]
                        sell_price = df_last_two_years['close'].iloc[j]
                        break

                if sell_price is None:
                    sell_price = df_last_two_years['close'].iloc[min(i + hold_period, len(df_last_two_years) - 1)]
                    sell_date = df_last_two_years['date'].iloc[min(i + hold_period, len(df_last_two_years) - 1)]

                profit_loss_pct = ((sell_price - buy_price) / buy_price) * 100
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

        if total_profit > best_profit:
            best_profit = total_profit
            best_hold_period = hold_period
            best_trades = trades

    best_trades_df = pd.DataFrame(best_trades)
    
    # Save trades to CSV file
    best_trades_df.to_csv(f'{stock_name}_trades.csv', index=False)

    # Calculate additional metrics
    total_trades = len(best_trades_df)
    average_profit_per_trade = best_trades_df['profit_loss_pct'].mean()
    total_profit = best_trades_df['profit_loss_pct'].sum()
    hit_ratio = len(best_trades_df[best_trades_df['profit_loss_pct'] > 0]) / total_trades if total_trades > 0 else 0

    # Save overall strategy performance to TXT file
    with open(f'{stock_name}_strategy_performance.txt', 'w') as f:
        f.write(f"Total Trades: {total_trades}\n")
        f.write(f"Best Hold Period: {best_hold_period} days\n")
        f.write(f"Total Profit: {total_profit:.2f}%\n")
        f.write(f"Average Profit per Trade: {average_profit_per_trade:.2f}%\n")
        f.write(f"Hit Ratio: {hit_ratio:.2%}\n")
    
    return best_trades_df, best_profit, best_hold_period