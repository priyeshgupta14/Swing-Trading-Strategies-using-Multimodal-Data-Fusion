import pandas as pd
import os
from datetime import datetime, timedelta

# Load stock and news data from respective folders
stock_folder = 'daily_data_stocks'
news_folder = 'news_sentiments'

# Load all stock data into a dictionary {stock_name: dataframe}
stock_data = {}
for file in os.listdir(stock_folder):
    stock_name = file.replace('.csv', '')
    df = pd.read_csv(os.path.join(stock_folder, file))
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')

    stock_data[stock_name] = df

# Load news data into a single dataframe
news_data = []
for file in os.listdir(news_folder):
    df = pd.read_csv(os.path.join(news_folder, file))
    # Use mixed format for automatic date parsing
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')

    news_data.append(df)
news_data = pd.concat(news_data)

# Function to calculate SMA manually
def calculate_sma(series, window):
    return series.rolling(window=window).mean()

# Load news data into a single dataframe with stock name as a new column
news_data = []
for file in os.listdir(news_folder):
    stock_name = file.replace('.csv', '')
    df = pd.read_csv(os.path.join(news_folder, file))
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['stock'] = stock_name  # Add stock name as a column
    news_data.append(df)
news_data = pd.concat(news_data)

# Function to get news sentiment for a given date range
def get_sentiment(stock, date, sentiment_type, days_range):
    relevant_news = news_data[
        (news_data['date'] >= date - timedelta(days=days_range)) & 
        (news_data['date'] <= date) & 
        (news_data['stock'] == stock)
    ]
    if sentiment_type == 'negative':
        sentiment_news = relevant_news[relevant_news['desc_sentiment'].isin(['negative'])]
    else:
        sentiment_news = relevant_news[relevant_news['desc_sentiment'].isin(['positive','negative'])]
    # Return the closest news sentiment to the given date, if any
    return sentiment_news.sort_values('date', ascending=False).head(1)

# Backtesting logic
results = []

for stock_name, df in stock_data.items():
    # Calculate SMA(200) for the stock
    df['sma_200'] = calculate_sma(df['close'], 200)
    
    # Iterate over the dataframe to identify buy/sell signals
    for i in range(1, len(df)):
        buy_signal = (df.loc[i, 'open'] == df.loc[i, 'low'] and 
                      df.loc[i, 'open'] > df.loc[i-1, 'close'] * 1.01 and 
                      df.loc[i, 'close'] >= df.loc[i, 'open'] * 1.01)
        
        if buy_signal:
            buy_date = df.loc[i, 'date']
            buy_price = df.loc[i, 'close']
            # Get positive sentiment within 30 days before the buy signal
            sentiment_news = get_sentiment(stock_name, buy_date, 'negative', 5)
            if sentiment_news.empty:
                continue  # Skip buy if no positive sentiment is found
            buying_day_near_desc_sentiment = sentiment_news.iloc[0]['desc_sentiment']
            
            # Find a sell signal after buying, within 60 days max
            for j in range(i+1, min(i+60, len(df))):
                sell_signal = (df.loc[j, 'close'] < df.loc[j, 'sma_200'] and 
                               df.loc[j-1, 'close'] >= df.loc[j-1, 'sma_200']
                               )
                
                if sell_signal:
                    sell_date = df.loc[j, 'date']
                    sell_price = df.loc[j, 'close']
                    # Get negative or neutral sentiment within 30 days before the sell signal
                    sell_sentiment_news = get_sentiment(stock_name, sell_date, 'positive', 5)
                    selling_day_near_desc_sentiment = sell_sentiment_news.iloc[0]['desc_sentiment'] if not sell_sentiment_news.empty else None

                    # Calculate holding days and profit/loss percentage
                    hold_days = (sell_date - buy_date).days
                    profit_loss_pct = (sell_price - buy_price) / buy_price * 100

                    # Store the result
                    results.append({
                        'stock': stock_name,
                        'buy_date': buy_date,
                        'sell_date': sell_date,
                        'buy_price': buy_price,
                        'buying_day_near_desc_sentiment': buying_day_near_desc_sentiment,
                        'sell_price': sell_price,
                        'selling_day_near_desc_sentiment': selling_day_near_desc_sentiment,
                        'hold_days': hold_days,
                        'profit_loss_pct': profit_loss_pct
                    })
                    break  # Exit the loop after finding a sell signal

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('sentiment_33_strategy_results.csv', index=False)

# Calculate and print summary statistics
total_trades = len(results_df)
profitable_trades = len(results_df[results_df['profit_loss_pct'] > 0])
losing_trades = len(results_df[results_df['profit_loss_pct'] < 0])
avg_trades_per_month = total_trades / ((results_df['buy_date'].max() - results_df['buy_date'].min()).days / 30)
max_profitable_streak = max((results_df['profit_loss_pct'] > 0).astype(int).groupby((results_df['profit_loss_pct'] <= 0).astype(int).cumsum()).sum())
max_losing_streak = max((results_df['profit_loss_pct'] < 0).astype(int).groupby((results_df['profit_loss_pct'] >= 0).astype(int).cumsum()).sum())
hit_ratio = profitable_trades / total_trades if total_trades else 0
average_return_per_trade = results_df['profit_loss_pct'].mean()
avg_profit_per_trade = results_df[results_df['profit_loss_pct'] > 0]['profit_loss_pct'].mean()
avg_loss_per_trade = results_df[results_df['profit_loss_pct'] < 0]['profit_loss_pct'].mean()
risk_reward_ratio = abs(avg_profit_per_trade / avg_loss_per_trade) if avg_loss_per_trade else float('inf')
total_profit = results_df['profit_loss_pct'].sum()

# Identify stocks where sentiment did not align with the strategy
wrong_signals = results_df[
    ((results_df['profit_loss_pct'] > 0) & (results_df['buying_day_near_desc_sentiment'] != 'negative')) |
    ((results_df['profit_loss_pct'] < 0) & (results_df['selling_day_near_desc_sentiment'] == 'negative'))
]

# Print the results
print(f"Total Profit/Loss: {total_profit:.2f}%")
print(f"Total Trades: {total_trades}")
print(f"Profitable Trades: {profitable_trades}")
print(f"Losing Trades: {losing_trades}")
print(f"Average Trades per Month: {avg_trades_per_month:.2f}")
print(f"Max Profitable Streak: {max_profitable_streak}")
print(f"Max Losing Streak: {max_losing_streak}")
print(f"Hit Ratio: {hit_ratio:.2f}")
print(f"Risk-Reward Ratio: {risk_reward_ratio:.2f}")
print(f"Average Return per Trade: {average_return_per_trade:.2f}%")
print(f"Average Profit per Trade: {avg_profit_per_trade:.2f}%")
print(f"Average Loss per Trade: {avg_loss_per_trade:.2f}%")
print(f"Stocks with wrong signals based on sentiment:\n{wrong_signals[['stock', 'buy_date', 'sell_date']]}")
