import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def load_stock_data(stock_folder):
    """Load all stock data from CSV files in the specified folder"""
    stock_data = {}
    for file in Path(stock_folder).glob('*.csv'):
        df = pd.read_csv(file)
        # Convert date to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        # Ensure column names are lowercase
        df.columns = df.columns.str.lower()
        stock_data[file.stem] = df
    return stock_data

def safe_parse_datetime(dt_str):
    """Safely parse different datetime formats"""
    try:
        # First try parsing with milliseconds
        return pd.to_datetime(dt_str, format='%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            # Try parsing without milliseconds
            return pd.to_datetime(dt_str, format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                # Try parsing just the date
                return pd.to_datetime(dt_str, format='%Y-%m-%d')
            except ValueError:
                # Fall back to pandas default parser
                return pd.to_datetime(dt_str, errors='coerce')

def load_news_data(news_folder):
    """Load all news data from CSV files in the specified folder"""
    news_data = {}
    for file in Path(news_folder).glob('*.csv'):
        df = pd.read_csv(file)
        # Convert datetime column using the safe parser
        df['datetime'] = df['date'].apply(safe_parse_datetime)
        # Create date column for comparison
        df['date'] = df['datetime'].dt.date
        # Drop any rows where datetime conversion failed
        df = df.dropna(subset=['datetime'])
        news_data[file.stem] = df
    return news_data

def get_sentiment_signal(news_df, check_date, days_range, signal_type='buy'):
    """
    Check sentiment signals in the specified date range
    signal_type: 'buy' for positive/neutral, 'sell' for negative/neutral
    """
    if isinstance(check_date, pd.Timestamp):
        check_date = check_date.date()
    start_date = check_date - timedelta(days=days_range)
    
    # Filter news within date range
    mask = (news_df['date'] >= start_date) & (news_df['date'] <= check_date)
    relevant_news = news_df[mask].sort_values('datetime', ascending=False)
    
    if len(relevant_news) == 0:
        return None, None
    
    # Get closest news
    closest_news = relevant_news.iloc[0]
    
    # Check both title and description sentiment
    title_sent = str(closest_news['title_sentiment']).lower()
    desc_sent = str(closest_news['desc_sentiment']).lower()
    
    if signal_type == 'buy':
        # For buy signals, look for positive or neutral sentiment
        if title_sent in ['positive', 'neutral'] or desc_sent in ['positive', 'neutral']:
            return True, closest_news['datetime']
        return False, closest_news['datetime']
    else:
        # For sell signals, look for negative or neutral sentiment
        if title_sent in ['negative', 'neutral'] or desc_sent in ['negative', 'neutral']:
            return True, closest_news['datetime']
        return False, closest_news['datetime']

def check_buy_signal(stock_df):
    """Check if buy signal is triggered based on the strategy"""
    if len(stock_df) < 60:  # Need at least 60 days for SMA calculation
        return False
    
    latest = stock_df.iloc[-1]
    prev = stock_df.iloc[-2]
    
    sma20 = calculate_sma(stock_df['close'], 20)
    sma60 = calculate_sma(stock_df['close'], 60)
    
    conditions = (
        latest['close'] >= 100 and
        sma20.iloc[-1] > sma60.iloc[-1] and
        sma20.iloc[-2] <= sma60.iloc[-2]
    )
    
    return conditions

def check_sell_signal(stock_df):
    """Check if sell signal is triggered based on the strategy"""
    if len(stock_df) < 200:  # Need at least 200 days for SMA calculation
        return False
    
    latest = stock_df.iloc[-1]
    prev = stock_df.iloc[-2]
    
    sma200 = calculate_sma(stock_df['close'], 200)
    
    conditions = (
        latest['close'] < sma200.iloc[-1] and
        prev['close'] >= sma200.iloc[-2] and
        latest['volume'] > 100000
    )
    
    return conditions

def backtest_strategy(stock_data, news_data, max_hold_days_range, max_concurrent_positions=10):
    """
    Backtest the trading strategy with grid search for optimal holding period
    """
    results = []
    current_positions = []
    capital_per_stock = 200000  # 2 lacs per stock
    
    for hold_days in range(2, max_hold_days_range + 1):
        for symbol in stock_data:
            if symbol not in news_data:
                continue
                
            stock_df = stock_data[symbol].copy()
            news_df = news_data[symbol]
            
            for i in range(200, len(stock_df) - hold_days):
                current_date = stock_df.index[i]
                
                # Skip if maximum positions reached
                if len(current_positions) >= max_concurrent_positions:
                    continue
                
                # Check buy signal
                df_slice = stock_df.iloc[:i+1]
                if check_buy_signal(df_slice):
                    sentiment_ok, news_date = get_sentiment_signal(news_df, current_date, 30, 'buy')
                    
                    if sentiment_ok:
                        buy_price = stock_df.iloc[i]['close']
                        sell_date = stock_df.index[i + hold_days]
                        sell_price = stock_df.loc[sell_date]['close']
                        
                        # Check if there's a sell signal before max hold period
                        sell_news_date = None
                        for j in range(i + 1, i + hold_days):
                            df_slice_sell = stock_df.iloc[:j+1]
                            if check_sell_signal(df_slice_sell):
                                sell_sentiment_ok, sell_news_date = get_sentiment_signal(
                                    news_df, stock_df.index[j], 30, 'sell'
                                )
                                if sell_sentiment_ok:
                                    sell_date = stock_df.index[j]
                                    sell_price = stock_df.iloc[j]['close']
                                    break
                        
                        profit_loss_pct = ((sell_price - buy_price) / buy_price) * 100
                        
                        # Track position
                        current_positions.append({
                            'symbol': symbol,
                            'entry_date': current_date,
                            'entry_price': buy_price
                        })
                        
                        results.append({
                            'stock': symbol,
                            'buy_date': current_date.strftime('%Y-%m-%d'),
                            'sell_date': sell_date.strftime('%Y-%m-%d'),
                            'buy_price': buy_price,
                            'buying_day_near_sentiment': news_date.strftime('%Y-%m-%d %H:%M:%S') if news_date is not None else None,
                            'sell_price': sell_price,
                            'selling_day_near_sentiment': sell_news_date.strftime('%Y-%m-%d %H:%M:%S') if sell_news_date is not None else None,
                            'hold_days': (sell_date - current_date).days,
                            'profit_loss_pct': profit_loss_pct
                        })
                        
                        # Remove position after sell
                        current_positions = [pos for pos in current_positions if pos['symbol'] != symbol]
    
    return pd.DataFrame(results)

def analyze_results(results_df):
    """Calculate and print trading statistics"""
    if len(results_df) == 0:
        return {"error": "No trades found"}
    
    stats = {
        'Total Trades': len(results_df),
        'Profitable Trades': len(results_df[results_df['profit_loss_pct'] > 0]),
        'Losing Trades': len(results_df[results_df['profit_loss_pct'] <= 0]),
        'Avg Trades per Month': len(results_df) / ((pd.to_datetime(results_df['sell_date']).max() - 
                                                   pd.to_datetime(results_df['buy_date']).min()).days / 30),
        'Hit Ratio': len(results_df[results_df['profit_loss_pct'] > 0]) / len(results_df) * 100,
        'Average Return per Trade': results_df['profit_loss_pct'].mean(),
        'Avg Profit per Trade': results_df[results_df['profit_loss_pct'] > 0]['profit_loss_pct'].mean(),
        'Avg Loss per Trade': results_df[results_df['profit_loss_pct'] <= 0]['profit_loss_pct'].mean(),
        'Total Profit/Loss %': results_df['profit_loss_pct'].sum(),
    }
    
    # Calculate streaks
    profit_losses = (results_df['profit_loss_pct'] > 0).astype(int)
    streaks = profit_losses.groupby((profit_losses != profit_losses.shift()).cumsum()).count()
    
    stats['Max Profitable Streak'] = streaks[profit_losses == 1].max() if len(streaks[profit_losses == 1]) > 0 else 0
    stats['Max Losing Streak'] = streaks[profit_losses == 0].max() if len(streaks[profit_losses == 0]) > 0 else 0
    
    # Calculate Risk-Reward Ratio
    avg_profit = abs(results_df[results_df['profit_loss_pct'] > 0]['profit_loss_pct'].mean())
    avg_loss = abs(results_df[results_df['profit_loss_pct'] <= 0]['profit_loss_pct'].mean())
    stats['Risk-Reward Ratio'] = avg_profit / avg_loss if avg_loss != 0 else float('inf')
    
    return stats

def main(stock_folder, news_folder, output_file):
    """Main function to run the strategy"""
    try:
        # Load data
        print("Loading stock data...")
        stock_data = load_stock_data(stock_folder)
        
        print("Loading news data...")
        news_data = load_news_data(news_folder)
        
        # Run backtest
        print("Running backtest...")
        results_df = backtest_strategy(stock_data, news_data, max_hold_days_range=60)
        
        # Save results
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Calculate and print statistics
        stats = analyze_results(results_df)
        if "error" in stats:
            print(stats["error"])
            return
        
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
        
        # Calculate last 2 years performance
        two_years_ago = datetime.now() - timedelta(days=730)
        recent_results = results_df[pd.to_datetime(results_df['buy_date']) >= two_years_ago]
        recent_stats = analyze_results(recent_results)
        
        print("\nLast 2 Years Performance:")
        print(f"Total Profit/Loss %: {recent_stats['Total Profit/Loss %']:.2f}")
        print(f"Total Profit Generated: ₹{(recent_stats['Total Profit/Loss %'] * 2000000 / 100):.2f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Replace these paths with actual paths
    stock_folder = "best_performing_stocks"
    news_folder = "news_sentiments"
    output_file = "sentiment_1_trading_results.csv"
    
    main(stock_folder, news_folder, output_file)