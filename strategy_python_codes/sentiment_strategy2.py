import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Dict, Tuple
import itertools
import warnings
warnings.filterwarnings('ignore')

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def load_stock_data(stock_folder: str) -> Dict[str, pd.DataFrame]:
    """Load all stock data from CSV files"""
    stock_data = {}
    for file in os.listdir(stock_folder):
        if file.endswith('.csv'):
            stock_name = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(stock_folder, file))
            df['date'] = pd.to_datetime(df['date'])
            stock_data[stock_name] = df
    return stock_data

def parse_datetime(date_str: str) -> pd.Timestamp:
    """Parse datetime strings in various formats"""
    try:
        # First try parsing with milliseconds
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            # Then try without milliseconds
            return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            # Finally try just the date
            return pd.to_datetime(date_str, format='%Y-%m-%d')

def load_news_data(news_folder: str) -> Dict[str, pd.DataFrame]:
    """Load all news data from CSV files with mixed datetime formats"""
    news_data = {}
    for file in os.listdir(news_folder):
        if file.endswith('.csv'):
            stock_name = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(news_folder, file))
            
            # Handle mixed datetime formats
            df['date'] = df['date'].apply(parse_datetime)
            
            # Sort by date
            df = df.sort_values('date')
            news_data[stock_name] = df
    return news_data

def check_buy_condition(row: pd.Series, prev_row: pd.Series, volume_sma: float) -> bool:
    """Check if buy conditions are met"""
    return (row['volume'] > volume_sma * 5 and 
            (row['close'] > prev_row['close'] * 1.05 or 
             row['close'] < prev_row['close'] * 0.95))

def check_sell_condition(row: pd.Series, prev_row: pd.Series, sma_200: float, prev_sma_200: float) -> bool:
    """Check if sell conditions are met"""
    return (row['close'] < sma_200 and 
            prev_row['close'] >= prev_sma_200 and 
            row['volume'] > 100000)

def get_market_time_range(date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get market start and end times for a given date"""
    market_start = pd.Timestamp(f"{date.date()} 09:15:00")
    market_end = pd.Timestamp(f"{date.date()} 15:30:00")
    return market_start, market_end

def get_nearby_sentiment(date: pd.Timestamp, news_df: pd.DataFrame, 
                        days_range: Tuple[int, int]) -> Tuple[str, pd.Timestamp]:
    """Get the closest sentiment within specified date range"""
    start_date = date - timedelta(days=days_range[1])
    end_date = date + timedelta(days=days_range[0])
    
    # Get market hours for the date
    market_start, market_end = get_market_time_range(date)
    
    # Filter news within the date range
    mask = (news_df['date'] >= start_date) & (news_df['date'] <= end_date)
    nearby_news = news_df[mask].copy()
    
    if len(nearby_news) == 0:
        return None, None
    
    # Find closest news to the date
    nearby_news['time_diff'] = abs(nearby_news['date'] - date)
    closest_news = nearby_news.loc[nearby_news['time_diff'].idxmin()]
    
    return closest_news['desc_sentiment'], closest_news['date']

def backtest_strategy(stock_data: Dict[str, pd.DataFrame], 
                     news_data: Dict[str, pd.DataFrame],
                     max_hold_days: int = 60,
                     max_positions: int = 10,
                     capital_per_stock: float = 200000) -> pd.DataFrame:
    """
    Backtest the trading strategy with news sentiment
    """
    results = []
    current_positions = []
    
    for stock_name, stock_df in stock_data.items():
        print(f"Processing stock: {stock_name}")
        
        if stock_name not in news_data:
            print(f"No news data found for {stock_name}, skipping...")
            continue
            
        news_df = news_data[stock_name]
        stock_df = stock_df.sort_values('date')
        
        # Calculate indicators
        stock_df['volume_sma_10'] = calculate_sma(stock_df['volume'], 10)
        stock_df['close_sma_200'] = calculate_sma(stock_df['close'], 200)
        
        for i in range(1, len(stock_df)):
            current_row = stock_df.iloc[i]
            prev_row = stock_df.iloc[i-1]
            
            # Check buy condition
            if len(current_positions) < max_positions and check_buy_condition(current_row, prev_row, stock_df.iloc[i]['volume_sma_10']):
                sentiment, news_date = get_nearby_sentiment(
                    current_row['date'], 
                    news_df, 
                    (2, 30)
                )
                
                if sentiment == 'negative':
                    best_profit = float('-inf')
                    best_trade = None
                    
                    for hold_days in range(2, max_hold_days + 1):
                        if i + hold_days >= len(stock_df):
                            continue
                            
                        sell_row = stock_df.iloc[i + hold_days]
                        sell_sentiment, sell_news_date = get_nearby_sentiment(
                            sell_row['date'],
                            news_df,
                            (2, 30)
                        )
                        
                        if check_sell_condition(sell_row, stock_df.iloc[i + hold_days - 1],
                                             stock_df.iloc[i + hold_days]['close_sma_200'],
                                             stock_df.iloc[i + hold_days - 1]['close_sma_200']) and \
                           sell_sentiment in ['positive', 'neutral']:
                            
                            profit_loss_pct = ((sell_row['close'] - current_row['close']) / 
                                             current_row['close'] * 100)
                            
                            if profit_loss_pct > best_profit:
                                best_profit = profit_loss_pct
                                best_trade = {
                                    'stock': stock_name,
                                    'buy_date': current_row['date'],
                                    'sell_date': sell_row['date'],
                                    'buy_price': current_row['close'],
                                    'buying_day_near_desc_sentiment': sentiment,
                                    'sell_price': sell_row['close'],
                                    'selling_day_near_desc_sentiment': sell_sentiment,
                                    'hold_days': hold_days,
                                    'profit_loss_pct': profit_loss_pct
                                }
                    
                    if best_trade is not None:
                        results.append(best_trade)
                        current_positions.append(stock_name)
                        
            # Remove from current positions if max hold days reached
            current_positions = [pos for pos in current_positions 
                               if (current_row['date'] - stock_df['date'].iloc[i-len(current_positions)]).days <= max_hold_days]
    
    results_df = pd.DataFrame(results)
    return results_df

def analyze_results(results_df: pd.DataFrame) -> Dict:
    """Analyze trading results and calculate metrics"""
    if len(results_df) == 0:
        print("No trades found to analyze.")
        return {}
        
    metrics = {
        'total_trades': len(results_df),
        'profitable_trades': len(results_df[results_df['profit_loss_pct'] > 0]),
        'losing_trades': len(results_df[results_df['profit_loss_pct'] <= 0]),
        'avg_trades_per_month': len(results_df) / ((results_df['sell_date'].max() - 
                                                   results_df['buy_date'].min()).days / 30),
        'hit_ratio': len(results_df[results_df['profit_loss_pct'] > 0]) / len(results_df) * 100,
        'avg_return_per_trade': results_df['profit_loss_pct'].mean(),
        'avg_profit_per_trade': results_df[results_df['profit_loss_pct'] > 0]['profit_loss_pct'].mean(),
        'avg_loss_per_trade': results_df[results_df['profit_loss_pct'] <= 0]['profit_loss_pct'].mean(),
        'total_profit_loss_pct': results_df['profit_loss_pct'].sum(),
    }
    
    # Calculate streaks
    profit_loss_series = (results_df['profit_loss_pct'] > 0).astype(int)
    metrics['max_profitable_streak'] = max(len(list(g)) for k, g in itertools.groupby(profit_loss_series) if k == 1)
    metrics['max_losing_streak'] = max(len(list(g)) for k, g in itertools.groupby(profit_loss_series) if k == 0)
    
    # Calculate risk-reward ratio
    avg_profit = abs(results_df[results_df['profit_loss_pct'] > 0]['profit_loss_pct'].mean())
    avg_loss = abs(results_df[results_df['profit_loss_pct'] <= 0]['profit_loss_pct'].mean())
    metrics['risk_reward_ratio'] = avg_profit / avg_loss if avg_loss != 0 else float('inf')
    
    return metrics

def main():
    # Set paths to your data folders
    stock_folder = 'daily_data_stocks'
    news_folder = 'news_sentiments'
    output_file = 'sentiment_2_trading_results.csv'
    
    try:
        # Load data
        print("Loading stock data...")
        stock_data = load_stock_data(stock_folder)
        print(f"Loaded {len(stock_data)} stocks")
        
        print("Loading news data...")
        news_data = load_news_data(news_folder)
        print(f"Loaded news data for {len(news_data)} stocks")
        
        # Run backtest
        print("Running backtest...")
        results_df = backtest_strategy(stock_data, news_data)
        
        # Save results
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Analyze and print metrics
        print("\nAnalyzing results...")
        metrics = analyze_results(results_df)
        
        print("\nTrading Strategy Results:")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
        
        # Calculate last 2 years performance
        two_years_ago = pd.Timestamp.now() - pd.DateOffset(years=2)
        recent_results = results_df[results_df['buy_date'] >= two_years_ago]
        recent_metrics = analyze_results(recent_results)
        
        print("\nLast 2 Years Performance:")
        print("=" * 50)
        print(f"Total Profit/Loss %: {recent_metrics['total_profit_loss_pct']:.2f}%")
        print(f"Number of Trades: {recent_metrics['total_trades']}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()