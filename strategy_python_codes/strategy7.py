import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class TradingStrategy:
    def __init__(self, max_holdings: int = 10, capital_per_stock: int = 200000):
        self.max_holdings = max_holdings
        self.capital_per_stock = capital_per_stock
        self.current_holdings = []
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate required technical indicators."""
        # Calculate 10-day high
        df['high_10'] = df['high'].rolling(window=10).max()
        
        # Calculate volume SMA
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        
        # Calculate ADX
        def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
            plus_dm = high.diff()
            minus_dm = low.diff()
            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift(1)))
            tr3 = pd.DataFrame(abs(low - close.shift(1)))
            frames = [tr1, tr2, tr3]
            tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            return adx
        
        df['adx_14'] = calculate_adx(df['high'], df['low'], df['close'])
        
        # Calculate EMA 50
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Calculate RSI
        def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = calculate_rsi(df['close'])
        
        return df
    
    def check_buy_signal(self, row: pd.Series, prev_row: pd.Series, prev2_row: pd.Series) -> bool:
        """Check if buy conditions are met."""
        try:
            conditions = [
                row['close'] > prev_row['high_10'],
                prev_row['close'] <= prev2_row['high_10'],
                prev2_row['high'] < prev_row['high_10'],
                row['close'] > 50,
                row['volume'] > row['volume_sma_10'],
                row['volume'] > prev_row['volume'],
                row['adx_14'] > 20,
                row['close'] > row['ema_50']
            ]
            return all(conditions)
        except:
            return False
    
    def check_sell_signal(self, row: pd.Series, buy_price: float, buy_low: float) -> bool:
        """Check if sell conditions are met."""
        try:
            return (row['close'] < buy_low) or \
                   (row['close'] < (buy_price * 0.95) and row['rsi_14'] < 45)
        except:
            return False
    
    def find_best_holding_period(self, df: pd.DataFrame, hold_days_range: Tuple[int, int]) -> Tuple[int, List[Dict]]:
        """Find the best holding period for a single stock."""
        best_profit = float('-inf')
        best_trades = []
        best_hold_days = 0
        
        for hold_days in range(hold_days_range[0], hold_days_range[1] + 1):
            trades = []
            for i in range(2, len(df)-1):
                if self.check_buy_signal(df.iloc[i], df.iloc[i-1], df.iloc[i-2]):
                    buy_price = df.iloc[i]['close']
                    buy_date = df.index[i]
                    buy_low = df.iloc[i]['low']
                    
                    for j in range(i+1, min(i+hold_days+1, len(df))):
                        if self.check_sell_signal(df.iloc[j], buy_price, buy_low):
                            sell_price = df.iloc[j]['close']
                            sell_date = df.index[j]
                            profit_pct = ((sell_price - buy_price) / buy_price) * 100
                            
                            trade = {
                                'buy_date': buy_date,
                                'sell_date': sell_date,
                                'buy_price': buy_price,
                                'sell_price': sell_price,
                                'hold_days': (sell_date - buy_date).days,
                                'profit_loss_pct': profit_pct
                            }
                            trades.append(trade)
                            break
                            
                        if j == min(i+hold_days, len(df)-1):
                            sell_price = df.iloc[j]['close']
                            sell_date = df.index[j]
                            profit_pct = ((sell_price - buy_price) / buy_price) * 100
                            
                            trade = {
                                'buy_date': buy_date,
                                'sell_date': sell_date,
                                'buy_price': buy_price,
                                'sell_price': sell_price,
                                'hold_days': (sell_date - buy_date).days,
                                'profit_loss_pct': profit_pct
                            }
                            trades.append(trade)
                    
                    i = j
            
            if trades:
                total_profit = sum(trade['profit_loss_pct'] for trade in trades)
                if total_profit > best_profit:
                    best_profit = total_profit
                    best_trades = trades
                    best_hold_days = hold_days
        
        return best_hold_days, best_trades

def main(data_folder: str, output_file: str, hold_days_range: Tuple[int, int] = (2, 60)):
    """Main function to run the strategy backtesting."""
    strategy = TradingStrategy()
    stock_results = []
    
    for filename in os.listdir(data_folder):
        if not filename.endswith('.csv'):
            continue
            
        stock_name = filename.replace('.csv', '')
        print(f"Processing {stock_name}...")
        
        df = pd.read_csv(os.path.join(data_folder, filename), parse_dates=['date'], index_col='date')
        
        # Get last 3 years of data
        three_years_ago = datetime.now() - timedelta(days=3*365)
        df = df[df.index >= three_years_ago]
        
        if len(df) < 50:  # Skip stocks with insufficient data
            continue
            
        df = strategy.calculate_indicators(df)
        best_hold_days, trades = strategy.find_best_holding_period(df, hold_days_range)
        
        if trades:
            total_profit = sum(trade['profit_loss_pct'] for trade in trades)
            stock_results.append({
                'stock': stock_name,
                'best_hold_days': best_hold_days,
                'total_profit_pct': total_profit,
                'num_trades': len(trades),
                'trades': trades
            })
    
    # Sort stocks by total profit
    stock_results.sort(key=lambda x: x['total_profit_pct'], reverse=True)
    
    # Save best performing trades to CSV
    all_trades = []
    for result in stock_results:
        for trade in result['trades']:
            trade['stock'] = result['stock']
            all_trades.append(trade)
    
    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df = trades_df[['stock', 'buy_date', 'sell_date', 'buy_price', 
                             'sell_price', 'hold_days', 'profit_loss_pct']]
        trades_df.to_csv(output_file, index=False)
        
        # Print top performing stocks and their statistics
        print("\nTop Performing Stocks:")
        for i, result in enumerate(stock_results[:10], 1):
            print(f"\n{i}. {result['stock']}:")
            print(f"   Best holding period: {result['best_hold_days']} days")
            print(f"   Total profit: {result['total_profit_pct']:.2f}%")
            print(f"   Number of trades: {result['num_trades']}")
            
            # Calculate additional statistics for this stock
            stock_trades = pd.DataFrame(result['trades'])
            profitable_trades = len(stock_trades[stock_trades['profit_loss_pct'] > 0])
            avg_profit = stock_trades[stock_trades['profit_loss_pct'] > 0]['profit_loss_pct'].mean()
            avg_loss = stock_trades[stock_trades['profit_loss_pct'] <= 0]['profit_loss_pct'].mean()
            
            print(f"   Profitable trades: {profitable_trades}/{result['num_trades']}")
            print(f"   Hit ratio: {(profitable_trades/result['num_trades'])*100:.2f}%")
            print(f"   Average profit per winning trade: {avg_profit:.2f}%")
            print(f"   Average loss per losing trade: {avg_loss:.2f}%")

if __name__ == "__main__":
    main("daily_data_stocks", "strategy7_results.csv")