import os
import pandas as pd
import numpy as np
import time
from datetime import timedelta

class StockAnalyzer:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.stocks_data = {}
        self.load_data()

    def load_data(self):
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.csv'):
                stock_name = filename.split('.')[0]
                df = pd.read_csv(os.path.join(self.data_folder, filename))
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                self.stocks_data[stock_name] = df

    def calculate_pivot_points(self, df, left_bars=15, right_bars=15):
        df['high_pivot'] = df['high'].rolling(window=left_bars+right_bars+1, center=True).apply(lambda x: x[left_bars] == max(x), raw=True)
        df['low_pivot'] = df['low'].rolling(window=left_bars+right_bars+1, center=True).apply(lambda x: x[left_bars] == min(x), raw=True)

        
        df['resistance'] = np.where(df['high_pivot'], df['high'], np.nan)
        df['support'] = np.where(df['low_pivot'], df['low'], np.nan)
        
        df['resistance'] = df['resistance'].fillna(method='ffill')
        df['support'] = df['support'].fillna(method='ffill')
        
        return df

    def calculate_volume_osc(self, df):
        df['volume_ema_short'] = df['volume'].ewm(span=5, adjust=False).mean()
        df['volume_ema_long'] = df['volume'].ewm(span=10, adjust=False).mean()
        df['volume_osc'] = 100 * (df['volume_ema_short'] - df['volume_ema_long']) / df['volume_ema_long']
        return df

    def analyze_stock(self, stock_name):
        df = self.stocks_data[stock_name].copy()
        df = self.calculate_pivot_points(df)
        df = self.calculate_volume_osc(df)
        return df

class TradingSystem:
    def __init__(self, analyzer, initial_capital=2000000):
        self.analyzer = analyzer
        self.capital = initial_capital
        self.positions = {}
        self.max_positions = 10
        self.position_size = 200000  # 2 lakhs
        self.stop_loss_percent = 0.10
        self.trades = []
        self.volume_thresh = 20

    def check_buy_signal(self, row, prev_row):
        return (row['close'] > row['resistance'] and 
                prev_row['close'] <= prev_row['resistance'] and 
                row['volume_osc'] > self.volume_thresh and
                not (row['open'] - row['low'] > row['close'] - row['open']))

    def check_sell_signal(self, row, prev_row, entry_price):
        return ((row['close'] < row['support'] and 
                 prev_row['close'] >= prev_row['support'] and 
                #  row['volume_osc'] > self.volume_thresh and
                 not (row['open'] - row['close'] < row['high'] - row['open'])) or
                row['close'] <= entry_price * (1 - self.stop_loss_percent))

    def execute_trade(self, stock_name, action, date, price, quantity):
        if action == 'BUY':
            cost = price * quantity
            self.capital -= cost
            self.positions[stock_name] = {'quantity': quantity, 'entry_price': price, 'entry_date': date}
        elif action == 'SELL':
            revenue = price * quantity
            self.capital += revenue
            entry_price = self.positions[stock_name]['entry_price']
            entry_date = self.positions[stock_name]['entry_date']
            del self.positions[stock_name]
            
            profit = revenue - (entry_price * quantity)
            profit_percent = (price - entry_price) / entry_price
            
            self.trades.append({
                'stock': stock_name,
                'buy_date': entry_date,
                'sell_date': date,
                'buy_price': entry_price,
                'sell_price': price,
                'quantity': quantity,
                'profit': profit,
                'profit_percent': profit_percent
            })

    def run_simulation(self):
        for stock_name, stock_data in self.analyzer.stocks_data.items():
            df = self.analyzer.analyze_stock(stock_name)
            
            for i in range(1, len(df)):
                current_row = df.iloc[i]
                prev_row = df.iloc[i-1]
                current_price = current_row['close']
                current_date = current_row['date']
                
                if stock_name in self.positions:
                    if self.check_sell_signal(current_row, prev_row, self.positions[stock_name]['entry_price']):
                        self.execute_trade(stock_name, 'SELL', current_date, current_price, self.positions[stock_name]['quantity'])
                elif len(self.positions) < self.max_positions:
                    if self.check_buy_signal(current_row, prev_row):
                        quantity = min(self.position_size // current_price, (self.capital * 0.9) // current_price)
                        if quantity > 0:
                            self.execute_trade(stock_name, 'BUY', current_date, current_price, quantity)

    def generate_report(self):
        total_profit = sum(trade['profit'] for trade in self.trades)
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade['profit'] > 0)
        
        report = {
            'total_profit': total_profit,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'hit_ratio': winning_trades / total_trades if total_trades > 0 else 0,
            'average_profit_percent': sum(trade['profit_percent'] for trade in self.trades) / total_trades if total_trades > 0 else 0,
            'yearly_gain': total_profit / (self.trades[-1]['sell_date'] - self.trades[0]['buy_date']).days * 365 if self.trades else 0
        }
        
        return report

    def save_results(self):
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv('trades_report_best.csv', index=False)

        with open('trading_summary_best.txt', 'w') as f:
            report = self.generate_report()
            f.write("Trading Summary:\n")
            for key, value in report.items():
                f.write(f"{key}: {value}\n")

# Usage
data_folder = 'best_performing_stocks'
start_time = time.time()

analyzer = StockAnalyzer(data_folder)
trading_system = TradingSystem(analyzer)

trading_system.run_simulation()
trading_system.save_results()

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.2f} seconds")