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

    def find_support_resistance(self, df, window=14):
        df['min_low'] = df['low'].rolling(window=window, center=True).min()
        df['max_high'] = df['high'].rolling(window=window, center=True).max()
        
        supports = df[df['low'] == df['min_low']]['low'].tolist()
        resistances = df[df['high'] == df['max_high']]['high'].tolist()
        
        return supports, resistances

    def analyze_stock(self, stock_name, current_date):
        df = self.stocks_data[stock_name]
        df = df[df['date'] <= current_date]
        
        short_term_df = df.tail(90)  # last 90 days
        long_term_df = df.tail(450)  # last 450 days
        
        short_term_supports, short_term_resistances = self.find_support_resistance(short_term_df)
        long_term_supports, long_term_resistances = self.find_support_resistance(long_term_df)
        
        return {
            'short_term_supports': short_term_supports,
            'short_term_resistances': short_term_resistances,
            'long_term_supports': long_term_supports,
            'long_term_resistances': long_term_resistances
        }

class TradingSystem:
    def __init__(self, analyzer, initial_capital=1000000):
        self.analyzer = analyzer
        self.capital = initial_capital
        self.positions = {}
        self.max_positions = 5
        self.position_size = 200000  # 2 lakhs
        self.stop_loss_percent = 0.05
        self.trades = []

    def check_buy_signal(self, stock_name, current_date, current_price):
        analysis = self.analyzer.analyze_stock(stock_name, current_date)
        all_supports = analysis['short_term_supports'] + analysis['long_term_supports']
        
        for support_price in all_supports:
            if abs(current_price - support_price) / support_price <= 0.01:  # Within 1% of support
                return True
        return False

    def check_sell_signal(self, stock_name, current_date, current_price, entry_price):
        analysis = self.analyzer.analyze_stock(stock_name, current_date)
        all_resistances = analysis['short_term_resistances'] + analysis['long_term_resistances']
        
        for resistance_price in all_resistances:
            if abs(current_price - resistance_price) / resistance_price <= 0.01:  # Within 1% of resistance
                return True
        
        if current_price <= entry_price * (1 - self.stop_loss_percent):  # Stop loss hit
            return True
        return False

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
        all_dates = sorted(set(date for stock_data in self.analyzer.stocks_data.values() for date in stock_data['date']))
        
        for current_date in all_dates:
            for stock_name, stock_data in self.analyzer.stocks_data.items():
                if current_date in stock_data['date'].values:
                    current_price = stock_data[stock_data['date'] == current_date]['close'].values[0]
                    
                    if stock_name in self.positions:
                        if self.check_sell_signal(stock_name, current_date, current_price, self.positions[stock_name]['entry_price']):
                            self.execute_trade(stock_name, 'SELL', current_date, current_price, self.positions[stock_name]['quantity'])
                    elif len(self.positions) < self.max_positions:
                        if self.check_buy_signal(stock_name, current_date, current_price):
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
        # Save trades to CSV file
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv('trades_report_best_performing.csv', index=False)

        # Save summary to txt file
        with open('trading_summary_best_performing.txt', 'w') as f:
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