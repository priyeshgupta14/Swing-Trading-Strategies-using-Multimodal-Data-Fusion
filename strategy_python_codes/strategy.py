import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Indicators:
    @staticmethod
    def sma(data, period):
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data, period):
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data, period):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def adx(high, low, close, period):
        tr = pd.concat([high - low, 
                        abs(high - close.shift(1)), 
                        abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * Indicators.ema(pd.Series(plus_dm), period) / atr
        minus_di = 100 * Indicators.ema(pd.Series(minus_dm), period) / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = Indicators.ema(dx, period)
        return adx

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

    def prepare_data(self, df):
        df['sma_volume_10'] = Indicators.sma(df['volume'], 10)
        df['ema_close_50'] = Indicators.ema(df['close'], 50)
        df['ema_close_200'] = Indicators.ema(df['close'], 200)
        df['rsi_3'] = Indicators.rsi(df['close'], 3)
        df['adx_14'] = Indicators.adx(df['high'], df['low'], df['close'], 14)
        
        # df['weekly_close'] = df['close'].rolling(window=5).shift(4)
        # df['monthly_close'] = df['close'].rolling(window=22).shift(21)
        
        df['max_10_high'] = df['high'].rolling(window=10).max()
        
        return df

    def check_buy_signal(self, row, prev_row, prev2_row):
        condition1 = (
            row['close'] > prev_row['max_10_high'] and
            prev_row['close'] <= prev2_row['max_10_high'] and
            prev2_row['high'] < prev_row['max_10_high'] and
            row['close'] > 50 and
            row['volume'] > row['sma_volume_10'] and
            row['volume'] > prev_row['volume'] and
            row['adx_14'] > 20
            # row['weekly_close'] > row['ema_close_50']
        )
        
        # condition2 = (
        #     row['low'] > prev_row['close'] and
        #     row['close'] > row['open'] and
        #     row['monthly_close'] > prev_row['open'] and
        #     row['weekly_close'] > prev_row['open'] and
        #     row['weekly_close'] > prev_row['close'] and
        #     row['monthly_close'] > prev_row['close']
        # )
        
        return condition1

    def check_sell_signal(self, row, prev_row, buy_signal_row):
        condition1 = (
            prev_row['ema_close_50'] < prev_row['ema_close_200'] and
            prev_row['close'] < prev_row['ema_close_50'] and
            prev_row['rsi_3'] < 20 and
            prev_row['close'] < prev_row['high'] and
            prev_row['close'] < prev_row['open']
        )
        
        condition2 = (
            row['open'] - row['close'] > (row['high'] - row['low']) * 0.5 and
            row['close'] - row['low'] < (row['high'] - row['low']) * 0.15
        )
        
        condition3 = (
            row['close'] < buy_signal_row['low'] or
            row['close'] < buy_signal_row['close'] * 0.95 or
            row['rsi_3'] < 45
        )
        
        return condition1 or condition2 or condition3

class TradingSystem:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.trades = []

    def run_simulation(self):
        for stock_name, df in self.analyzer.stocks_data.items():
            df = self.analyzer.prepare_data(df)
            in_position = False
            buy_signal_row = None
            
            for i in range(2, len(df)):
                if not in_position:
                    if self.analyzer.check_buy_signal(df.iloc[i], df.iloc[i-1], df.iloc[i-2]):
                        buy_signal_row = df.iloc[i]
                        in_position = True
                        entry_date = df.iloc[i]['date']
                        entry_price = df.iloc[i]['close']
                else:
                    if self.analyzer.check_sell_signal(df.iloc[i], df.iloc[i-1], buy_signal_row):
                        exit_date = df.iloc[i]['date']
                        exit_price = df.iloc[i]['close']
                        profit_pct = (exit_price - entry_price) / entry_price * 100
                        
                        self.trades.append({
                            'stock': stock_name,
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit_pct': profit_pct
                        })
                        
                        in_position = False
                        buy_signal_row = None

    def generate_report(self):
        if not self.trades:
            return "No trades were executed."

        df = pd.DataFrame(self.trades)
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df['trade_duration'] = (df['exit_date'] - df['entry_date']).dt.days

        total_trades = len(df)
        profitable_trades = len(df[df['profit_pct'] > 0])
        losing_trades = total_trades - profitable_trades

        start_date = df['entry_date'].min()
        end_date = df['exit_date'].max()
        total_days = (end_date - start_date).days
        avg_trades_per_month = total_trades / (total_days / 30)

        df['cumulative_profit'] = df['profit_pct'].cumsum()
        df['drawdown'] = df['cumulative_profit'] - df['cumulative_profit'].cummax()
        max_drawdown = df['drawdown'].min()

        df['streak'] = (df['profit_pct'] > 0).groupby((df['profit_pct'] > 0).ne(df['profit_pct'].shift() > 0).cumsum()).cumcount() + 1
        max_win_streak = df[df['profit_pct'] > 0]['streak'].max()
        max_lose_streak = df[df['profit_pct'] <= 0]['streak'].max()

        hit_ratio = profitable_trades / total_trades if total_trades > 0 else 0
        avg_profit = df[df['profit_pct'] > 0]['profit_pct'].mean()
        avg_loss = df[df['profit_pct'] <= 0]['profit_pct'].mean()
        risk_reward_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')

        total_return = df['profit_pct'].sum()
        avg_return_per_trade = df['profit_pct'].mean()

        report = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'avg_trades_per_month': avg_trades_per_month,
            'max_win_streak': max_win_streak,
            'max_lose_streak': max_lose_streak,
            'hit_ratio': hit_ratio,
            'risk_reward_ratio': risk_reward_ratio,
            'avg_return_per_trade': avg_return_per_trade,
            'avg_profit_per_trade': avg_profit,
            'avg_loss_per_trade': avg_loss,
            'total_return': total_return,
            'max_drawdown': max_drawdown
        }

        return report

    def save_results(self):
        # Save trades to CSV file
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv('trades_report_bestss1.csv', index=False)

        # Save summary to txt file
        with open('trading_summary_bestss1.txt', 'w') as f:
            report = self.generate_report()
            f.write("Trading Summary:\n")
            for key, value in report.items():
                f.write(f"{key}: {value}\n")

        # Save stocks where strategy works
        successful_stocks = trades_df[trades_df['profit_pct'] > 0]['stock'].unique()
        with open('successful_stocks1.csv', 'w') as f:
            f.write("stock,entry_date,exit_date,profit_pct\n")
            for stock in successful_stocks:
                stock_trades = trades_df[trades_df['stock'] == stock]
                for _, trade in stock_trades.iterrows():
                    f.write(f"{trade['stock']},{trade['entry_date']},{trade['exit_date']},{trade['profit_pct']}\n")

# Usage
data_folder = 'best_performing_stocks'
analyzer = StockAnalyzer(data_folder)
trading_system = TradingSystem(analyzer)

trading_system.run_simulation()
trading_system.save_results()

print("Simulation completed. Results saved to CSV and TXT files.")