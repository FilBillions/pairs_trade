import sys
sys.path.append(".")
import csv
import random
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

from contents.pairs_trade_obj import Pairs_Trade
from contents.simple_return import Simple_Return

class Backtest():
    def __init__(self):
        with open('inputs.csv', 'r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            self.ticker_list = [row['Ticker'] for row in rows if row['Ticker']]
            self.ticker_list_2 = [row['Ticker 2'] for row in rows if row['Ticker 2']]
        #Argument checks
        if len(sys.argv) != 4:
            print("Error: Incorrect number of arguments.")
            print("Usage: python backtest.py <number of iterations> <algorithm method 1 or 2> <backtest cash or equity>")
            sys.exit(1)
        if len(sys.argv) == 4:
            number_check = None
            try:
                number_check = int(sys.argv[1])
            except ValueError:
                print("Error: Number of iterations must be an integer.")
                sys.exit(1)
            if isinstance(number_check, int):
                self.arg1 = int(sys.argv[1])
            else:
                print("Error: Number of iterations must be an integer.")
                sys.exit(1)
            if sys.argv[2] not in ['1', '2']:
                print("Error: Algorithm method must be 1 or 2.")
                sys.exit(1)
            else:
                self.arg2 = int(sys.argv[2])
            if sys.argv[3] == 'cash':
                self.cash = True
                self.equity = False
            elif sys.argv[3] == 'equity':
                self.cash = False
                self.equity = True
            else:
                print("Error: Backtest type must be 'cash' or 'equity'.")
                sys.exit(1)
        self.extra_short_days_list=['2m', '5m']
        self.short_days_list =['15m', '30m']
        self.medium_days_list = ['60m', '90m', '1h']
        self.long_days_list = ['1d', '5d', '1wk', '1mo', '3mo']
        self.today = datetime.now()

    def backtest(self):
        for ticker in self.ticker_list:
            symbol_list = [ticker, self.ticker_list_2[self.ticker_list.index(ticker)]]
            self.universe = datetime.strptime("2000-01-01", "%Y-%m-%d")
            self.tie_in = 365
            self.end_date_range = self.today - timedelta(days=self.tie_in)
            try:
                print("-" * 50)
                print(f"Downloading {ticker} and {symbol_list[1]} data...")
                self.df = yf.download(symbol_list, start=self.universe, end=str(date.today() - timedelta(days=1)), multi_level_index=False, ignore_tz=True)
                if isinstance(self.df.columns, pd.MultiIndex):
                    self.df.columns = [' '.join(col).strip() for col in self.df.columns.values]
                self.index1 = self.df.index[0]
                if self.df is None or self.df.empty:
                    print("-" * 50)
                    print("-" * 50)
                    print(f"Data for {ticker} and {symbol_list[1]} is empty or None. Skipping...")
                    print("-" * 50)
                    print("-" * 50)
                    sys.exit(1)
                if ticker != 'SPY':
                    print('Downloading SPY data for benchmark...')
                    self.spydf = yf.download('SPY', start = self.universe, end = str(date.today() - timedelta(1)), multi_level_index=False, ignore_tz=True)
            except Exception as e:
                print(f"Error downloading data for {ticker} and {symbol_list[1]}: {e}")
            print(f'Running backtest for {ticker} and {symbol_list[1]} with {self.arg1} iterations, method {self.arg2}, and {"cash" if self.cash else "equity"} backtest type...')
            for i in range(self.arg1):
                print("-" * 50)
                print(f"Iteration {i + 1} of {self.arg1} for {ticker} and {symbol_list[1]}...")
                random_input = random.randint(0, (self.end_date_range - self.index1).days)
                input_start_date = pd.to_datetime(self.index1 + timedelta(days=random_input))
                input_end_date = pd.to_datetime(input_start_date + timedelta(days=self.tie_in))
                if input_end_date < self.today:
                    try:
                        model = Pairs_Trade(symbol_list=symbol_list, start = self.universe, optional_df=self.df)
                        model.run_algo(method=self.arg2, start_date=input_start_date, end_date=input_end_date, return_table=True)
                    except Exception as e:
                        print(f"Error running algorithm for {ticker} and {symbol_list[1]}: {e}")
                        continue
                    real_start_date = model.df.index[0]  # Get the first date in the DataFrame
                    real_end_date = model.df.index[-1]
                    if ticker != 'SPY':
                        spy_model = Simple_Return(ticker=ticker, start=input_start_date, end=real_end_date, optional_df=self.spydf)
                    if self.equity == True and self.cash == False:
                        backtest_result = model.backtest(method='equity', return_table=False, print_statement=False, model_return=True)
                        buy_hold_result = model.backtest(method='equity', return_table=False, buy_hold=True)
                    if self.cash == True and self.equity == False:
                        backtest_result = model.backtest(method='cash', return_table=False, print_statement=False, model_return=True)
                        buy_hold_result = model.backtest(method='cash', return_table=False, buy_hold=True)
                    
                    backtest_sharpe = model.sharpe_ratio(return_model=True)
                    buy_hold_sharpe = model.sharpe_ratio(return_buy_hold=True)
                    if ticker != 'SPY':
                        spy_result = spy_model.get_return()
                        spy_sharpe = spy_model.get_sharpe()
                        spy_delta = backtest_result - spy_result
                        print(f"SPY Buy/Hold Result: {spy_result}")
                    delta = backtest_result - buy_hold_result

                    if np.isnan(backtest_sharpe):
                        print(f"Error: Errors found in backtest due to overload. Backtest #{i + 1} scrapped.")
                        return
                    else:
                        filename = f'backtest_output/backtest_results_{ticker}_{symbol_list[1]}_method_{self.arg2}_{"cash" if self.cash else "equity"}.csv'
                        with open(filename, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if f.tell() == 0:
                                if ticker != 'SPY':
                                    writer.writerow(['Input Start Date', 'Input End Date', 'Start Date', 'End Date', 'Model Result', 'Buy/Hold Result', 'Delta', 'Model Sharpe', 'Buy/Hold Sharpe', 'SPY Buy/Hold Result', 'SPY Sharpe', 'SPY Delta'])
                                else:
                                    writer.writerow(['Input Start Date', 'Input End Date', 'Start Date', 'End Date', 'Model Result', 'Buy/Hold Result', 'Delta', 'Model Sharpe', 'Buy/Hold Sharpe'])
                            if ticker != 'SPY':
                                writer.writerow([input_start_date, input_end_date, real_start_date, real_end_date, backtest_result, buy_hold_result, round(delta, 2), backtest_sharpe, buy_hold_sharpe, spy_result, spy_sharpe, round(spy_delta, 2)])
                            else:
                                writer.writerow([input_start_date, input_end_date, real_start_date, real_end_date, backtest_result, buy_hold_result, round(delta, 2), backtest_sharpe, buy_hold_sharpe])
                elif input_end_date >= self.today:
                    print(f"End Date is not valid, no entry recorded")
                    print(f"{input_end_date}")
            print("-" * 50)
            print(f"Backtest for {ticker} and {symbol_list[1]} completed. Results saved to {filename}")
            print("-" * 50)
        print("-" * 50)
        print("All backtests completed.")
        print("-" * 50)


if __name__ == "__main__":
    backtest = Backtest()
    backtest.backtest()