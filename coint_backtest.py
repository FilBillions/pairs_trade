import sys
sys.path.append(".")
import csv
import yfinance as yf
from datetime import timedelta, datetime

# The idea of this backtest is to import a list of stock tickers from a CSV file
# and run the stationarity test between all the pairs of stocks to see which pairs are highly cointegrated.

from contents.pairs_trade_obj import Pairs_Trade

class CointegrationBacktest():
    def __init__(self):
        with open('inputs.csv', 'r') as file:
            reader = csv.DictReader(file)
            self.ticker_list = [row['Ticker'] for row in reader if row['Ticker']]
    def backtest(self):
        self.symbol_list = []
        self.universe = datetime.strptime("2000-01-01", "%Y-%m-%d")
        filename = 'successful_pairs.csv'
        filename2 = 'all_pairs_and_percentage.csv'
        iteration = 0
        total_iterations = len(self.ticker_list) * (len(self.ticker_list) - 1)
        for ticker in self.ticker_list:
            self.symbol_list.append(ticker)
            for other in self.ticker_list:
                iteration += 1
                if ticker != other:
                    self.symbol_list.append(other)
                    self.df = yf.download(self.symbol_list, start=self.universe, multi_level_index=False, ignore_tz=True)['Close']
                    self.df.dropna(inplace=True)  # drop rows with NaN values
                else:
                    continue # skip if the same ticker
                if ticker in self.symbol_list and other in self.symbol_list:
                    print(f"Iteration {iteration} of {total_iterations}: Testing pair {ticker} and {other}")
                    try:
                        pair = Pairs_Trade(symbol_list=self.symbol_list, optional_df=self.df)
                        pair_percentage = pair.robust_stationarity_test(return_percentage=True)
                        with open(filename2, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if f.tell() == 0:
                                writer.writerow(['Ticker', 'Other Ticker', 'Percentage'])
                            writer.writerow([ticker, other, f"{pair_percentage:.2f}%"])
                        if pair_percentage > 95:  # threshold for successful pairs
                            print(f"Successful pairs: {ticker} and {other} with percentage: {pair_percentage:.2f}%")
                            with open(filename, 'a', newline='') as f:
                                writer = csv.writer(f)
                                if f.tell() == 0:
                                    writer.writerow(['Ticker', 'Other Ticker', 'Percentage'])
                                writer.writerow([ticker, other, f"{pair_percentage:.2f}%"])
                    except Exception as e:
                        print(f"Error processing pair {ticker} and {other}: {e}")
                self.symbol_list.remove(other) # remove the other ticker to avoid duplicates
            self.symbol_list.remove(ticker) # remove the ticker to avoid duplicates
        print("-" * 50)
        print(f"Backtest completed. Successful pairs saved to {filename}")
        print("-" * 50)

if __name__ == "__main__":
    backtest = CointegrationBacktest()
    backtest.backtest()