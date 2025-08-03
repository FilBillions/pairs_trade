import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import statsmodels.api as sm
from scipy import stats
import sys
import os
sys.path.append(os.path.dirname(__file__))

from stationarity_test import check_for_stationarity


np.set_printoptions(legacy='1.25')

class Pairs_Trade():
    def __init__(self, symbol_list, start = str(date.today() - timedelta(59)), end = str(date.today() - timedelta(1)), interval = '1d', optional_df=None):
        if optional_df is not None:
            self.df = optional_df
        else:
            df = yf.download(symbol_list, start, end, interval = interval, multi_level_index=False, ignore_tz=True)
            self.df = df
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = [' '.join(col).strip() for col in self.df.columns.values]
        self.symbol_list = symbol_list
        self.x1 = self.df[f'Close {self.symbol_list[0]}'] # price series of the first symbol
        self.x2 = self.df[f'Close {self.symbol_list[1]}'] # price series of the second symbol
        day_count = np.arange(1, len(self.df) + 1)
        self.df['Day Count'] = day_count
        self.ticker1 = self.symbol_list[0]
        self.ticker2 = self.symbol_list[1]
        self.df[f'{self.symbol_list[0]} Return'] = np.log(self.x1).diff()
        self.df[f'{self.symbol_list[1]} Return'] = np.log(self.x2).diff()
        self.interval = interval
        self.df[f'{self.symbol_list[0]} Previous Close'] = self.x1.shift(1)
        self.df[f'{self.symbol_list[1]} Previous Close'] = self.x2.shift(1)
        self.df.dropna(inplace=True)

        # Group of data that will be recalculated every x steps
        self.df['Long'] = np.nan
        self.df['Short'] = np.nan
        self.df['Long2'] = np.nan
        self.df['Short2'] = np.nan
        self.df['Exit'] = np.nan
        self.df['Z-Score'] = np.nan

    def robust_stationarity_test(self, end_date=date.today(), step_input=5, return_percentage=False):
        start_date = self.df.index[0]
        days = 60
        if self.df.index.tz is not None:
            if pd.Timestamp(start_date).tzinfo is None:
                start_date = pd.Timestamp(start_date).tz_localize('UTC')
            else:
                start_date = pd.Timestamp(start_date).tz_convert('UTC')
            if pd.Timestamp(end_date).tzinfo is None:
                end_date = pd.Timestamp(end_date).tz_localize('UTC')
            else:
                end_date = pd.Timestamp(end_date).tz_convert('UTC')
        else:
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
        if isinstance(start_date, int):
            post_data = self.df[self.df.index.year >= start_date]
            data_cutoff = []
        else:
            post_data = self.df[self.df.index >= start_date]
            end_date = self.df.index[-1]
            data_cutoff = self.df[self.df.index >= end_date]
        post_idx = self.df.index.get_loc(post_data.index[0])
        stationarity_count = 0
        total_count = 0
        while post_idx < len(self.df) - len(data_cutoff):
            if (post_idx - self.df.index.get_loc(post_data.index[0])) % step_input == 0: # every x steps, recalculate the beta using new data
            # Get the last date in the current window
                forward_date = self.df.index[post_idx] + pd.DateOffset(days=days)
                # Select only the last  month of data up to forward_date
                one_month_ago = start_date
                current_pre_data = self.df[(self.df.index < forward_date) & (self.df.index >= one_month_ago)]
                x1constant = sm.add_constant(current_pre_data[f'{self.symbol_list[0]} Return'])
                x1results = sm.OLS(current_pre_data[f'{self.symbol_list[1]} Return'], x1constant).fit()
                x1constant = x1constant[f'{self.symbol_list[0]} Return'] # drop the constant term for the next step
                beta = x1results.params[f'{self.symbol_list[0]} Return'] # This is the beta for the last months worth of data
                z = current_pre_data[f'{self.symbol_list[1]} Return'] - beta * current_pre_data[f'{self.symbol_list[0]} Return']
                z.name = f'Cointegration of {self.symbol_list[0]} and {self.symbol_list[1]}'
                if check_for_stationarity(z, print_results=False) is not True:
                    pass
                else:
                    stationarity_count += 1
                total_count += 1
                # cut loop if we reach the end date
                if forward_date >= end_date:
                    break
            start_date = self.df.index[post_idx]
            post_idx += 1
        print(f"Total Stationary Checks: {total_count}")
        print(f"Total Stationary Checks Passed: {stationarity_count}")
        print(f"Percentage of Stationary Checks Passed: {stationarity_count / total_count * 100:.2f}%")
        if stationarity_count / total_count < 0.95:
            print("Warning: Less than 95% of the checks passed. The data may not be suitable for pairs trading.")
        else:
            print("The data is suitable for pairs trading based on stationarity checks.")
        if return_percentage:
            return stationarity_count / total_count * 100

    def run_algo(self, method = 1,start_date=date.today().year- 1, end_date=date.today(), step_input = 5, return_table=False):
            # - - - Run the Algorithm - - -
        # - - - Initialize post data and pre data sets - - -
        # - - - We only use data from before the specified start date - - -
        # Ensure start_date and end_date are timezone-aware and in UTC
        if self.df.index.tz is not None:
            if pd.Timestamp(start_date).tzinfo is None:
                start_date = pd.Timestamp(start_date).tz_localize('UTC')
            else:
                start_date = pd.Timestamp(start_date).tz_convert('UTC')
            if pd.Timestamp(end_date).tzinfo is None:
                end_date = pd.Timestamp(end_date).tz_localize('UTC')
            else:
                end_date = pd.Timestamp(end_date).tz_convert('UTC')
        else:
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
        if isinstance(start_date, int):
            post_data = self.df[self.df.index.year >= start_date]
            data_cutoff = []
        else:
            post_data = self.df[self.df.index >= start_date]
            if end_date == date.today():
                end_date = str(date.today())
            data_cutoff = self.df[self.df.index >= end_date]
        # Prepare lists to collect actions
        actions = []
        dates = []
        post_idx = self.df.index.get_loc(post_data.index[0])

        prev_action = 'No Action'  # Initialize previous action
        next_action = None  # This will hold the action for the next day
        action = 'No Action'  # <-- Ensure action is always initialized


        #Algorithm starts here
        # - - - Loop through the post data set - - -
        while post_idx < (len(self.df) - len(data_cutoff)):
            # ----
        # Update the beta every step and close any positions if the z-score is not stationary
            month_ago = self.df.index[post_idx] - pd.DateOffset(months=2)
            current_pre_data = self.df[(self.df.index < self.df.index[post_idx + 1]) & (self.df.index >= month_ago)]
            x1constant = sm.add_constant(current_pre_data[f'{self.symbol_list[0]} Return'])
            x1results = sm.OLS(current_pre_data[f'{self.symbol_list[1]} Return'], x1constant).fit()
            x1constant = x1constant[f'{self.symbol_list[0]} Return'] # drop the constant term for the next step
            beta = x1results.params[f'{self.symbol_list[0]} Return'] # This is the beta for the last years worth of data
            self.z = current_pre_data[f'{self.symbol_list[1]} Return'] - beta * current_pre_data[f'{self.symbol_list[0]} Return']
            self.z.name = f'Cointegration of {self.symbol_list[0]} and {self.symbol_list[1]}'
            # Only update Z-Score where it is currently NaN
            mask = self.df['Z-Score'].isna()
            self.df.loc[mask, 'Z-Score'] = self.z[mask]
            #if check_for_stationarity(self.z, print_results=False) is not True:
            #   print(f"Start Date: {current_pre_data.index[0]} End Date: {current_pre_data.index[-1]} not Stationary. Not entering any new positions until stationarity is restored.")
            if method == 1:
                # Method 1 Calculates the thresholds base on the standard deviation of the z-score -> think bollinger bands
                std = self.z.std()
                self.df.loc[self.df.index[post_idx], 'Long'] = self.z.mean() - std
                self.df.loc[self.df.index[post_idx], 'Short'] = self.z.mean() + std
                self.df.loc[self.df.index[post_idx], 'Long2'] = self.z.mean() - (std * 2)
                self.df.loc[self.df.index[post_idx], 'Short2'] = self.z.mean() + (std * 2)
                self.df.loc[self.df.index[post_idx], 'Exit'] = self.z.mean()

                self.df.loc[:self.df.index[post_idx], 'Long'] = self.df.loc[:self.df.index[post_idx], 'Long'].ffill()
                self.df.loc[:self.df.index[post_idx], 'Short'] = self.df.loc[:self.df.index[post_idx], 'Short'].ffill()
                self.df.loc[:self.df.index[post_idx], 'Long2'] = self.df.loc[:self.df.index[post_idx], 'Long2'].ffill()
                self.df.loc[:self.df.index[post_idx], 'Short2'] = self.df.loc[:self.df.index[post_idx], 'Short2'].ffill()
                self.df.loc[:self.df.index[post_idx], 'Exit'] = self.df.loc[:self.df.index[post_idx], 'Exit'].ffill()
            elif method == 2:
                # Method 2 calculates the thresholds based on the percentiles of the ratio from the entire universe of previous data
                percentiles = [5, 10, 50, 90, 95]
                self.p = np.percentile(self.z, percentiles)
                self.df.loc[self.df.index[post_idx], 'Long'] = self.p[0]
                self.df.loc[self.df.index[post_idx], 'Long2'] = self.p[1]
                self.df.loc[self.df.index[post_idx], 'Short'] = self.p[4]
                self.df.loc[self.df.index[post_idx], 'Short2'] = self.p[3]
                self.df.loc[self.df.index[post_idx], 'Exit'] = self.p[2]
                    
            start_date = self.df.index[post_idx]
        # generate a buy signal when the ratio CROSSES below the long theshold
            if self.z.iloc[-2] >= self.df['Long'].iloc[post_idx] and self.z.iloc[-1] < self.df['Long'].iloc[post_idx]:
                if prev_action != 'Sell to Open' and prev_action != 'Hold (Short)':
                    action = 'Buy to Open'
        # generate a buy signal when the ratio CROSSES below the long2 theshold
            elif self.z.iloc[-2] >= self.df['Long2'].iloc[post_idx] and self.z.iloc[-1] < self.df['Long2'].iloc[post_idx]:
                if prev_action != 'Sell to Open' and prev_action != 'Hold (Short)':
                    action = 'Buy to Open'
        # generate a sell signal when the ratio CROSSES above the short theshold
            elif self.z.iloc[-2] <= self.df['Short'].iloc[post_idx] and self.z.iloc[-1] > self.df['Short'].iloc[post_idx]:
                if prev_action != 'Buy to Open' and prev_action != 'Hold (Long)':
                    action = 'Sell to Open'
        # generate a sell signal when the ratio CROSSES above the short2 theshold
            elif self.z.iloc[-2] <= self.df['Short2'].iloc[post_idx] and self.z.iloc[-1] > self.df['Short2'].iloc[post_idx]:
                if prev_action != 'Buy to Open' and prev_action != 'Hold (Long)':
                    action = 'Sell to Open'
        # generate a Sell to Close signal when the ratio crosses above the exit theshold and we are in a long position or holding a long position
            elif (prev_action == 'Buy to Open' or prev_action == 'Hold (Long)') and self.z.iloc[-2] < self.df['Exit'].iloc[post_idx] and self.z.iloc[-1] >= self.df['Exit'].iloc[post_idx]:
                action = 'Sell to Close'
        # generate a Buy to Close signal when the ratio crosses below the exit theshold and we are in a short position or holding a short position
            elif (prev_action == 'Sell to Open' or prev_action == 'Hold (Short)') and self.z.iloc[-2] > self.df['Exit'].iloc[post_idx] and self.z.iloc[-1] <= self.df['Exit'].iloc[post_idx]:
                action = 'Buy to Close'
        # Hold long if we are in a long position and we have not yet hit the exit threshold
            elif (prev_action == 'Buy to Open' or prev_action == 'Hold (Long)') and self.z.iloc[-1] < self.df['Exit'].iloc[post_idx]:
                action = 'Hold (Long)'
        # Hold short if we are in a short position and we have not yet hit the exit threshold
            elif (prev_action == 'Sell to Open' or prev_action == 'Hold (Short)') and self.z.iloc[-1] > self.df['Exit'].iloc[post_idx]:
                action = 'Hold (Short)'
            else:
                if prev_action == "Buy to Open" or prev_action == "Hold (Long)":
                    action = 'Sell to Close'
                elif prev_action == "Sell to Open" or prev_action == "Hold (Short)":
                    action = 'Buy to Close'
                elif prev_action == "No Action" or prev_action == "Buy to Close" or prev_action == "Sell to Close":
                    action = 'No Action'
                else:
                    action = 'Error'
        # Only append the action for the previous day (to avoid lookahead bias)
            if next_action is not None:
                dates.append(start_date)
                actions.append(next_action)

            prev_action = action
            next_action = action
            post_idx += 1
            #break when we rech the end date

        df_actions = pd.DataFrame({'Date': dates, 'Action': actions})

        self.df = self.df.join(df_actions.set_index('Date'), how='left')
        # Signals to long the spread
        self.df['Buy Signal Primary'] = np.where(self.df['Action'] == 'Buy to Open', self.df[f'{self.symbol_list[0]} Previous Close'], (np.where(self.df['Action'] == 'Buy to Close', self.df[f'{self.symbol_list[0]} Previous Close'], np.nan)))
        self.df['Sell Signal Secondary'] = np.where(self.df['Action'] == 'Buy to Open', self.df[f'{self.symbol_list[1]} Previous Close'], (np.where(self.df['Action'] == 'Buy to Close', self.df[f'{self.symbol_list[1]} Previous Close'], np.nan)))
        # Signals to short the spread
        self.df['Sell Signal Primary'] = np.where(self.df['Action'] == 'Sell to Open', self.df[f'{self.symbol_list[0]} Previous Close'], (np.where(self.df['Action'] == 'Sell to Close', self.df[f'{self.symbol_list[0]} Previous Close'], np.nan)))
        self.df['Buy Signal Secondary'] = np.where(self.df['Action'] == 'Sell to Open', self.df[f'{self.symbol_list[1]} Previous Close'], (np.where(self.df['Action'] == 'Sell to Close', self.df[f'{self.symbol_list[1]} Previous Close'], np.nan)))
       
       
       # remove rows with NaN in action
        self.df.dropna(subset=['Action'], inplace=True)

        if return_table:
            print(f"Total Trades: {df_actions['Action'].value_counts()['Buy to Open'] + df_actions['Action'].value_counts()['Sell to Open']}")
            print(f"Buys/Sells {df_actions['Action'].value_counts()}")
            pd.set_option('display.max_rows', None)
            #only show the below columns
            return self.df[['Day Count', 'Z-Score', 'Long', 'Short', 'Long2', 'Short2', 'Exit', 'Action', 'Buy Signal Primary', 'Sell Signal Secondary', 'Sell Signal Primary', 'Buy Signal Secondary']]

    def backtest(self, method='equity', print_statement=True, return_table=False, model_return=False, buy_hold=False, return_model_df=False):
        # this backtest assumes that we are using a percentage of our equity to trade, rather than a fixed amount
        initial_investment = 10000
        cash = initial_investment
        total_cash_spent_1 = 0
        total_cash_spent_2 = 0
        shares_1 = 0
        shares_2 = 0
        long_value = 0
        short_value = 0
        if method == 'equity':
            trade_weight = 0.5 # for every trade signal, we will invest some percent of our cash
            cash_spent_1 = 0
            cash_spent_2 = 0
        if method == 'cash':
            percent = 0.1 # for every trade signal, we will invest 10% of our cash
        portfolio_value = []

        share_cost_1 = self.df[f"{self.symbol_list[0]} Previous Close"].iloc[0]
        num_shares_1 = initial_investment / share_cost_1
        self.df['Buy/Hold Value'] = num_shares_1 * self.df[f'Close {self.symbol_list[0]}']
        self.df['Model Value'] = 0
        # Iterate through the DataFrame
        for i in range(0,len(self.df)):
            action = self.df['Action'].iloc[i]
            price_1 = self.df[f'{self.symbol_list[0]} Previous Close'].iloc[i]
            price_2 = self.df[f'{self.symbol_list[1]} Previous Close'].iloc[i]
            cash_1 = cash /2
            cash_2 = cash /2
            if action == 'Buy to Open' and cash > 0:
                if method == 'equity':
                    shares_1 = ((cash_1 * trade_weight) / price_1) + shares_1
                    shares_2 = ((cash_2 * trade_weight) / price_2) + shares_2
                    cash_spent_1 = (cash_1 * trade_weight)
                    cash_spent_2 = (cash_2 * trade_weight)
                    total_cash_spent_1 += cash_spent_1
                    total_cash_spent_2 += cash_spent_2
                    avg_price_1 = total_cash_spent_1 / shares_1
                    avg_price_2 = total_cash_spent_2 / shares_2
                    long_price_1 = avg_price_1
                    long_price_2 = avg_price_2
                    long_value = (shares_1 * price_1) + (shares_2 * price_2)
                    cash = cash - (cash_spent_1 + cash_spent_2)
                elif method == 'cash':
                    beginning_cash_1 = cash_1 + total_cash_spent_1
                    beginning_cash_2 = cash_2 + total_cash_spent_2
                    trade_amt_1 = beginning_cash_1 * percent
                    trade_amt_2 = beginning_cash_2 * percent
                    shares_1 = (trade_amt_1 / price_1) + shares_1
                    shares_2 = (trade_amt_2 / price_2) + shares_2
                    total_cash_spent_1 += trade_amt_1
                    total_cash_spent_2 += trade_amt_2
                    avg_price_1 = total_cash_spent_1 / shares_1
                    avg_price_2 = total_cash_spent_2 / shares_2
                    long_value = (shares_1 * price_1) + (shares_2 * price_2)
                    cash = cash - (trade_amt_1 + trade_amt_2)
            elif action == 'Sell to Close' and shares_1 > 0:
                if method == 'equity':
                    cash = (shares_1 * long_price_1) + ((shares_1 * long_price_1) * ((price_1 - long_price_1) / long_price_1)) + (shares_2 * long_price_2) - ((shares_2 * long_price_2) * ((price_2 - long_price_2) / long_price_2)) + cash
                    shares_1 = 0
                    shares_2 = 0
                    long_value = 0
                    avg_price_1 = 0
                    long_price_1 = 0
                    avg_price_2 = 0
                    long_price_2 = 0
                    cash_spent_1 = 0
                    total_cash_spent_1 = 0
                    cash_spent_2 = 0
                    total_cash_spent_2 = 0
                elif method == 'cash':
                    cash = (shares_1 * avg_price_1) + ((shares_1 * avg_price_1) * ((price_1 - avg_price_1) / avg_price_1)) + (shares_2 * avg_price_2) - ((shares_2 * avg_price_2) * ((price_2 - avg_price_2) / avg_price_2)) + cash
                    shares_1 = 0
                    shares_2 = 0
                    long_value = 0
                    avg_price_1 = 0
                    avg_price_2 = 0
                    total_cash_spent_1 = 0
                    total_cash_spent_2 = 0
            elif action == 'Sell to Open' and cash > 0:
                if method == 'equity':
                    shares_1 = ((cash_1 * trade_weight) / price_1) + shares_1
                    shares_2 = ((cash_2 * trade_weight) / price_2) + shares_2
                    cash_spent_1 = (cash_1 * trade_weight)
                    cash_spent_2 = (cash_2 * trade_weight)
                    total_cash_spent_1 += cash_spent_1
                    total_cash_spent_2 += cash_spent_2
                    avg_price_1 = total_cash_spent_1 / shares_1
                    avg_price_2 = total_cash_spent_2 / shares_2
                    short_price_1 = avg_price_1
                    short_price_2 = avg_price_2
                    short_value = (shares_1 * price_1) + (shares_2 * price_2)
                    cash = cash - (cash_spent_1 + cash_spent_2)
                elif method == 'cash':
                    beginning_cash_1 = cash_1 + total_cash_spent_1
                    beginning_cash_2 = cash_2 + total_cash_spent_2
                    trade_amt_1 = beginning_cash_1 * percent
                    trade_amt_2 = beginning_cash_2 * percent
                    shares_1 = (trade_amt_1 / price_1) + shares_1
                    shares_2 = (trade_amt_2 / price_2) + shares_2
                    total_cash_spent_1 += trade_amt_1
                    total_cash_spent_2 += trade_amt_2
                    avg_price_1 = total_cash_spent_1 / shares_1
                    avg_price_2 = total_cash_spent_2 / shares_2
                    short_value = (shares_1 * price_1) + (shares_2 * price_2)
                    cash = cash - (trade_amt_1 + trade_amt_2)
            elif action == 'Buy to Close' and shares_1 > 0:
                if method == 'equity':
                    cash = (shares_1 * short_price_1) - ((shares_1 * short_price_1) * ((price_1 - short_price_1) / short_price_1)) + (shares_2 * short_price_2) + ((shares_2 * short_price_2) * ((price_2 - short_price_2) / short_price_2)) + cash
                    shares_1 = 0
                    shares_2 = 0
                    short_value = 0
                    avg_price_1 = 0
                    avg_price_2 = 0
                    short_price_1 = 0
                    short_price_2 = 0
                    cash_spent_1 = 0
                    total_cash_spent_1 = 0
                    cash_spent_2 = 0
                    total_cash_spent_2 = 0
                elif method == 'cash':
                    cash = (shares_1 * avg_price_1) - ((shares_1 * avg_price_1) * ((price_1 - avg_price_1) / avg_price_1)) + (shares_2 * avg_price_2) + ((shares_2 * avg_price_2) * ((price_2 - avg_price_2) / avg_price_2)) + cash
                    shares_1 = 0
                    shares_2 = 0
                    short_value = 0
                    avg_price_1 = 0
                    avg_price_2 = 0
                    total_cash_spent_1 = 0
                    total_cash_spent_2 = 0
            elif action == "Hold (Short)":
                short_value = (shares_1 * price_1) + (shares_2 * price_2)
            elif action == 'Hold (Long)':
                long_value = (shares_1 * price_1) + (shares_2 * price_2)

            model_value = (cash + long_value + short_value)
            portfolio_value.append(model_value)
        self.df['Model Value'] = portfolio_value
                #dropping unnecessary columns
        if 'Volume' in self.df.columns:
                self.df.drop(columns=['Volume'], inplace = True)
        if 'Previous Bin' in self.df.columns:
                self.df.drop(columns=['Previous Bin'], inplace = True)
        if 'Current Bin' in self.df.columns:
                self.df.drop(columns=['Current Bin'], inplace = True)
        
        if print_statement:
            print(f"{self.symbol_list[0]} Buy/Hold Result: {round(((self.df['Buy/Hold Value'].iloc[-1] - self.df['Buy/Hold Value'].iloc[0])/self.df['Buy/Hold Value'].iloc[0]) * 100, 2)}%")
            print(f"{self.symbol_list[0]} and {self.symbol_list[1]} Model Result: {round(((self.df['Model Value'].iloc[-1] - self.df['Model Value'].iloc[0])/self.df['Model Value'].iloc[0]) * 100, 2)}%")
            print(f" from {self.df.index[0]} to {self.df.index[-1]}")
        if return_table:
            return self.df
        if model_return:
            return round(((self.df['Model Value'].iloc[-1] - self.df['Model Value'].iloc[0])/self.df['Model Value'].iloc[0]) * 100, 2)
        if buy_hold:
            return round(((self.df['Buy/Hold Value'].iloc[-1] - self.df['Buy/Hold Value'].iloc[0])/self.df['Buy/Hold Value'].iloc[0]) * 100, 2)
        if return_model_df:
            return self.df['Model Value']
        
    def sharpe_ratio(self, return_model=True, return_buy_hold=False):
        # factor answers the question: how many of this interval are in the total timespan
        if self.interval == "1d":
            #There are 252 trading days in a year
            annualized_factor = 252
        elif self.interval == "1wk":
            #52 weeks in a year
            annualized_factor = 52
        elif self.interval == "1mo":
            #12 months in a year
            annualized_factor = 12
        else:
            annualized_factor = 1
        model_descriptives = stats.describe(self.df['Model Value'].pct_change().dropna())
        model_mean = model_descriptives.mean
        model_std = model_descriptives.variance ** 0.5
        model_sharpe = model_mean / model_std * (annualized_factor ** 0.5)
        buy_hold_descriptives = stats.describe(self.df['Buy/Hold Value'].pct_change().dropna())
        buy_hold_mean = buy_hold_descriptives.mean
        buy_hold_std = buy_hold_descriptives.variance ** 0.5
        buy_hold_sharpe = buy_hold_mean / buy_hold_std * (annualized_factor ** 0.5)
        if return_buy_hold:
            return round(buy_hold_sharpe, 6)
        if return_model:
            return round(model_sharpe, 6)