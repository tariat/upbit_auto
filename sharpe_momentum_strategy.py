import os
import sys
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Union
import traceback

from autils.bitcoin.upbit import api # If using upbit data directly
# from autils.db.mysql_utils import MySQL # If using DB loading part of load_price
# from db_job import get_table_name, init_db # If using DB loading part of load_price

# from model import Screening as scr # Not needed for this strategy
from backtesting import Backtest, Strategy
from backtesting.lib import crossover # barssince might be useful later
from backtesting.test import SMA, RSI # Keep SMA, RSI might be removed if not used


def load_price(market, mode, minutes_lag, count=200, is_load_db=False):
    """
        is_load_db: db에서 불러올지 입력
        market='KRW-BTC';mode="backtest";minutes_lag=3;count=200
    """
    # init_db(market, mode, minutes_lag)
    # trading_table_nm, history_table_nm = get_table_name(market, mode, minutes_lag)

    if is_load_db==False:
        data = api.get_minutes_candle(market, count = count, minutes_lag=minutes_lag)
        # with MySQL() as ms:
        #     for _, candle in data.iterrows():
        #         # print(candle) # Commented out
        #         ms.execute("INSERT IGNORE INTO " + history_table_nm + '''(
        #                 timestamp, market, opening_price, high_price, low_price,
        #                 trade_price, candle_acc_trade_price, candle_acc_trade_volume
        #             )
        #             VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''', (
        #             candle['candle_date_time_kst'],
        #             candle['market'],
        #             candle['opening_price'],
        #             candle['high_price'],
        #             candle['low_price'],
        #             candle['trade_price'],
        #             candle['candle_acc_trade_price'],
        #             candle['candle_acc_trade_volume']))
    else:
        # with MySQL() as ms:
        #     query = f'''SELECT * FROM {history_table_nm} LIMIT {count}'''
        #     data = ms.get(query)
        print("Loading from DB is currently disabled in sharpe_momentum_strategy.py")
        data = pd.DataFrame() # Return empty dataframe if DB load is attempted

    # Ensure data is not empty before proceeding
    if data.empty:
        print(f"No data loaded for {market}. Please check API or DB connection if is_load_db=True was intended.")
        # Return a DataFrame with expected columns but no rows to prevent errors downstream
        # if further processing assumes these columns exist.
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close']).set_index('Date')


    data = data[['timestamp', 'opening_price', 'high_price', 'low_price', 'trade_price']]
    data = data.rename(
        {'timestamp': 'Date', 'opening_price': 'Open', 'high_price': 'High', 'low_price': 'Low',
         'trade_price': 'Close'}, axis=1)

    return data.set_index('Date')


class MomentumSharpeStrategy(Strategy):
    # Define parameters with default values
    sma_short_period = 20
    sma_long_period = 50
    sharpe_lookback_period = 60
    sharpe_threshold_entry = 0.05
    sharpe_threshold_exit = 0.0

    @staticmethod
    def _calculate_rolling_sharpe(price_series_array, lookback_window, n_day_return_period=1, annualization_factor=None, risk_free_rate_period=0):
        s_prices = pd.Series(price_series_array)
        # Ensure returns are calculated correctly based on data frequency.
        # If data is minutely, pct_change(1) is minute return.
        # If data is daily, pct_change(1) is daily return.
        returns = s_prices.pct_change(n_day_return_period)

        excess_returns = returns - risk_free_rate_period

        # Use min_periods to get values even if window is not full yet, helps with startup
        rolling_mean_excess = excess_returns.rolling(window=lookback_window, min_periods=max(1,int(lookback_window*0.6))).mean()
        rolling_std_excess = excess_returns.rolling(window=lookback_window, min_periods=max(1,int(lookback_window*0.6))).std()

        sharpe = rolling_mean_excess / rolling_std_excess

        # Handle potential NaNs from early values or zero std dev
        sharpe = sharpe.fillna(0)
        # Replace inf/-inf with 0 that can occur if std is zero but mean is non-zero
        sharpe.replace([np.inf, -np.inf], 0, inplace=True)

        if annualization_factor:
            sharpe = sharpe * np.sqrt(annualization_factor)
        return sharpe.values

    def init(self):
        # Parameters like self.sma_short_period are automatically set by the backtesting framework
        # if provided in Backtest() or run() call, overriding class defaults.

        self.sma_short = self.I(SMA, self.data.Close, self.sma_short_period, name=f"SMA_short({self.sma_short_period})")
        self.sma_long = self.I(SMA, self.data.Close, self.sma_long_period, name=f"SMA_long({self.sma_long_period})")

        self.rolling_sharpe = self.I(
            self._calculate_rolling_sharpe, # Static method
            self.data.Close,
            lookback_window=self.sharpe_lookback_period,
            n_day_return_period=1,
            annualization_factor=None,
            risk_free_rate_period=0,
            name=f"RollingSharpe({self.sharpe_lookback_period})",
            plot=True,
            overlay=False
        )

    def next(self):
        current_sharpe = self.rolling_sharpe[-1]

        # Entry condition
        if not self.position:
            if crossover(self.sma_short, self.sma_long) and current_sharpe > self.sharpe_threshold_entry:
                self.buy()

        # Exit condition
        elif self.position.is_long:
            if crossover(self.sma_long, self.sma_short) or current_sharpe < self.sharpe_threshold_exit:
                self.position.close()


if __name__ == "__main__":
    # Ensure traceback and sys are imported at the top of the file:
    # import traceback (already present)
    # import sys (already present)

    # 1. Load Data
    print("Loading data...")

    param_sma_long_period = 48
    param_sharpe_lookback_period = 120
    # Calculate minimum data length required: longest lookback + some buffer (e.g., 50 candles)
    min_data_length = max(param_sma_long_period, param_sharpe_lookback_period) + 50

    data = load_price(
        market='KRW-BTC',
        mode="backtest",
        minutes_lag=60, # Hourly candles
        count=24*100,   # 100 days of hourly data (2400 candles)
        is_load_db=False
    )

    if data.empty:
        print(f"No data loaded for market KRW-BTC. Exiting.")
        sys.exit(1)

    if len(data) < min_data_length:
        print(f"Warning: Data length ({len(data)}) is less than recommended minimum ({min_data_length}).")
        print("Backtest might produce unexpected results or fail. Consider increasing 'count' in load_price.")
        # Optionally, one might choose to exit here as well:
        # sys.exit(1)

    print(f"Data loaded: {len(data)} candles from {data.index[0]} to {data.index[-1]}")

    # 2. Instantiate Backtest
    print("Initializing backtest...")
    bt = Backtest(
        data,
        MomentumSharpeStrategy,
        cash=100_000_000,  # Starting with 100 million KRW
        commission=.0005    # Typical Upbit commission 0.05%
    )

    # 3. Run the Backtest with Strategy Parameters
    print("Running backtest...")
    stats = bt.run(
        sma_short_period=12,
        sma_long_period=param_sma_long_period,
        sharpe_lookback_period=param_sharpe_lookback_period,
        sharpe_threshold_entry=0.01,
        sharpe_threshold_exit=-0.01
    )

    # 4. Print Results
    print("\nBacktest Results:")
    print(stats)

    # Print individual trades if available
    if '_trades' in stats and not stats['_trades'].empty:
        print("\nTrades:")
        print(stats['_trades'])
    else:
        print("\nNo trades were made.")

    # 5. Plot Results
    plot_filename = "sharpe_momentum_hourly_plot.html"
    print(f"Plotting results to {plot_filename}...")
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
        traceback.print_exc()
