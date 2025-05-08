import os
import sys
import pandas as pd
import numpy as np
import time
import logging
import jwt
import uuid
import requests
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Union
import traceback
import threading

from autils.bitcoin.upbit import api
from db_job import get_table_name, init_db

from model import Screening as scr
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, barssince
from backtesting.test import SMA, RSI

from autils.bitcoin.upbit import api
from autils.db.mysql_utils import MySQL
# from autils.messenser.telegram import send_message_default


def load_price(market, mode, minutes_lag, count=200, is_load_db=False):
    """
        is_load_db: db에서 불러올지 입력
        market='KRW-BTC';mode="backtest";minutes_lag=3;count=200
    """
    init_db(market, mode, minutes_lag)
    trading_table_nm, history_table_nm = get_table_name(market, mode, minutes_lag)

    if is_load_db==False:
        data = api.get_minutes_candle(market, count = count, minutes_lag=minutes_lag)
        with MySQL() as ms:
            for _, candle in data.iterrows():
                print(candle)
                ms.execute("INSERT IGNORE INTO " + history_table_nm + '''(
                        timestamp, market, opening_price, high_price, low_price, 
                        trade_price, candle_acc_trade_price, candle_acc_trade_volume
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''', (
                    candle['candle_date_time_kst'],
                    candle['market'],
                    candle['opening_price'],
                    candle['high_price'],
                    candle['low_price'],
                    candle['trade_price'],
                    candle['candle_acc_trade_price'],
                    candle['candle_acc_trade_volume']))
    else:
        with MySQL() as ms:
            query = f'''SELECT * FROM {history_table_nm} LIMIT {count}'''
            data = ms.get(query)
        
    data = data[['timestamp', 'opening_price', 'high_price', 'low_price', 'trade_price']]
    data = data.rename(
        {'timestamp': 'Date', 'opening_price': 'Open', 'high_price': 'High', 'low_price': 'Low',
         'trade_price': 'Close'}, axis=1)

    return data.set_index('Date')        


class SmaRsi(Strategy):
    def init(self):
        self.level = 30
        self.ma10 = self.I(SMA, self.data.Close, 5)
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.rsi = self.I(RSI, self.data.Close, 14)

    def next(self):
        price = self.data.Close[-1]
        if (not self.position and
                self.rsi[-1] > self.level and
                price > self.ma10):
            self.buy(sl=.9*price)
        elif price < .98 * self.ma10[-1]:
            self.position.close()

if __name__=="__main__":
    scr = scr()
    # coin_list, coin_info = scr.get_basic_info()

    data = load_price('KRW-BTC', "backtest", 3, count=200, is_load_db=True)

    bt = Backtest(data, SmaRsi, commission=.002, cash=100000000, exclusive_orders=True)
    res = bt.run()
    bt.plot()
    print(res)
