"""
    mode: backtest, observe, trade
"""
# 5선 기준 돌파 전략
# 0.05%
import jwt
import uuid
import math
import pandas as pd
import numpy as np
import os
import requests
import threading
import logging
import traceback
from datetime import datetime, timedelta
from typing import Optional, Tuple
import traceback
import time
from autils.bitcoin.upbit import api
from autils.messenser.telegram import send_message_default

if os.getcwd().find("Library")>-1:
    import sys
    sys.path.append('/Users/kimjinhyung/PycharmProjects/FlaskStock')
    IS_LOCAL_RUN  = True
else:
    IS_LOCAL_RUN  = False
    import sys
    sys.path.append('/home/tariat/FlaskStock')

from autils.db.mysql_utils import MySQL
ms = MySQL()

class MomentumTrader:
    def __init__(self, market: str, mode: str = 'observe', investment_amount: float = 6000, minutes_lag=5):
        self.market = market
        self.mode = mode  # 'observe', 'trade', or 'backtest'
        self.initial_investment = investment_amount
        self.investment_amount = investment_amount
        self.cash_balance = investment_amount
        self.crypto_balance = 0.0
        self.current_position = None
        self.entry_price = None
        self.BUY_FEE = 0.0005  # 0.05%
        self.SELL_FEE = 0.0005  # 0.05%
        self.minutes_lag = minutes_lag
        self.func = self.check_momentum_signal_v1

        print(f"You choose {self.func}")

        self.history_table_nm = f"trading_history_{self.mode}_{self.minutes_lag}"
        self.price_table_nm = f"historical_prices_{self.minutes_lag}"

        if IS_LOCAL_RUN==True:
            from autils import BASE_DIR
            filename=f'{BASE_DIR}/log/momentum_{market}_{datetime.now().strftime("%Y%m%d")}.log'
        else:
            filename=f'./log/momentum_{market}_{datetime.now().strftime("%Y%m%d")}.log'
                
        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        if self.mode in ("observe", "trade"):
            self.init_db()

    def init_db(self):
        """
            데이터베이스 초기화
        """
        ms.execute(f'''CREATE TABLE IF NOT EXISTS {self.history_table_nm} (
            timestamp DATETIME PRIMARY KEY,
            market VARCHAR(20),
            current_price DECIMAL(20,8),
            previous_price DECIMAL(20,8),
            current_ma5 DECIMAL(20,8),
            previous_ma5 DECIMAL(20,8),
            trade_signal VARCHAR(20),
            trade_position VARCHAR(10),
            entry_price DECIMAL(20,8),
            cash_balance DECIMAL(20,8),
            crypto_balance DECIMAL(20,8),
            total_value DECIMAL(20,8),
            total_return_percentage DECIMAL(10,2)
        )''')


    def collect_historical_data(self, iterations: int = 50):
        """
        과거 시세 데이터를 수집하여 DB에 저장
        iterations: 반복 횟수 (한 번에 200개씩 가져옴)
        """
        # 테이블 생성
        ms.execute(f'''CREATE TABLE IF NOT EXISTS {self.price_table_nm} (
            timestamp DATETIME PRIMARY KEY,
            market VARCHAR(20),
            opening_price DECIMAL(20,8),
            high_price DECIMAL(20,8),
            low_price DECIMAL(20,8),
            trade_price DECIMAL(20,8),
            candle_acc_trade_price DECIMAL(20,8),
            candle_acc_trade_volume DECIMAL(20,8)
        )''')

        for i in range(iterations):
            try:
                # 가장 오래된 데이터 시간 조회
                result = ms.read_sql('''
                    SELECT MIN(timestamp) as min_time 
                    FROM ''' + self.price_table_nm + '''
                    WHERE market = %s
                ''', (self.market,))
                
                if result.iloc[0]['min_time'] is None:
                    # 데이터가 없는 경우 현재 시간 기준
                    to_time = datetime.now()
                else:
                    # 있는 경우 가장 오래된 데이터 시간 기준
                    to_time = result.iloc[0]['min_time']

                # API 호출
                url = f"{self.server_url}/v1/candles/minutes/{self.minutes_lag}"
                headers = {"Accept": "application/json"}
                params = {
                    "market": self.market,
                    "count": 200,
                    "to": to_time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                response = requests.get(url, headers=headers, params=params)
                data = response.json()
                
                # 데이터 저장
                for candle in data:
                    ms.execute('''
                        INSERT IGNORE INTO ''' + self.price_table_nm + '''(
                            timestamp, market, opening_price, high_price, low_price, 
                            trade_price, candle_acc_trade_price, candle_acc_trade_volume
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        candle['candle_date_time_utc'],
                        self.market,
                        candle['opening_price'],
                        candle['high_price'],
                        candle['low_price'],
                        candle['trade_price'],
                        candle['candle_acc_trade_price'],
                        candle['candle_acc_trade_volume']
                    ))
                
                logging.info(f"Collected data batch {i+1}/{iterations}")
                
                # API 호출 제한 고려하여 잠시 대기
                time.sleep(0.1)
                
            except Exception as e:
                error_message = traceback.print_exc()
                print(error_message)
                print(f"Error collecting historical data: {str(e)}")
                logging.error(f"Error collecting historical data: {str(e)}")
                break

    # get_historical_data 함수 수정
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """
        백테스트용 과거 데이터 조회 (DB에서)
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        query = '''
            SELECT * FROM '''+ self.price_table_nm +'''
            WHERE market = %s 
            AND timestamp BETWEEN %s AND %s 
            ORDER BY timestamp
        '''
        
        data = ms.read_sql(query, (self.market, from_date, to_date))
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            raise Exception("No historical data found. Please run collect_historical_data first.")
        
        # 5선 이동평균 계산
        df[f'ma5'] = df['trade_price'].rolling(window=5).mean()
        
        return df
    

    def save_to_db(self, timestamp, current_price, previous_price, current_ma5, previous_ma5, signal, total_value, total_return):
        """거래 기록 저장"""
        timestamp_str = str(timestamp)

        if np.isnan(previous_ma5):
            previous_ma5 = None
        
        if np.isnan(current_ma5):
            current_ma5 = None
        
        ms.execute(f'INSERT INTO ' + self.history_table_nm + ''' (
                        timestamp, market, current_price, previous_price, current_ma5, previous_ma5, 
                        trade_signal, trade_position, entry_price, cash_balance, crypto_balance, total_value, 
                        total_return_percentage
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        market = VALUES(market),
                        current_price = VALUES(current_price),
                        previous_price = VALUES(previous_price),
                        current_ma5 = VALUES(current_ma5),
                        previous_ma5 = VALUES(previous_ma5),
                        trade_signal = VALUES(trade_signal),
                        trade_position = VALUES(trade_position),
                        entry_price = VALUES(entry_price),
                        cash_balance = VALUES(cash_balance),
                        crypto_balance = VALUES(crypto_balance),
                        total_value = VALUES(total_value),
                        total_return_percentage = VALUES(total_return_percentage)
                ''', (
                    timestamp_str, self.market, current_price, previous_price, current_ma5, previous_ma5,
                    signal, self.current_position, self.entry_price, self.cash_balance, self.crypto_balance,
                    total_value, total_return
                ))
                

    def plot_backtest_results(self, results_df: pd.DataFrame):
        """백테스트 결과를 Plotly를 사용하여 시각화"""
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Candlestick 차트 생성
        fig = make_subplots(rows=2, cols=1, 
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f'Bitcoin Price with Trading Signals - {self.market}', 
                                         'Portfolio Value & Returns'))

        # 메인 가격 차트
        fig.add_trace(
            go.Scatter(x=results_df['timestamp'], 
                      y=results_df['price'],
                      name='Price',
                      line=dict(color='royalblue', width=1)),
            row=1, col=1
        )
        
        # MA5 라인
        fig.add_trace(
            go.Scatter(x=results_df['timestamp'], 
                      y=results_df['ma5'],
                      name='MA5',
                      line=dict(color='red', width=1)),
            row=1, col=1
        )
        
        # 매수 시그널
        buy_signals = results_df[results_df['signal'] == 'buy']
        fig.add_trace(
            go.Scatter(x=buy_signals['timestamp'],
                      y=buy_signals['price'],
                      name='Buy Signal',
                      mode='markers',
                      marker=dict(symbol='triangle-up',
                                size=10,
                                color='green',
                                line=dict(width=2))),
            row=1, col=1
        )
        
        # 매도 시그널
        sell_signals = results_df[results_df['signal'] == 'sell']
        fig.add_trace(
            go.Scatter(x=sell_signals['timestamp'],
                      y=sell_signals['price'],
                      name='Sell Signal',
                      mode='markers',
                      marker=dict(symbol='triangle-down',
                                size=10,
                                color='red',
                                line=dict(width=2))),
            row=1, col=1
        )
        
        # 포트폴리오 가치 변화
        fig.add_trace(
            go.Scatter(x=results_df['timestamp'],
                      y=results_df['total_value'],
                      name='Portfolio Value',
                      line=dict(color='green', width=1)),
            row=2, col=1
        )
        
        # 수익률
        fig.add_trace(
            go.Scatter(x=results_df['timestamp'],
                      y=results_df['return_pct'],
                      name='Return %',
                      line=dict(color='orange', width=1),
                      visible='legendonly'),  # 기본적으로 숨김
            row=2, col=1
        )

        # 레이아웃 설정
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text=f"Backtest Results - {self.market}",
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Price (KRW)",
            yaxis2_title="Value/Return",
            template='plotly_white',
            hovermode='x unified'
        )
        
        # X축 레이아웃
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        # Y축 레이아웃
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        
        # HTML 파일로 저장
        if IS_LOCAL_RUN==True:
            from autils import BASE_DIR        
            save_path = f'{BASE_DIR}/bitcoin/backtest_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(save_path)
            print(f"Interactive graph saved to: {save_path}")                    
            
            # 브라우저에서 그래프 표시
            fig.show()
        else:
            pass        

    def run_backtest(self, days: int = 30) -> pd.DataFrame:
        """백테스트 실행"""
        if self.mode!="backtest":
            raise Exception("{self.mode} is not backtest mode!")

        print(f"Starting backtest for the last {days} days...")
        ms.execute(f"DROP TABLE IF EXISTS {self.history_table_nm}")

        # 테이블 생성
        self.init_db()
                
        # 히스토리컬 데이터 로드
        df = self.get_historical_data(days)
        print("start: ",df["timestamp"].min())
        print("end: ",df["timestamp"].max())
        results_lst = []
        trades_lst = []        
        
        # 각 시점마다 check_momentum_signal 실행
        for i in range(5, len(df)):  # MA5를 위해 5개 이상의 데이터 필요            
            current_slice = df.iloc[i-5:i+1]  # 현재 점까��의 데이터            
            result, trade = self.func(current_slice)
            results_lst.append(result)
            trades_lst.append(trade)
                
        results_df = pd.DataFrame(results_lst)
        trades_df = pd.DataFrame(trades_lst)
        
        # 백테스트 결과 분석
        self.analyze_backtest_results(results_df, trades_df)
        # 결과 시각화
        self.plot_backtest_results(results_df)
                
        return results_df

    def check_momentum_signal_v3(self, df: pd.DataFrame, kst_timesamp) -> Optional[str]:
        """모멘텀 신호 확인"""
        if len(df) < 2:
            return None
        
        current_price = df.iloc[-1]['trade_price']
        previous_price = df.iloc[-2]['trade_price']
        current_ma5 = df.iloc[-1][f'ma5']
        previous_ma5 = df.iloc[-2][f'ma5']
        
        # 가격 변동률 계산
        price_change = (current_price - previous_price) / previous_price * 100
        ma5_change = (current_ma5 - previous_ma5) / previous_ma5 * 100
        
        signal = 'pass'
        
        if self.current_position is None:  # 매수 시그널
            if (previous_price <= previous_ma5 and 
                current_price > current_ma5 and
                price_change > 0.2 and
                ma5_change > 0):
                
                signal = 'buy'
                self.current_position = 'long'
                self.entry_price = current_price

                if self.mode == 'backtest':
                    self.crypto_balance = self.cash_balance / (current_price * (1 + self.BUY_FEE))
                    self.cash_balance = 0
            
        elif self.current_position == 'long':  # 매도 시그널
            profit_ratio = ((current_price - self.entry_price) / self.entry_price) * 100
            
            if profit_ratio < -1.5:  # 손절
                signal = 'sell'
                if self.mode == 'backtest':
                    # 전량 매도
                    self.cash_balance = self.crypto_balance * (current_price * (1 - self.SELL_FEE))
                    self.crypto_balance = 0
                    self.current_position = None
                    self.entry_price = None
                
            elif profit_ratio > 2.0:  # 이익실현
                signal = 'partial_sell'
                if self.mode == 'backtest':
                    # 50% 매도
                    sell_amount = self.crypto_balance * 0.5
                    self.cash_balance = sell_amount * (current_price * (1 - self.SELL_FEE))
                    self.crypto_balance -= sell_amount
                    # 진입가격을 현재가격으로 업데이트 (남은 물량에 대한 새로운 기준점)
                    self.entry_price = current_price
                
            elif (current_price < current_ma5 and price_change < -0.2):  # 추세 반전
                signal = 'sell'
                if self.mode == 'backtest':
                    # 전량 매도
                    self.cash_balance = self.cash_balance + self.crypto_balance * (current_price * (1 - self.SELL_FEE))
                    self.crypto_balance = 0
                    self.current_position = None
                    self.entry_price = None

        total_value, total_return = self.calculate_total_value(current_price)
        
        if self.mode == 'backtest':
            return signal, total_value, total_return
        
        self.save_to_db(kst_timesamp, current_price, previous_price, current_ma5, previous_ma5, signal, total_value, total_return)
        
        return signal, total_value, total_return


    def check_momentum_signal_v1(self, df: pd.DataFrame, kst_timesamp) -> Optional[str]:
        """모멘텀 신호 확인"""
        if len(df) < 2:
            return None
            
        current_price = df.iloc[-1]['trade_price']
        previous_price = df.iloc[-2]['trade_price']
        current_ma5 = df.iloc[-1]['ma10']
        previous_ma5 = df.iloc[-2]['ma10']
        
        # MA5와의 차이를 백분율로 계산
        price_to_ma5_ratio = ((current_price - current_ma5) / current_ma5) * 100
        
        signal = 'pass'

        if self.current_position is None:  # 포지션이 없을 때만 매수 시그널
            # if previous_price < previous_ma5 and current_price > current_ma5:
            # if previous_price < current_ma5 and current_price > current_ma5:
            if current_price > current_ma5:
                signal = 'buy'
                self.current_position = 'long'
                self.entry_price = current_price

                if self.mode == 'backtest':
                    self.crypto_balance = self.cash_balance / (current_price * (1 + self.BUY_FEE))
                    self.cash_balance = 0
                
        elif self.current_position == 'long':  # 롱 포지션일 때
            # 현재가가 MA5의 1% 이내로 접근했을 때 매도
            # if abs(price_to_ma5_ratio) <= 1.0 and current_price > current_ma5:
            if price_to_ma5_ratio <= 0:
                signal = 'sell'
                self.current_position = None
                self.entry_price = None
                if self.mode == 'backtest':
                    self.cash_balance = self.crypto_balance * (current_price * (1 - self.SELL_FEE))
                    self.crypto_balance = 0

        total_value, total_return = self.calculate_total_value(current_price)
        
        if self.mode == 'backtest':
            return signal, total_value, total_return
        
        self.save_to_db(kst_timesamp, current_price, previous_price, current_ma5, previous_ma5, signal, total_value, total_return)
        print(kst_timesamp, "current_position:", self.current_position, current_price, previous_price, current_ma5
              ,previous_ma5, signal, total_value, total_return)

        return signal, total_value, total_return
                    

    def analyze_backtest_results(self, results_df: pd.DataFrame, trades_df: pd.DataFrame):
        """백테스트 결과 분석"""
        # 전체 수익률
        total_return = results_df['return_pct'].iloc[-1]
        total_value = results_df['total_value'].iloc[-1]
        
        # 연간 수익률
        start_date = results_df['timestamp'].min()
        end_date = results_df['timestamp'].max()
        days = max((end_date - start_date).days, 1)  # 최소 1일로 설정
        annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100
        
        # 최대 낙폭 (Maximum Drawdown)
        cummax = results_df['total_value'].cummax()
        drawdown = (results_df['total_value'] - cummax) / cummax * 100
        mdd = drawdown.min()
        
        # 승률 계산
        if len(trades_df) > 0:
            trade_results = []
            for i in range(0, len(trades_df)-1, 2):
                if i+1 < len(trades_df):
                    buy_price = trades_df.iloc[i]['price']
                    sell_price = trades_df.iloc[i+1]['price']
                    trade_return = (sell_price - buy_price) / buy_price * 100
                    trade_results.append(trade_return)
            
            winning_trades = sum(1 for x in trade_results if x > 0)
            win_rate = (winning_trades / len(trade_results)) * 100 if trade_results else 0
        else:
            win_rate = 0
            trade_results = []
        
        # Sharpe Ratio (무위험 수익률 2% 가정)
        daily_returns = results_df['total_value'].pct_change()
        excess_returns = daily_returns - 0.02/365
        sharpe_ratio = np.sqrt(365) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0

        print("\n=== Backtest Results ===")
        print(f"테스트 기간: {days}일")
        print(f"총 수익률: {total_return:.2f}%")
        print(f"총 자산가치: {total_value:,.0f}원")
        print("----------------------------------")
        print(f"연간 수익률: {annual_return:.2f}%")
        print(f"최대 낙폭 (MDD): {mdd:.2f}%")
        print(f"승률: {win_rate:.2f}%")
        print(f"총 거래 횟수: {len(trade_results)}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print("----------------------------------")
        
        if trade_results:
            print(f"평균 거래당 수익률: {np.mean(trade_results):.2f}%")
            print(f"최대 수익 거래: {max(trade_results):.2f}%")
            print(f"최대 손실 거래: {min(trade_results):.2f}%")            

    def handle_data(self):
        """메인 로직"""
        if self.mode == 'backtest':
            self.run_backtest()
            return
            
        try:
            df = api.get_minutes_candle(self.market, minutes_lag=self.minutes_lag)
            kst_timesamp = df.iloc[-1]['candle_date_time_kst']
            # print(kst_timesamp, end=": ")
            # 시그널 판단
            signal, total_value, total_return = self.func(df, kst_timesamp)
            current_price = df.iloc[-1]['trade_price']
            current_ma5 = df.iloc[-1][f'ma5']
            previous_price = df.iloc[-2]['trade_price']
            previous_ma5 = df.iloc[-2][f'ma5']
            # print(f"Signal: {signal}, Price: {current_price}, MA5: {current_ma5}, total_value: {total_value}, total_return: {total_return}")
            
            if signal in ['buy', 'sell']:
                volume = self.investment_amount / current_price                                
                logging.info(f"Signal: {signal}, Price: {current_price}, MA5: {current_ma5}")
                if self.mode=="trade":
                    send_message_default(f"Signal: {signal}, Price: {current_price}, MA5: {current_ma5}")
                    # 거래 모듈
                    self.execute_order_and_save(kst_timesamp, signal, total_value, total_return
                                       , current_price, current_ma5, previous_price, previous_ma5, volume)

        except Exception as e:            
            print(traceback.format_exc())
            print(e)
            logging.error(f"Error in handle_data: {str(e)}")
        
        if self.mode != 'backtest':
            threading.Timer(60*self.minutes_lag, self.handle_data).start()
    
    def execute_order_and_save(self, kst_timesamp, signal, total_value, total_return, current_price, current_ma5, previous_price, previous_ma5, volume):
        """
            current_slice: 최근 시세 데이터
        """
        candle_date_time_kst = kst_timesamp
        current_time = candle_date_time_kst
                
        # DB에 결과 저장 (실제 캔들 시간 사용)
        self.save_to_db(
            timestamp=current_time,
            current_price=current_price,
            previous_price=previous_price,
            current_ma5=current_ma5,
            previous_ma5=previous_ma5,
            signal=signal,
            total_value=total_value,
            total_return=total_return
        )
        
        if signal in ['buy', 'sell']:
            trade = {
                'timestamp': current_time,
                'type': signal,
                'price': current_price,
                'amount': self.crypto_balance if signal == 'sell' else self.cash_balance/current_price
            }
            if self.mode == "trade" and signal=="buy":
                won = float([rlt["balance"] for rlt in api.get_balance() if rlt.get("currency")=="KRW"][0])
                order_won = int(min(self.cash_balance, won)*0.995)
                rlt = api.order(self.market, "bid", volume=None, price=order_won, ord_type="price")
                print(rlt)
            elif self.mode == "trade" and signal=="sell":
                coin_cla = self.market.split("-")[1]
                coin_balance = float([rlt["balance"] for rlt in api.get_balance() if rlt.get("currency")==coin_cla][0])
                # order_coin = min(coin_balance, trade.get("amount"))
                # order_coin = math.floor(order_coin * 10**7) / 10**7
                # print(order_coin)
                print(coin_balance)
                rlt = api.order(self.market, "ask", volume=coin_balance, price=None, ord_type="market")
                print(rlt)

        result = {
            'timestamp': current_time,
            'price': current_price,
            'ma5': current_ma5,
            'signal': signal,
            'position': self.current_position,
            'cash': self.cash_balance,
            'crypto': self.crypto_balance,
            'total_value': total_value,
            'return_pct': total_return
        }

        return result, trade
    

    def calculate_total_value(self, current_price: float) -> tuple:
        """총 자산가치 및 수익률 계산"""
        crypto_value = self.crypto_balance * current_price
        total_value = self.cash_balance + crypto_value
        total_return = ((total_value - self.initial_investment) / self.initial_investment) * 100
        
        return total_value, total_return
    

if __name__ == '__main__':
    # 1, 3, 5, 10, 30, 60
    # mode: observe, trade, backtest
    # mode = sys.argv[1]

    # # 백테스트 사용 예시
    # mode = "backtest"    
    # trader = MomentumTrader(
    #     market='KRW-BTC',
    #     mode = mode, 

    #     minutes_lag= 60,
    #     investment_amount=10000000  # 1천만원으로 테스트
    # )
    # # # 과거 데이터 수집
    # trader.collect_historical_data(iterations=10)
    # results = trader.run_backtest(days=30)  # 최근 30일 백테스트
    # # results = trader.handle_data()  # 관찰

    # observe 사용 예시
    mode = "trade"    
    trader = MomentumTrader(
        # market='KRW-BTC',
        # market='KRW-TT',
        market='KRW-SOL',
        # market='KRW-AGLD',
        mode = mode, 
        minutes_lag= 3,
        investment_amount=30000  # 1천만원으로 테스트
    )
    result = trader.handle_data()
    