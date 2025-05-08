import jwt
import hashlib
import os
import requests
import time
import uuid
from urllib.parse import urlencode, unquote

import logging
import pandas as pd
from tqdm import tqdm


class UpBit():
    def __init__(self, access_key, secret_key):        
        self.server_url = 'https://api.upbit.com'
        self.access_key = access_key
        self.secret_key = secret_key        
    
    def get_market(self):
        url = f"{self.server_url}/v1/market/all"
        headers = {"Accept": "application/json"}
        params = {"is_details": "true"}

        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        return data
    
    def get_min_gap_market(self, cutoff=None):
        data = self.get_market()
        df = pd.DataFrame(data)
        df = df[df['market'].str.contains('KRW')]

        price_dict = {}
        min_gap = 999
        min_market = None

        for market in tqdm(df['market'].to_list()):
            price = self.get_minutes_candle(market,20,minutes_lag=3)
            if isinstance(price, pd.DataFrame) == False:
                continue
            last_price = price.iloc[-1]

            if last_price['trade_price'] > last_price['ma5'] and last_price['ma5'] > last_price['ma10'] and last_price['ma10'] > last_price['ma20']:
                price_20_ratio = last_price['trade_price'] / last_price['ma20'] -1
                price_dict.update({market : price_20_ratio})

                if cutoff!=None:
                    if price_20_ratio < cutoff:
                        return market, price_20_ratio, price_dict
                    
                if min_gap > price_20_ratio:
                    min_gap = price_20_ratio
                    min_market = market
            
            time.sleep(0.1)

        return min_market, min_gap, price_dict


    def get_minutes_candle(self, market, count: int = 100, minutes_lag=3) -> pd.DataFrame:
        """3분봉 데이터 조회
        minutes_lag: 몇 분봉인지
        """
        url = f"{self.server_url}/v1/candles/minutes/{minutes_lag}"
        headers = {"Accept": "application/json"}
        params = {"market": market, "count": count}

        response = requests.get(url, headers=headers, params=params)
        if response.status_code==400:
            return None
                
        data = response.json()
        try:
            df = pd.DataFrame(data)
        except:
            print(data)
            if data.get("name")=="too_many_requests":
                time.sleep(1)
            return None
        
        df = df.sort_values("candle_date_time_kst", ascending=True).reset_index(drop=True)        

        # 5선 이평선
        df['ma5'] = df['trade_price'].rolling(window=5).mean()
        df['ma10'] = df['trade_price'].rolling(window=10).mean()
        df['ma20'] = df['trade_price'].rolling(window=20).mean()

        return df

    def execute_trade(self, market, signal: str, price: float, volume: float):
        """실제 거래 실행"""
        if signal == 'buy':
            self.order(market, 'bid', volume, price)
        elif signal == 'sell':
            self.order(market, 'ask', volume, price)

    def order(self, market: str, side: str, volume: float, price: float, ord_type="limit"):
        """업비트 주문 API
        side - bid:매수, ask:매도
        volume - 주문량
        price - 주문가격
        ord_type - limit: 지정가 주문, price: 시장가, market: 시장가(매도), best: 최유리(time_in_force 설정 필수)
        """
        params = {
            'market': market,
            'side': side,
            'volume': str(volume),
            'price': str(price),
            'ord_type': ord_type,
        }
        if ord_type == "price":
            params.pop("volume", None)
        elif ord_type == "market":
            params.pop("price", None)

        query_string = unquote(urlencode(params, doseq=True)).encode("utf-8")

        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512',
        }

        jwt_token = jwt.encode(payload, self.secret_key)
        authorization = 'Bearer {}'.format(jwt_token)
        headers = {
        'Authorization': authorization,
        }

        try:
            res = requests.post(self.server_url + '/v1/orders', json=params, headers=headers)            
            return res.json()
        except Exception as e:
            logging.error(f"Order failed: {str(e)}")
            return None
    
    def get_balance(self):
        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
        }

        jwt_token = jwt.encode(payload, self.secret_key)
        authorization = 'Bearer {}'.format(jwt_token)
        headers = {
        'Authorization': authorization,
        }

        res = requests.get(self.server_url + '/v1/accounts', headers=headers)
        res.json()

        return res.json()
    
        
if __name__=="__main__":
    from autils import config_dir
    from autils.secrets import Config
    cf = Config(config_dir)

    upbit_access_key = cf.get_config_value("UPBIT_ACCESS_KEY")
    upbit_secret_key = cf.get_config_value("UPBIT_SECRET_KEY")

    from autils.bitcoin.__upbit__ import UpBit
    api = UpBit(access_key=upbit_access_key, secret_key=upbit_secret_key)
    print(api.get_minutes_candle())
    
