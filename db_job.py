from autils.db.mysql_utils import MySQL

def get_table_name(market, mode, minutes_lag):
    market_nm = market.replace("-","_")
    trading_table_nm = f"trading_history_{market_nm}_{mode}_{minutes_lag}"
    history_table_nm = f"historical_prices_{market_nm}_{mode}_{minutes_lag}"

    return trading_table_nm, history_table_nm

def init_db(market, mode, minutes_lag):
    """
    market: KRW-BTC
    mode: backtest, real, trade
    minutes_lag: 5, 3
    """

    trading_table_nm, history_table_nm = get_table_name(market, mode, minutes_lag)

    ms = MySQL()
    ms.execute(f'''CREATE TABLE IF NOT EXISTS {trading_table_nm} (
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

    ms.execute(f'''CREATE TABLE IF NOT EXISTS {history_table_nm} (
        timestamp DATETIME PRIMARY KEY,
        market VARCHAR(20),
        opening_price DECIMAL(20,8),
        high_price DECIMAL(20,8),
        low_price DECIMAL(20,8),
        trade_price DECIMAL(20,8),
        candle_acc_trade_price DECIMAL(20,8),
        candle_acc_trade_volume DECIMAL(20,8)
    )''')

        
if __name__=="__main__":
    print("main")
