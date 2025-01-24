# a. b. c. d. e. 
# When Bitcoin goes up X% on Upbit Exchange, Long Bitcoin in Binance; when Bitcoin goes

# ans:
# Strategy Description:
# 1. Strategy Objective:
# When Bitcoin rises more than X% on the Upbit exchange, go long on Bitcoin on Binance.
# When Bitcoin falls more than Y% on the Upbit exchange, go short on Bitcoin on Binance.
# Backtesting Process:
# First, use the get_upbit_kline() function to obtain kline data from Upbit.
# Calculate the percentage change in price for each time period and identify long and short signals.
# For each long signal, record the start and end time of the signal.
# Use the get_binance_kline() function to obtain kline data from Binance for the corresponding time range.
# Loop through the kline data from Binance to simulate the execution of the trading strategy. down Y% on UpBit exchange, Short Bitcoin in Binance.

# You need to collect price of Bitcoin Perpetual Futures from UpBit & Binance respectively.
# Divide the whole data period into 2 parts. 50% of the time for backtest another 50% for
# forward test. For example, you have data from 01/2021 to 01/2023. Then 01/2021 to 01/2022 is used for backtest. 01/2022 to 01/2023 is used for forward test.
# Generate a sharpe heat map to loop and find the best X & Y combination. (like image above)
# Calculate CAGR, Maximum Drawdown and Sharpe Ratio


import requests
from datetime import datetime, timezone
import pytz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
import os



def get_upbit_kline(x: float = 5, y: float = 5):
    market = 'SGD-BTC'
    count = 200  # limitation 200days on upbit
    url = f'https://sg-api.upbit.com/v1/candles/days?market={market}&count={count}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f'Length of data: {len(data)}')
        
        # Reverse data to chronological order
        data = data[::-1]
        previous_close = None
        signals = []
        
        for candle in data:
            timestamp = candle.get('candle_date_time_utc')
            candle_time = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
            utc_time = candle_time

            current_close = candle['trade_price']
            if previous_close is not None:
                change_percentage = ((current_close - previous_close) / previous_close) * 100

                if change_percentage > x:
                    signals.append({
                        'timestamp': utc_time,
                        'price': current_close,
                        'position': 'long',
                        'change': change_percentage
                    })
                elif change_percentage < -y:
                    signals.append({
                        'timestamp': utc_time,
                        'price': current_close,
                        'position': 'short',
                        'change': change_percentage
                    })
            previous_close = current_close
          
        # Convert signals to DataFrame
        signals_df = pd.DataFrame(signals)
        return signals_df
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return pd.DataFrame()  # Return empty DataFrame in case of error

def get_binance_kline(start_time=None, end_time=None):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1d',
        'limit': 1000
    }

    if start_time or end_time:
        params['startTime'] = int(start_time.timestamp() * 1000)
        params['endTime'] = int(end_time.timestamp() * 1000)

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Convert string values to float for price columns
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        return df
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
    """计算交易性能指标"""
    print("\n计算交易指标...")
    
    if trades_df.empty:
        print("无交易记录，跳过指标计算")
        return {}
    
    # 计算每日收益
    trades_df['daily_returns'] = trades_df['profit_loss_pct'] / 100
    
    # 计算指标
    total_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
    
    # CAGR计算
    total_return = (1 + trades_df['daily_returns']).prod() - 1
    cagr = (1 + total_return) ** (365 / total_days) - 1
    
    # 计算最大回撤
    cumulative_returns = (1 + trades_df['daily_returns']).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # 夏普比率 (假设无风险利率为2%)
    rf_rate = 0.02
    excess_returns = trades_df['daily_returns'] - (rf_rate / 365)
    sharpe_ratio = np.sqrt(365) * (excess_returns.mean() / excess_returns.std())
    
    metrics = {
        'CAGR': cagr,
        'Max_Drawdown': max_drawdown,
        'Sharpe_Ratio': sharpe_ratio,
        'Total_Trades': len(trades_df),
        'Win_Rate': (trades_df['reason'] == 'take_profit').mean()
    }
    
    print(f"计算完成: {len(trades_df)}笔交易")
    return metrics

def generate_sharpe_heatmap(x_range: List[float], y_range: List[float]) -> None:
    """生成并保存夏普比率热力图"""
    print("\n=== 开始生成夏普比率热力图 ===")
    print(f"测试范围: X (做多信号): {min(x_range)}% 到 {max(x_range)}%")
    print(f"测试范围: Y (做空信号): {min(y_range)}% 到 {max(y_range)}%")
    print("=" * 50)
    
    results = []
    total_combinations = len(x_range) * len(y_range)
    current_count = 0
    
    for y in y_range:  # 反转循环顺序
        row = []
        for x in x_range:  # 反转循环顺序
            current_count += 1
            print(f"\n测试组合 [{current_count}/{total_combinations}]: X={x}%, Y={y}%")
            
            try:
                signals_df_upbit = get_upbit_kline(x=x, y=y)
                trade_records = backtesting_kimchi(
                    signals_df_upbit, 
                    take_profit_pct=2.0,
                    stop_loss_pct=2.0
                )
                
                if trade_records:
                    trades_df = pd.DataFrame(trade_records)
                    metrics = calculate_metrics(trades_df)
                    sharpe = metrics['Sharpe_Ratio']
                    row.append(sharpe)
                    print(f"完成计算: Sharpe比率 = {sharpe:.2f}")
                else:
                    row.append(np.nan)
                    print("此组合无交易记录")
            except Exception as e:
                print(f"错误: {str(e)}")
                row.append(np.nan)
        results.append(row)
    
    print("\n=== 生成热力图 ===")
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        np.array(results).T,  # 转置矩阵
        xticklabels=[f"{x:.1f}%" for x in x_range],  # 调整标签
        yticklabels=[f"{y:.1f}%" for y in y_range],  # 调整标签
        annot=True,
        fmt='.2f',
        cmap='RdYlGn'
    )
    plt.title('Sharpe Ratio Heatmap (夏普比率热力图)')
    plt.xlabel('X% (做多信号百分比)')  # 调整标签
    plt.ylabel('Y% (做空信号百分比)')  # 调整标签
    
    # 保存热力图
    plt.savefig('sharpe_heatmap.png')
    print(f"\n热力图已保存至: {os.path.abspath('sharpe_heatmap.png')}")
    plt.close()

def backtesting_kimchi(signals_df_upbit=None, take_profit_pct: float = 2.0, stop_loss_pct: float = 2.0):
    """
    回测交易策略
    
    Args:
        signals_df_upbit: 交易信号数据
        take_profit_pct: 止盈百分比 (默认2%)
        stop_loss_pct: 止损百分比 (默认2%)
    """
    if signals_df_upbit is None:
        signals_df_upbit = get_upbit_kline()
    
    print(f"\n使用止盈止损设置:")
    print(f"止盈: {take_profit_pct}%")
    print(f"止损: {stop_loss_pct}%")
    
    trade_records = []
    
    if not signals_df_upbit.empty:
        for _, signal_row in signals_df_upbit.iterrows():
            signal_time = signal_row['timestamp']
            position_type = signal_row['position']
            
            binance_klines = get_binance_kline(signal_time, datetime.now(timezone.utc))
            if binance_klines is None:
                continue

            for _, binance_row in binance_klines.iterrows():
                if binance_row['timestamp'] == signal_time:
                    entry_price = binance_row['close']
                    # print(f"Executing {position_type} position at {signal_time} with entry price: {entry_price}")
                    
                    trade_record = {
                        'entry_time': signal_time,
                        'position': position_type,
                        'entry_price': entry_price,
                        'exit_time': None,
                        'exit_price': None,
                        'profit_loss_pct': None,
                        'reason': None
                    }
                
                elif entry_price:
                    tp_multiplier = 1 + (take_profit_pct / 100)
                    sl_multiplier = 1 - (stop_loss_pct / 100)
                    
                    if position_type == 'long':
                        # Long position take profit
                        if binance_row['close'] >= entry_price * tp_multiplier:
                            trade_record.update({
                                'exit_time': binance_row['timestamp'],
                                'exit_price': binance_row['close'],
                                'profit_loss_pct': ((binance_row['close'] - entry_price) / entry_price) * 100,
                                'reason': 'take_profit'
                            })
                            trade_records.append(trade_record)
                            break
                        # Long position stop loss
                        elif binance_row['close'] <= entry_price * sl_multiplier:
                            trade_record.update({
                                'exit_time': binance_row['timestamp'],
                                'exit_price': binance_row['close'],
                                'profit_loss_pct': ((binance_row['close'] - entry_price) / entry_price) * 100,
                                'reason': 'stop_loss'
                            })
                            trade_records.append(trade_record)
                            break
                    else:  # short position
                        # Short position take profit
                        if binance_row['close'] <= entry_price * sl_multiplier:
                            trade_record.update({
                                'exit_time': binance_row['timestamp'],
                                'exit_price': binance_row['close'],
                                'profit_loss_pct': ((entry_price - binance_row['close']) / entry_price) * 100,
                                'reason': 'take_profit'
                            })
                            trade_records.append(trade_record)
                            break
                        # Short position stop loss
                        elif binance_row['close'] >= entry_price * tp_multiplier:
                            trade_record.update({
                                'exit_time': binance_row['timestamp'],
                                'exit_price': binance_row['close'],
                                'profit_loss_pct': ((entry_price - binance_row['close']) / entry_price) * 100,
                                'reason': 'stop_loss'
                            })
                            trade_records.append(trade_record)
                            break

    # 修改性能分析输出部分
    if trade_records:
        trades_df = pd.DataFrame(trade_records)
        
        # 保存交易记录
        csv_filename = f'trade_records_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv'
        trades_df.to_csv(csv_filename, index=False)
        
        # 计算并显示指标
        metrics = calculate_metrics(trades_df)
        
        print("\n=== Trading Performance Analysis ===")
        print(f"Take Profit: {take_profit_pct}% | Stop Loss: {stop_loss_pct}%")  # 现在会显示正确的2%
        print(f"Total trades: {metrics['Total_Trades']}")
        print(f"CAGR: {metrics['CAGR']:.2%}")
        print(f"Maximum Drawdown: {metrics['Max_Drawdown']:.2%}")
        print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
        print(f"Win Rate: {metrics['Win_Rate']:.2%}")
        
    return trade_records

# Example usage for optimization
def optimize_strategy():
    """优化策略参数"""
    print("\n=== 开始策略优化 ===")
    x_range = np.arange(3, 5, 0.5)  # X% from 3% to 5% with 0.5% steps
    y_range = np.arange(3, 5, 0.5)  # Y% from 3% to 5% with 0.5% steps
    
    print(f"X范围: {x_range}")
    print(f"Y范围: {y_range}")
    print(f"总共需要测试 {len(x_range) * len(y_range)} 种组合")
    
    generate_sharpe_heatmap(x_range.tolist(), y_range.tolist())


optimize_strategy()

# Run
# signals_df = get_upbit_kline()
# print(f"Long Signal: {long_signal}")
# convert a
# print(f"Short Signal: {short_signal}")

# if not signals_df.empty:
#     start_time = signals_df['timestamp'].iloc[0]
#     end_time = datetime.utcnow()
#     get_binance_kline(start_time, end_time)
