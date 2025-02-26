import requests
from datetime import datetime, timezone, timedelta
import pytz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
import os



def get_upbit_kline(x: float = 5, y: float = 5):
    try:
        market = 'SGD-BTC'
        count = 200  # limitation 200 days on Upbit
        url = f'https://sg-api.upbit.com/v1/candles/days?market={market}&count={count}'
        # url = f'https://sg-api.upbit.com/v1/candles/days?market={market}&count={count}&to=2025-01-25T00:00:00Z'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
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
            # Ensure the timestamp column is in datetime64[ns] format
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            
            # Define the backtest and forward test periods using numpy.datetime64
            backtest_start = np.datetime64('2024-06-18')
            backtest_end = np.datetime64('2024-08-31')
            forward_test_start = np.datetime64('2024-09-01')
            forward_test_end = np.datetime64('2025-01-24')
            # forward_test_end = np.datetime64(datetime.now().strftime('%Y-%m-%d'))

            # Convert signals_df timestamp to numpy.datetime64
            signals_df['timestamp'] = signals_df['timestamp'].astype('datetime64[ns]')
            # signals_df.to_csv('signals_df.csv', index=False)
            # Filter the signals DataFrame
            backtest_signals = signals_df[(signals_df['timestamp'] >= backtest_start) & (signals_df['timestamp'] <= backtest_end)]
            forward_test_signals = signals_df[(signals_df['timestamp'] > forward_test_start) & (signals_df['timestamp'] <= forward_test_end)] 
            return backtest_signals, forward_test_signals
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None, None
    except Exception as e:
        print(f"Error in get_upbit_kline: {str(e)}")
        return None, None

def get_binance_kline(start_time=None, end_time=None):
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1d',
            'limit': 1000
        }

        if start_time or end_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
            # params['endTime'] = int(end_time.timestamp() * 1000)
            params['endTime'] = int(datetime.now(timezone.utc).timestamp() * 1000)

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
    except Exception as e:
        print(f"Error in get_binance_kline: {str(e)}")
        return None

def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
    try:
        print("\nCalculating trading metrics...")
        
        if trades_df.empty:
            print("No trade records, skipping metrics calculation")
            return {}
        
        # Calculate daily returns
        trades_df['daily_returns'] = trades_df['profit_loss_pct'] / 100
        
        # Calculate metrics
        total_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
        
        # CAGR calculation
        total_return = (1 + trades_df['daily_returns']).prod() - 1
        cagr = (1 + total_return) ** (365 / total_days) - 1
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + trades_df['daily_returns']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Sharpe Ratio (assuming risk-free rate is 2%)
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
        
        print(f"Calculation complete: {len(trades_df)} trades")
        return metrics
    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        return {}

def generate_sharpe_heatmap(x_range: List[float], y_range: List[float]) -> None:
    try:
        print("\n=== Starting Sharpe Ratio Heatmap Generation (Backtest Period) ===")
        print(f"Testing range: X (Long signal): {min(x_range)}% to {max(x_range)}%")
        print(f"Testing range: Y (Short signal): {min(y_range)}% to {max(y_range)}%")
        print("=" * 50)
        
        results = []
        total_combinations = len(x_range) * len(y_range)
        current_count = 0
        
        for y in y_range:
            row = []
            for x in x_range:
                current_count += 1
                print(f"\nTesting combination [{current_count}/{total_combinations}]: X={x}%, Y={y}%")
                
                try:
                    # Get signals
                    backtest_signals, _ = get_upbit_kline(x=x, y=y)
                    
                    # Correctly pass parameters to backtesting_kimchi
                    trade_records = backtesting_kimchi(
                        signals_df_upbit=(backtest_signals, None),  # Named parameter
                        take_profit_pct=1.0,
                        stop_loss_pct=1.0,
                        x=x,
                        y=y
                    )
                    
                    if trade_records:
                        trades_df = pd.DataFrame(trade_records)
                        metrics = calculate_metrics(trades_df)
                        sharpe = metrics['Sharpe_Ratio']
                        row.append(sharpe)
                        print(f"Completed calculation: Sharpe ratio = {sharpe:.2f}")
                    else:
                        row.append(np.nan)
                        print("This combination has no trade records")
                except Exception as e:
                    print(f"Error: {str(e)}")
                    row.append(np.nan)
            results.append(row)
        
        print("\n=== Generating Heatmap ===")
        plt.figure(figsize=(14, 10))  # Increase figure size
        sns.heatmap(
            np.array(results).T,  # Transpose matrix
            xticklabels=[f"{x:.1f}%" for x in x_range],  # Adjust labels
            yticklabels=[f"{y:.1f}%" for y in y_range],  # Adjust labels
            annot=True,
            fmt='.2f',
            cmap='RdYlGn'
        )
        plt.title('Sharpe Ratio Heatmap')
        plt.xlabel('X% (price up % percentage)', fontsize=12)  # Adjust font size
        plt.ylabel('Y% (price down % percentage)', fontsize=12)  # Adjust font size
        
        # Save heatmap
        plt.savefig('sharpe_heatmap.png')
        print(f"\nHeatmap saved to: {os.path.abspath('sharpe_heatmap.png')}")
        plt.close()
    except Exception as e:
        print(f"Error in generate_sharpe_heatmap: {str(e)}")

def backtesting_kimchi(signals_df_upbit=None, take_profit_pct: float = 2.0, stop_loss_pct: float = 2.0, x: float = None, y: float = None):
    try:
        if signals_df_upbit is None:
            signals_df_upbit = get_upbit_kline()
        
        # Unpack the tuple - use backtest signals
        backtest_signals, _ = signals_df_upbit
        print(f"\nUsing take profit and stop loss settings:")
        print(f"Take Profit: {take_profit_pct}%")
        print(f"Stop Loss: {stop_loss_pct}%")
        
        trade_records = []
        
        if not backtest_signals.empty:
            for _, signal_row in backtest_signals.iterrows():
                signal_time = signal_row['timestamp']
                position_type = signal_row['position']
                binance_klines = get_binance_kline(signal_time, datetime.now(timezone.utc))
                if binance_klines is None:
                    continue

                for _, binance_row in binance_klines.iterrows():
                    if binance_row['timestamp'] == signal_time:
                        entry_price = binance_row['close']
                        # print(f"Executing {position_type} position at binance {signal_time} with entry price: {entry_price}")
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

        # Modify performance analysis output section
        if trade_records:
            trades_df = pd.DataFrame(trade_records)
            
            # Modify filename to include x and y parameters
            csv_filename = f'trade_records_x{x}_y{y}_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv'
            # trades_df.to_csv(csv_filename, index=False)
            
            # Calculate and display metrics
            metrics = calculate_metrics(trades_df)
            
            print("\n=== Trading Performance Analysis ===")
            print(f"Take Profit: {take_profit_pct}% | Stop Loss: {stop_loss_pct}%")
            print(f"Total trades: {metrics['Total_Trades']}")
            print(f"CAGR: {metrics['CAGR']:.2%}")
            print(f"Maximum Drawdown: {metrics['Max_Drawdown']:.2%}")
            print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
            print(f"Win Rate: {metrics['Win_Rate']:.2%}")
            
        return trade_records
    except Exception as e:
        print(f"Error in backtesting_kimchi: {str(e)}")
        return []

# Example usage for optimization
def optimize_strategy(x: float, y: float, step: float):
    try:
        print("\n=== Starting Strategy Optimization ===")
        x_range = np.arange(x, y, step)  # X% from x to x+2 with step steps
        y_range = np.arange(x, y, step)  # Y% from y to y+2 with step steps
        
        print(f"X range: {x_range}")
        print(f"Y range: {y_range}")
        print(f"Total combinations to test: {len(x_range) * len(y_range)}")
        
        generate_sharpe_heatmap(x_range.tolist(), y_range.tolist())
    except Exception as e:
        print(f"Error in optimize_strategy: {str(e)}")

def forward_testing(x: float, y: float, take_profit_pct: float, stop_loss_pct: float):
    try:
        print("\n=== Starting Forward Test ===")
        
        # Get forward test data
        backtest_signals, forward_test_signals = get_upbit_kline(x=x, y=y)
        # Correctly pass parameters to backtesting_kimchi
        trade_records = backtesting_kimchi(
            signals_df_upbit=(forward_test_signals, None),  # Named parameter
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            x=x,
            y=y
        )
        
        # Calculate and display forward test performance metrics
        if trade_records:
            print(f'trade_records: \n {trade_records}')
            trades_df = pd.DataFrame(trade_records)
            metrics = calculate_metrics(trades_df)
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame([metrics])  # Create a DataFrame from the metrics dictionary
            metrics_csv_filename = f'forward_test_metrics_x{x}_y{y}.csv'
            metrics_df.to_csv(metrics_csv_filename, index=False)
            
            # Modify filename to include x and y parameters for trade records
            csv_filename = f'forward_test_trade_records_x{x}_y{y}_tp{take_profit_pct}_sl{stop_loss_pct}.csv'
            trades_df.to_csv(csv_filename, index=False)
            
            print("\n=== Forward Testing Performance Analysis ===")
            print(f"Total trades: {metrics['Total_Trades']}")
            print(f"CAGR: {metrics['CAGR']:.2%}")
            print(f"Maximum Drawdown: {metrics['Max_Drawdown']:.2%}")
            print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
            print(f"Win Rate: {metrics['Win_Rate']:.2%}")
            print(f'take_profit_pct: {take_profit_pct}')
            print(f'stop_loss_pct: {stop_loss_pct}')
            print(f"Metrics saved to: {os.path.abspath(metrics_csv_filename)}")
            print(f"Forward Trade records saved to: {os.path.abspath(csv_filename)}")
    except Exception as e:
        print(f"Error in forward_testing: {str(e)}")


# run start:
optimize_strategy(3, 5, 0.5)
forward_testing(3.5,4.5)
# forward_testing(3.5,4)
# get_upbit_kline()

# best params x= 3.5, y= 4 in backtest 
# quick spot: I also tested x= 3.5, y= 4.5 in forward test , this is better than pervious best params, please reference in sharpe_heatmap.png


# optimize_strategy(3, 8, 0.5)
# forward_testing(3.2,5.4,10.0,5.0)

