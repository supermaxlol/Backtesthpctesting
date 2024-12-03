import os
import pandas as pd
import numpy as np
import talib
from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import warnings
import logging
from datetime import datetime, timedelta
import multiprocessing as mp

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_historical_data(directory_path):
    """Load all historical stock data from CSV files in the specified directory."""
    all_data = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            stock_code = os.path.splitext(filename)[0].replace('_data', '')
            file_path = os.path.join(directory_path, filename)
            try:
                df = pd.read_csv(file_path)
                df = validate_and_process_data(df, filename)
                all_data[stock_code] = df
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
    
    if not all_data:
        raise ValueError("No valid data files found in the directory.")
    return all_data

def validate_and_process_data(df, filename):
    """Validate and process the loaded data."""
    df.rename(columns=lambda x: x.strip().lower(), inplace=True)
    required_columns = ['date', 'close', 'open', 'high', 'low', 'volume']
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"{filename} is missing required columns.")
    
    df.rename(columns={
        'date': 'Date',
        'close': 'Close',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'volume': 'Volume'
    }, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    name: str
    function: callable
    params: dict
    standard_params: dict  # Added standard parameters
    signal_thresholds: List[float] = None
    comparison_type: str = 'threshold'

class TechnicalIndicatorGenerator:
    def __init__(self):
        self.indicators = self._initialize_indicators()
        
    def _initialize_indicators(self) -> Dict[str, IndicatorConfig]:
        """Initialize all technical indicators with their configurations"""
        indicators = {
            'SMA': IndicatorConfig(
                name='SMA',
                function=talib.SMA,
                params={'timeperiod': list(range(5, 201, 5))},
                standard_params={'timeperiod': 20},
                comparison_type='crossover'
            ),
            'EMA': IndicatorConfig(
                name='EMA',
                function=talib.EMA,
                params={'timeperiod': list(range(5, 201, 5))},
                standard_params={'timeperiod': 20},
                comparison_type='crossover'
            ),
            'RSI': IndicatorConfig(
                name='RSI',
                function=talib.RSI,
                params={'timeperiod': [7, 14, 21, 28]},
                standard_params={'timeperiod': 14},
                signal_thresholds=[20, 30, 70, 80],
                comparison_type='range'
            ),
            'MACD': IndicatorConfig(
                name='MACD',
                function=talib.MACD,
                params={
                    'fastperiod': [8, 12, 16],
                    'slowperiod': [21, 26, 31],
                    'signalperiod': [7, 9, 11]
                },
                standard_params={
                    'fastperiod': 12,
                    'slowperiod': 26,
                    'signalperiod': 9
                },
                comparison_type='crossover'
            ),
            'BBANDS': IndicatorConfig(
                name='BBANDS',
                function=talib.BBANDS,
                params={
                    'timeperiod': [10, 20, 30],
                    'nbdevup': [1.5, 2.0, 2.5],
                    'nbdevdn': [1.5, 2.0, 2.5]
                },
                standard_params={
                    'timeperiod': 20,
                    'nbdevup': 2.0,
                    'nbdevdn': 2.0
                },
                comparison_type='range'
            ),
            'STOCH': IndicatorConfig(
                name='STOCH',
                function=talib.STOCH,
                params={
                    'fastk_period': [5, 14, 21],
                    'slowk_period': [3, 5, 7],
                    'slowd_period': [3, 5, 7]
                },
                standard_params={
                    'fastk_period': 14,
                    'slowk_period': 3,
                    'slowd_period': 3
                },
                signal_thresholds=[20, 80],
                comparison_type='range'
            ),
            'ADX': IndicatorConfig(
                name='ADX',
                function=talib.ADX,
                params={'timeperiod': [7, 14, 21, 28]},
                standard_params={'timeperiod': 14},
                signal_thresholds=[20, 25, 30],
                comparison_type='threshold'
            ),
            'CCI': IndicatorConfig(
                name='CCI',
                function=talib.CCI,
                params={'timeperiod': [14, 20, 30]},
                standard_params={'timeperiod': 20},
                signal_thresholds=[-100, 100],
                comparison_type='range'
            ),
            'MFI': IndicatorConfig(
                name='MFI',
                function=talib.MFI,
                params={'timeperiod': [7, 14, 21]},
                standard_params={'timeperiod': 14},
                signal_thresholds=[20, 80],
                comparison_type='range'
            ),
            'OBV': IndicatorConfig(
                name='OBV',
                function=talib.OBV,
                params={},
                standard_params={},
                comparison_type='crossover'
            ),
            'WILLR': IndicatorConfig(
                name='WILLR',
                function=talib.WILLR,
                params={'timeperiod': [7, 14, 21]},
                standard_params={'timeperiod': 14},
                signal_thresholds=[-80, -20],
                comparison_type='range'
            ),
            'ROC': IndicatorConfig(
                name='ROC',
                function=talib.ROC,
                params={'timeperiod': [5, 10, 15, 20]},
                standard_params={'timeperiod': 10},
                signal_thresholds=[-5, 5],
                comparison_type='range'
            )
        }
        return indicators
class StrategyGenerator:
    def __init__(self, indicator_generator: TechnicalIndicatorGenerator):
        self.indicator_generator = indicator_generator
        
    def generate_indicator_signals(self, data: pd.DataFrame, indicator: IndicatorConfig, 
                                 params: dict) -> pd.Series:
        """Generate trading signals for a single indicator"""
        high, low, close, volume = data['High'], data['Low'], data['Close'], data['Volume']
        
        if indicator.name == 'MACD':
            macd, signal, hist = indicator.function(close, **params)
            return pd.Series(np.where(macd > signal, 1, np.where(macd < signal, -1, 0)), 
                           index=close.index)
            
        elif indicator.name == 'BBANDS':
            upper, middle, lower = indicator.function(close, **params)
            return pd.Series(np.where(close < lower, 1, 
                                    np.where(close > upper, -1, 0)), 
                           index=close.index)
            
        elif indicator.name in ['SMA', 'EMA']:
            ma = indicator.function(close, **params)
            return pd.Series(np.where(close > ma, 1, 
                                    np.where(close < ma, -1, 0)), 
                           index=close.index)
            
        elif indicator.name == 'RSI':
            rsi = indicator.function(close, **params)
            return pd.Series(np.where(rsi < indicator.signal_thresholds[0], 1,
                                    np.where(rsi > indicator.signal_thresholds[-1], -1, 0)),
                           index=close.index)
                           
        elif indicator.name == 'STOCH':
            slowk, slowd = indicator.function(high, low, close, **params)
            return pd.Series(np.where(slowk < indicator.signal_thresholds[0], 1,
                                    np.where(slowk > indicator.signal_thresholds[1], -1, 0)),
                           index=close.index)
                           
        elif indicator.name == 'ADX':
            adx = indicator.function(high, low, close, **params)
            return pd.Series(np.where(adx > indicator.signal_thresholds[1], 1,
                                    np.where(adx < indicator.signal_thresholds[0], -1, 0)),
                           index=close.index)
                           
        elif indicator.name == 'CCI':
            cci = indicator.function(high, low, close, **params)
            return pd.Series(np.where(cci < indicator.signal_thresholds[0], 1,
                                    np.where(cci > indicator.signal_thresholds[1], -1, 0)),
                           index=close.index)
                           
        elif indicator.name == 'MFI':
            mfi = indicator.function(high, low, close, volume, **params)
            return pd.Series(np.where(mfi < indicator.signal_thresholds[0], 1,
                                    np.where(mfi > indicator.signal_thresholds[1], -1, 0)),
                           index=close.index)
                           
        elif indicator.name == 'OBV':
            obv = indicator.function(close, volume)
            obv_ma = talib.SMA(obv, timeperiod=20)  # Use 20-period MA of OBV
            return pd.Series(np.where(obv > obv_ma, 1,
                                    np.where(obv < obv_ma, -1, 0)),
                           index=close.index)
                           
        elif indicator.name == 'WILLR':
            willr = indicator.function(high, low, close, **params)
            return pd.Series(np.where(willr < indicator.signal_thresholds[0], 1,
                                    np.where(willr > indicator.signal_thresholds[1], -1, 0)),
                           index=close.index)
                           
        elif indicator.name == 'ROC':
            roc = indicator.function(close, **params)
            return pd.Series(np.where(roc < indicator.signal_thresholds[0], -1,
                                    np.where(roc > indicator.signal_thresholds[1], 1, 0)),
                           index=close.index)
                           
        elif indicator.name == 'TRIX':
            trix = indicator.function(close, **params)
            trix_signal = talib.SMA(trix, timeperiod=9)  # 9-period signal line
            return pd.Series(np.where(trix > trix_signal, 1,
                                    np.where(trix < trix_signal, -1, 0)),
                           index=close.index)
                           
        elif indicator.name == 'ULTOSC':
            ultosc = indicator.function(high, low, close, **params)
            return pd.Series(np.where(ultosc < indicator.signal_thresholds[0], 1,
                                    np.where(ultosc > indicator.signal_thresholds[1], -1, 0)),
                           index=close.index)
                           
        elif indicator.name == 'PPO':
            ppo = indicator.function(close, **params)
            ppo_signal = talib.SMA(ppo, timeperiod=9)  # 9-period signal line
            return pd.Series(np.where(ppo > ppo_signal, 1,
                                    np.where(ppo < ppo_signal, -1, 0)),
                           index=close.index)
            
        return pd.Series(0, index=close.index)

    def generate_combined_strategy(self, data: pd.DataFrame, 
                                 indicator_combinations: List[Tuple[str, dict]]) -> pd.Series:
        """Generate trading signals by combining multiple indicators"""
        signals = pd.DataFrame(index=data.index)
        
        for ind_name, params in indicator_combinations:
            indicator = self.indicator_generator.indicators[ind_name]
            signals[ind_name] = self.generate_indicator_signals(data, indicator, params)
        
        combined_signals = signals.mean(axis=1)
        return pd.Series(np.where(combined_signals > 0.3, 1,
                                np.where(combined_signals < -0.3, -1, 0)),
                        index=data.index)

class StrategyTester:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        
    def calculate_metrics(self, trades: List[Dict], final_value: float, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'num_trades': 0
            }
            
        # Calculate daily returns for Sharpe ratio
        daily_returns = []
        portfolio_value = pd.Series(index=data.index, dtype=float)
        portfolio_value.iloc[0] = self.initial_capital
        
        current_position = 0
        current_shares = 0
        
        for trade in trades:
            date_idx = data.index.get_loc(trade['date'])
            if trade['type'] == 'buy':
                current_position = trade['cost']
                current_shares = trade['shares']
            else:  # sell
                daily_returns.append(trade['revenue'] / current_position - 1)
                current_position = 0
                current_shares = 0
            
            # Update portfolio value
            if current_position > 0:
                portfolio_value.iloc[date_idx] = current_shares * data['Close'].iloc[date_idx]
            else:
                portfolio_value.iloc[date_idx] = trade['revenue'] if trade['type'] == 'sell' else trade['cost']
        
       
        
        # Fill forward portfolio value
        portfolio_value.ffill(inplace=True)
        portfolio_value.fillna(self.initial_capital, inplace=True)
        
        # Calculate metrics
        returns = np.array(daily_returns)
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0
            win_rate = np.sum(returns > 0) / len(returns)
            
            winning_trades = returns[returns > 0].sum() if len(returns[returns > 0]) > 0 else 0
            losing_trades = abs(returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else 1
            profit_factor = winning_trades / losing_trades if losing_trades > 0 else 0
            
            # Calculate maximum drawdown
            rolling_max = portfolio_value.expanding().max()
            drawdowns = (portfolio_value - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
        else:
            sharpe_ratio = 0
            win_rate = 0
            profit_factor = 0
            max_drawdown = 0
            
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(trades),
            'final_value': final_value
        }
        
    def backtest_strategy(self, data: pd.DataFrame, signals: pd.Series,
                         transaction_cost: float = 0.001) -> Dict:
        """Backtest a trading strategy with improved position tracking"""
        position = 0  # -1: short, 0: neutral, 1: long
        capital = self.initial_capital
        holdings = 0
        trades = []
        
        for i in range(1, len(signals)):  # Start from second day to avoid lookback bias
            current_price = data['Close'].iloc[i]
            current_signal = signals.iloc[i]
            
            # Only trade if we have a signal and it's different from current position
            if current_signal != position:
                if current_signal == 1:  # Buy signal
                    shares = (capital * 0.95) // current_price  # Use 95% of capital for positions
                    if shares > 0:
                        cost = shares * current_price * (1 + transaction_cost)
                        if cost <= capital:
                            holdings = shares
                            capital -= cost
                            position = 1
                            trades.append({
                                'date': data.index[i],
                                'type': 'buy',
                                'price': current_price,
                                'shares': shares,
                                'cost': cost
                            })
                
                elif current_signal == -1 and holdings > 0:  # Sell signal
                    revenue = holdings * current_price * (1 - transaction_cost)
                    capital += revenue
                    trades.append({
                        'date': data.index[i],
                        'type': 'sell',
                        'price': current_price,
                        'shares': holdings,
                        'revenue': revenue
                    })
                    holdings = 0
                    position = -1
        
        # Force close any remaining position at the end
        if holdings > 0:
            final_price = data['Close'].iloc[-1]
            revenue = holdings * final_price * (1 - transaction_cost)
            capital += revenue
            trades.append({
                'date': data.index[-1],
                'type': 'sell',
                'price': final_price,
                'shares': holdings,
                'revenue': revenue
            })
        
        final_value = capital
        metrics = self.calculate_metrics(trades, final_value, data)
        metrics['trades'] = trades
        metrics['final_value'] = final_value
        
        return metrics

def process_strategy(args):
    """Process a single strategy combination"""
    data, indicator_combo, params_combo, strategy_gen, tester = args
    
    # Combine indicators with their parameters
    indicator_params = []
    for ind, params in zip(indicator_combo, params_combo):
        indicator_params.append((ind, params))
    
    # Generate and test strategy
    signals = strategy_gen.generate_combined_strategy(data, indicator_params)
    results = tester.backtest_strategy(data, signals)
    
    results['indicators'] = indicator_combo
    results['params'] = indicator_params
    return results

def generate_parameter_combinations(indicator_combo, indicator_gen, use_standard_params=False):
    """Generate parameter combinations based on mode"""
    if use_standard_params:
        return [tuple(indicator_gen.indicators[ind].standard_params for ind in indicator_combo)]
    else:
        param_combinations = []
        for ind in indicator_combo:
            indicator = indicator_gen.indicators[ind]
            param_keys = list(indicator.params.keys())
            param_values = list(indicator.params.values())
            if param_values:
                param_combinations.append([dict(zip(param_keys, v)) for v in product(*param_values)])
            else:
                param_combinations.append([{}])
        return list(product(*param_combinations))
def analyze_strategies(results: List[Dict]) -> pd.DataFrame:
    """Analyze and rank strategies based on multiple metrics"""
    analysis = []
    for result in results:
        if result['total_return'] == 0:  # Skip strategies with no trades
            continue
            
        analysis.append({
            'indicators': ','.join(result['indicators']),
            'total_return': result['total_return'],
            'sharpe_ratio': result['sharpe_ratio'],
            'max_drawdown': result['max_drawdown'],
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'num_trades': result['num_trades'],
            'final_value': result['final_value'],
            'parameters': str(result['params'])
        })
    
    if not analysis:
        return pd.DataFrame()
        
    df = pd.DataFrame(analysis)
    
    # Calculate composite score
    df['score'] = (
        df['sharpe_ratio'].rank(pct=True) * 0.3 +
        df['total_return'].rank(pct=True) * 0.3 +
        df['win_rate'].rank(pct=True) * 0.2 +
        df['profit_factor'].rank(pct=True) * 0.2 -
        df['max_drawdown'].rank(pct=True) * 0.2
    )
    
    return df.sort_values('score', ascending=False)
def analyze_strategies(results: List[Dict], stock_code: str = None) -> pd.DataFrame:
    """Analyze and rank strategies based on multiple metrics"""
    analysis = []
    for result in results:
        if result['total_return'] == 0:  # Skip strategies with no trades
            continue
            
        strategy_key = (
            ','.join(result['indicators']),
            str(result['params'])
        )
            
        analysis.append({
            'stock_code': stock_code,
            'strategy_key': str(strategy_key),  # Convert tuple to string for DataFrame
            'indicators': ','.join(result['indicators']),
            'parameters': str(result['params']),
            'total_return': result['total_return'],
            'sharpe_ratio': result['sharpe_ratio'],
            'max_drawdown': result['max_drawdown'],
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'num_trades': result['num_trades'],
            'final_value': result['final_value']
        })
    
    if not analysis:
        return pd.DataFrame()
        
    df = pd.DataFrame(analysis)
    
    # Calculate composite score
    df['score'] = (
        df['sharpe_ratio'].rank(pct=True) * 0.3 +
        df['total_return'].rank(pct=True) * 0.3 +
        df['win_rate'].rank(pct=True) * 0.2 +
        df['profit_factor'].rank(pct=True) * 0.2 -
        df['max_drawdown'].rank(pct=True) * 0.2
    )
    
    return df

def create_master_report(all_results: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a master report analyzing strategy performance across all stocks
    
    Parameters:
    all_results: Dict mapping stock codes to their strategy analysis DataFrames
    
    Returns:
    Tuple of (strategy_summary, top_strategies, stock_summary)
    """
    # Combine all results
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    
    # 1. Strategy Performance Summary
    strategy_summary = combined_df.groupby(['indicators', 'parameters']).agg({
        'total_return': ['mean', 'std', 'min', 'max'],
        'sharpe_ratio': ['mean', 'std', 'min', 'max'],
        'win_rate': ['mean', 'std'],
        'profit_factor': ['mean', 'std'],
        'max_drawdown': ['mean', 'max'],
        'score': ['mean', 'std'],
        'stock_code': 'count'  # Number of stocks this strategy was tested on
    }).round(4)
    
    # Flatten column names
    strategy_summary.columns = [f'{col[0]}_{col[1]}' for col in strategy_summary.columns]
    
    # Calculate consistency score
    strategy_summary['consistency_score'] = (
        strategy_summary['score_mean'] * 0.4 +
        (1 - strategy_summary['score_std']) * 0.2 +
        (1 - strategy_summary['total_return_std']) * 0.2 +
        (1 - strategy_summary['max_drawdown_mean']) * 0.2
    )
    
    # 2. Top Performing Strategies by Stock
    top_strategies = combined_df.sort_values('score', ascending=False).groupby('stock_code').head(25)
    
    # 3. Stock Performance Summary
    stock_summary = combined_df.groupby('stock_code').agg({
        'total_return': ['max', 'mean'],
        'sharpe_ratio': ['max', 'mean'],
        'win_rate': ['max', 'mean'],
        'score': ['max', 'mean']
    }).round(4)
    
    # Flatten column names
    stock_summary.columns = [f'{col[0]}_{col[1]}' for col in stock_summary.columns]
    
    return strategy_summary, top_strategies, stock_summary
def process_single_stock(args):
    """Process a single stock with all its strategy combinations"""
    stock_code, data, indicator_gen, strategy_gen, tester, min_indicators, max_indicators, use_standard_params = args
    
    logging.info(f"Starting analysis for stock: {stock_code}")
    process_args = []
    
    # Generate indicator combinations
    indicators = list(indicator_gen.indicators.keys())
    
    for n in range(min_indicators, max_indicators + 1):
        indicator_combos = list(combinations(indicators, n))
        
        for indicator_combo in indicator_combos:
            param_combos = generate_parameter_combinations(
                indicator_combo, 
                indicator_gen, 
                use_standard_params
            )
            for params_combo in param_combos:
                process_args.append((data, indicator_combo, params_combo, strategy_gen, tester))
    
    # Process strategies for this stock
    total_combinations = len(process_args)
    logging.info(f"Testing {total_combinations} strategy combinations for {stock_code}...")
    
    # Use a ProcessPoolExecutor for strategy testing within each stock
    with ProcessPoolExecutor(max_workers=mp.cpu_count() // 2) as executor:  # Use half CPUs for nested parallelism
        results = list(executor.map(process_strategy, process_args))
    
    # Analyze results for this stock
    analysis_df = analyze_strategies(results, stock_code)
    
    if not analysis_df.empty:
        # Save individual stock results
        output_file = f"strategy_analysis_{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_directory = "/Users/shehzad/Desktop/alpha/output"
        analysis_df.to_csv(os.path.join(output_directory, output_file))
        logging.info(f"Analysis for {stock_code} saved to: {output_file}")
        
    return stock_code, analysis_df


def main():
    # Configuration
    DIRECTORY_PATH = '/Users/shehzad/Downloads/Dhaka Stock Exchange Historical Data/Adjusted Data'
    min_indicators = 1
    max_indicators = 9

    # Initialize components
    indicator_gen = TechnicalIndicatorGenerator()
    strategy_gen = StrategyGenerator(indicator_gen)
    tester = StrategyTester()

    # Load historical data
    logging.info("Loading historical data...")
    historical_data = load_historical_data(DIRECTORY_PATH)

    # Get user input for analysis mode
    print("\nAnalysis modes:")
    print("1. Single stock")
    print("2. All stocks")
    mode = input("Choose analysis mode (1/2): ").strip()

    # Get parameter mode
    print("\nParameter modes:")
    print("1. Standard parameters (faster)")
    print("2. Multiple parameter combinations (comprehensive)")
    param_mode = input("Choose parameter mode (1/2): ").strip()
    use_standard_params = param_mode == "1"

    stocks_to_analyze = []
    if mode == "1":
        available_stocks = list(historical_data.keys())
        print("\nAvailable stock codes:", ", ".join(available_stocks))
        stock_code = input("\nEnter the stock code: ").strip().upper()
        if stock_code not in historical_data:
            print(f"Error: Stock code {stock_code} not found.")
            return
        stocks_to_analyze = [stock_code]
    else:
        stocks_to_analyze = list(historical_data.keys())

    # Prepare arguments for parallel processing
    stock_args = [
        (
            stock_code,
            historical_data[stock_code],
            indicator_gen,
            strategy_gen,
            tester,
            min_indicators,
            max_indicators,
            use_standard_params
        )
        for stock_code in stocks_to_analyze
    ]

    # Process stocks in parallel
    all_results = {}
    num_workers = max(1, mp.cpu_count() // 2)  # Use half of available CPUs for top-level parallelism
    logging.info(f"Processing {len(stocks_to_analyze)} stocks using {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for stock_code, analysis_df in executor.map(process_single_stock, stock_args):
            if not analysis_df.empty:
                all_results[stock_code] = analysis_df
    
    # Generate master report
    if len(all_results) > 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create master reports
        strategy_summary, top_strategies, stock_summary = create_master_report(all_results)
        
        # Define output directory and create a new subdirectory with timestamp
        output_directory = "/Users/shehzad/Desktop/alpha/output"
        new_directory = os.path.join(output_directory, f"output_{timestamp}")
        os.makedirs(new_directory, exist_ok=True)

        # Save master reports
        strategy_summary.sort_values('consistency_score', ascending=False).to_csv(
            os.path.join(new_directory, f'master_strategy_summary_{timestamp}.csv')
        )
        top_strategies.to_csv(os.path.join(new_directory, f'top_strategies_by_stock_{timestamp}.csv'))
        stock_summary.to_csv(os.path.join(new_directory, f'stock_performance_summary_{timestamp}.csv'))

        # Print summary
        print("\n=== Master Report Summary ===")
        
        print("\nTop 10 Most Consistent Strategies:")
        print(strategy_summary.sort_values('consistency_score', ascending=False).head(10))
        
        print("\nBest Performing Stocks:")
        print(stock_summary.sort_values('total_return_max', ascending=False).head(10))
        
        print("\nReport files generated:")
        print(f"1. Master Strategy Summary: master_strategy_summary_{timestamp}.csv")
        print(f"2. Top Strategies by Stock: top_strategies_by_stock_{timestamp}.csv")
        print(f"3. Stock Performance Summary: stock_performance_summary_{timestamp}.csv")

if __name__ == "__main__":
    main()