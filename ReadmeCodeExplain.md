# DSE Trading Strategy Code Explanation 

## 1. Core Data Loading and Processing

```python
def load_historical_data(directory_path):
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
```
This function:
- Loads stock data from CSV files
- Processes each file into a DataFrame
- Validates data format and columns
- Returns a dictionary of stock code to DataFrame mappings

## 2. Technical Indicators Configuration

```python
@dataclass
class IndicatorConfig:
    name: str
    function: callable
    params: dict
    standard_params: dict
    signal_thresholds: List[float] = None
    comparison_type: str = 'threshold'
```
Configuration class that defines:
- Indicator name and function
- Parameter ranges for optimization
- Standard parameter sets
- Signal thresholds and comparison types

## 3. Strategy Generation

### 3.1 Indicator Signal Generation
```python
def generate_indicator_signals(self, data: pd.DataFrame, indicator: IndicatorConfig, 
                             params: dict) -> pd.Series:
    if indicator.name == 'MACD':
        macd, signal, hist = indicator.function(close, **params)
        return pd.Series(np.where(macd > signal, 1, 
                                np.where(macd < signal, -1, 0)), 
                       index=close.index)
```
This generates trading signals for individual indicators:
- Takes price data and indicator parameters
- Applies indicator calculations
- Returns buy (1), sell (-1), or hold (0) signals

### 3.2 Strategy Combination
```python
def generate_combined_strategy(self, data: pd.DataFrame, 
                             indicator_combinations: List[Tuple[str, dict]]) -> pd.Series:
    signals = pd.DataFrame(index=data.index)
    
    for ind_name, params in indicator_combinations:
        signals[ind_name] = self.generate_indicator_signals(data, indicator, params)
    
    combined_signals = signals.mean(axis=1)
    return pd.Series(np.where(combined_signals > 0.3, 1,
                            np.where(combined_signals < -0.3, -1, 0)),
                    index=data.index)
```
Combines multiple indicator signals:
- Generates signals for each indicator
- Takes average of all signals
- Applies thresholds for final decisions

## 4. Backtesting Engine

```python
def backtest_strategy(self, data: pd.DataFrame, signals: pd.Series,
                     transaction_cost: float = 0.001) -> Dict:
    position = 0
    capital = self.initial_capital
    holdings = 0
    trades = []
    
    for i in range(1, len(signals)):
        current_price = data['Close'].iloc[i]
        current_signal = signals.iloc[i]
        
        if current_signal != position:
            if current_signal == 1:  # Buy signal
                shares = (capital * 0.95) // current_price
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
```
Implements backtesting logic:
- Tracks positions and capital
- Handles buy/sell executions
- Accounts for transaction costs
- Records all trades

## 5. Performance Metrics Calculation

```python
def calculate_metrics(self, trades: List[Dict], final_value: float, data: pd.DataFrame) -> Dict:
    if not trades:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'num_trades': 0
        }
        
    daily_returns = []
    portfolio_value = pd.Series(index=data.index, dtype=float)
    portfolio_value.iloc[0] = self.initial_capital
```
Calculates key performance metrics:
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor
- Number of trades

## 6. Parallel Processing Implementation

```python
def process_single_stock(args):
    stock_code, data, indicator_gen, strategy_gen, tester, min_indicators, max_indicators = args
    
    logging.info(f"Starting analysis for stock: {stock_code}")
    process_args = []
    
    for n in range(min_indicators, max_indicators + 1):
        indicator_combos = list(combinations(indicators, n))
        
        for indicator_combo in indicator_combos:
            param_combos = generate_parameter_combinations(
                indicator_combo, 
                indicator_gen
            )
            for params_combo in param_combos:
                process_args.append((data, indicator_combo, params_combo, strategy_gen, tester))
```
Handles parallel processing:
- Processes each stock independently
- Tests multiple indicator combinations
- Optimizes parameters
- Uses ProcessPoolExecutor for parallelization

## 7. Main Execution Flow

```python
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
    historical_data = load_historical_data(DIRECTORY_PATH)
```
Main execution:
1. Loads configuration
2. Initializes components
3. Processes data
4. Generates reports

## 8. Strategy Analysis and Reporting

```python
def analyze_strategies(results: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(analysis)
    
    df['score'] = (
        df['sharpe_ratio'].rank(pct=True) * 0.3 +
        df['total_return'].rank(pct=True) * 0.3 +
        df['win_rate'].rank(pct=True) * 0.2 +
        df['profit_factor'].rank(pct=True) * 0.2 -
        df['max_drawdown'].rank(pct=True) * 0.2
    )
```
Analyzes strategy performance:
- Ranks strategies based on multiple metrics
- Calculates composite scores
- Generates performance reports
