# Strategy Finding Process and Technical Indicators 

## 1. Strategy Discovery Process

The code finds strategies through systematic combination of technical indicators:

```python
def process_single_stock(args):
    stock_code, data, indicator_gen, strategy_gen, tester, min_indicators, max_indicators = args
    
    # Generate indicator combinations (1 to 9 indicators)
    for n in range(min_indicators, max_indicators + 1):
        indicator_combos = list(combinations(indicators, n))
        
        for indicator_combo in indicator_combos:
            # For each combination, test different parameters
            param_combos = generate_parameter_combinations(
                indicator_combo, 
                indicator_gen
            )
            for params_combo in param_combos:
                test_strategy(combo, params)
```

### 1.1 Search Space
For each stock:
1. Tests combinations of 1-9 indicators
2. For each combination:
   - Tests multiple parameter sets
   - Evaluates performance metrics
   - Ranks strategies based on composite score

## 2. Technical Indicators Implementation

### 2.1 Trend Following Indicators

#### Simple Moving Average (SMA)
```python
'SMA': IndicatorConfig(
    name='SMA',
    function=talib.SMA,
    params={'timeperiod': list(range(5, 201, 5))},
    standard_params={'timeperiod': 20},
    comparison_type='crossover'
)
```
- **Implementation**: Average price over N periods
- **Parameters**: Timeperiods from 5 to 200 in steps of 5
- **Signal Generation**:
  ```python
  signals = np.where(close > sma, 1, np.where(close < sma, -1, 0))
  ```
- **Usage**: Trend identification and support/resistance levels

#### Exponential Moving Average (EMA)
```python
'EMA': IndicatorConfig(
    name='EMA',
    function=talib.EMA,
    params={'timeperiod': list(range(5, 201, 5))},
    standard_params={'timeperiod': 20},
    comparison_type='crossover'
)
```
- **Implementation**: Weighted moving average giving more weight to recent prices
- **Parameters**: Timeperiods 5-200
- **Signal Logic**: Same as SMA but more responsive to recent price changes

#### MACD (Moving Average Convergence Divergence)
```python
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
)
```
- **Implementation**: Difference between fast and slow EMAs
- **Signal Generation**:
  ```python
  macd, signal, hist = indicator.function(close, **params)
  signals = np.where(macd > signal, 1, np.where(macd < signal, -1, 0))
  ```
- **Usage**: Trend and momentum identification

### 2.2 Momentum Indicators

#### Relative Strength Index (RSI)
```python
'RSI': IndicatorConfig(
    name='RSI',
    function=talib.RSI,
    params={'timeperiod': [7, 14, 21, 28]},
    standard_params={'timeperiod': 14},
    signal_thresholds=[20, 30, 70, 80],
    comparison_type='range'
)
```
- **Implementation**: Measures price momentum and overbought/oversold conditions
- **Thresholds**: Buy below 30, sell above 70
- **Signal Generation**:
  ```python
  signals = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
  ```

#### Stochastic Oscillator
```python
'STOCH': IndicatorConfig(
    name='STOCH',
    function=talib.STOCH,
    params={
        'fastk_period': [5, 14, 21],
        'slowk_period': [3, 5, 7],
        'slowd_period': [3, 5, 7]
    },
    signal_thresholds=[20, 80],
    comparison_type='range'
)
```
- **Implementation**: Compares closing price to price range over time
- **Thresholds**: Buy below 20, sell above 80

### 2.3 Volume Indicators

#### On Balance Volume (OBV)
```python
'OBV': IndicatorConfig(
    name='OBV',
    function=talib.OBV,
    params={},
    standard_params={},
    comparison_type='crossover'
)
```
- **Implementation**: Cumulative volume measure based on price direction
- **Signal Generation**: 
  ```python
  obv = indicator.function(close, volume)
  obv_ma = talib.SMA(obv, timeperiod=20)
  signals = np.where(obv > obv_ma, 1, np.where(obv < obv_ma, -1, 0))
  ```

#### Money Flow Index (MFI)
```python
'MFI': IndicatorConfig(
    name='MFI',
    function=talib.MFI,
    params={'timeperiod': [7, 14, 21]},
    standard_params={'timeperiod': 14},
    signal_thresholds=[20, 80],
    comparison_type='range'
)
```
- **Implementation**: Volume-weighted RSI
- **Thresholds**: Buy below 20, sell above 80

### 2.4 Volatility Indicators

#### Bollinger Bands
```python
'BBANDS': IndicatorConfig(
    name='BBANDS',
    function=talib.BBANDS,
    params={
        'timeperiod': [10, 20, 30],
        'nbdevup': [1.5, 2.0, 2.5],
        'nbdevdn': [1.5, 2.0, 2.5]
    },
    comparison_type='range'
)
```
- **Implementation**: Moving average with standard deviation bands
- **Signal Generation**:
  ```python
  upper, middle, lower = indicator.function(close, **params)
  signals = np.where(close < lower, 1, np.where(close > upper, -1, 0))
  ```

## 3. Strategy Combination Process

```python
def generate_combined_strategy(self, data: pd.DataFrame, 
                             indicator_combinations: List[Tuple[str, dict]]) -> pd.Series:
    signals = pd.DataFrame(index=data.index)
    
    # Generate individual indicator signals
    for ind_name, params in indicator_combinations:
        indicator = self.indicator_generator.indicators[ind_name]
        signals[ind_name] = self.generate_indicator_signals(data, indicator, params)
    
    # Combine signals through averaging
    combined_signals = signals.mean(axis=1)
    
    # Apply thresholds
    return pd.Series(np.where(combined_signals > 0.3, 1,
                            np.where(combined_signals < -0.3, -1, 0)),
                    index=data.index)
```

### 3.1 Combination Method
1. Generate signals for each indicator
2. Average all signals
3. Apply thresholds:
   - Buy if average > 0.3
   - Sell if average < -0.3
   - Hold otherwise

## 4. Strategy Evaluation

```python
def calculate_metrics(self, trades: List[Dict], final_value: float, data: pd.DataFrame) -> Dict:
    metrics = {
        'total_return': (final_value - self.initial_capital) / self.initial_capital,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': len(trades)
    }
```

### 4.1 Ranking System
```python
df['score'] = (
    df['sharpe_ratio'].rank(pct=True) * 0.3 +
    df['total_return'].rank(pct=True) * 0.3 +
    df['win_rate'].rank(pct=True) * 0.2 +
    df['profit_factor'].rank(pct=True) * 0.2 -
    df['max_drawdown'].rank(pct=True) * 0.2
)
```

Strategy ranking weights:
- 30% Sharpe Ratio
- 30% Total Return
- 20% Win Rate
- 20% Profit Factor
- -20% Maximum Drawdown
