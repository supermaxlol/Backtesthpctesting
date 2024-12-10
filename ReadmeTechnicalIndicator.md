## we're using the TA-Lib library for technical analysis and break down each technical indicator in our code
## Each technical indicator in our code uses TA-Lib's optimized C/C++ implementations.
## Technical Analysis Implementation Using TA-Lib

## 1. Trend Following Indicators

### 1.1 Simple Moving Average (SMA)
```python
'SMA': IndicatorConfig(
    name='SMA',
    function=talib.SMA,
    params={'timeperiod': list(range(5, 201, 5))},
    standard_params={'timeperiod': 20},
    comparison_type='crossover'
)
```
**Implementation Details:**
- Function: `talib.SMA(close_prices, timeperiod)`
- Parameters tested: 5 to 200 days in steps of 5
- Signal Generation:
```python
def generate_sma_signals(data, indicator, params):
    sma = indicator.function(data['Close'], **params)
    return np.where(data['Close'] > sma, 1, 
                   np.where(data['Close'] < sma, -1, 0))
```
- Trading Logic:
  - Buy when price crosses above SMA
  - Sell when price crosses below SMA

### 1.2 Exponential Moving Average (EMA)
```python
'EMA': IndicatorConfig(
    name='EMA',
    function=talib.EMA,
    params={'timeperiod': list(range(5, 201, 5))},
    standard_params={'timeperiod': 20},
    comparison_type='crossover'
)
```
**Implementation Details:**
- Function: `talib.EMA(close_prices, timeperiod)`
- More weight to recent prices using multiplier: 2/(period + 1)
- Signal Generation similar to SMA but more responsive to recent price changes

### 1.3 MACD (Moving Average Convergence Divergence)
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
**Implementation Details:**
- Function: `talib.MACD(close_prices, fastperiod, slowperiod, signalperiod)`
- Returns: (macd, signal, histogram)
- Signal Generation:
```python
def generate_macd_signals(data, indicator, params):
    macd, signal, hist = indicator.function(data['Close'], **params)
    return np.where(macd > signal, 1, 
                   np.where(macd < signal, -1, 0))
```
- Trading Logic:
  - Buy when MACD crosses above signal line
  - Sell when MACD crosses below signal line

## 2. Momentum Indicators

### 2.1 Relative Strength Index (RSI)
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
**Implementation Details:**
- Function: `talib.RSI(close_prices, timeperiod)`
- Measures average gains vs average losses
- Signal Generation:
```python
def generate_rsi_signals(data, indicator, params):
    rsi = indicator.function(data['Close'], **params)
    return np.where(rsi < 30, 1,  # Oversold
                   np.where(rsi > 70, -1, 0))  # Overbought
```

### 2.2 Stochastic Oscillator
```python
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
)
```
**Implementation Details:**
- Function: `talib.STOCH(high, low, close, fastk_period, slowk_period, slowd_period)`
- Compares closing price to price range
- Signal Generation:
```python
def generate_stoch_signals(data, indicator, params):
    slowk, slowd = indicator.function(data['High'], data['Low'], 
                                    data['Close'], **params)
    return np.where(slowk < 20, 1,  # Oversold
                   np.where(slowk > 80, -1, 0))  # Overbought
```

## 3. Volume Indicators

### 3.1 On Balance Volume (OBV)
```python
'OBV': IndicatorConfig(
    name='OBV',
    function=talib.OBV,
    params={},
    standard_params={},
    comparison_type='crossover'
)
```
**Implementation Details:**
- Function: `talib.OBV(close, volume)`
- Cumulative volume measure
- Signal Generation:
```python
def generate_obv_signals(data, indicator, params):
    obv = indicator.function(data['Close'], data['Volume'])
    obv_ma = talib.SMA(obv, timeperiod=20)
    return np.where(obv > obv_ma, 1, 
                   np.where(obv < obv_ma, -1, 0))
```

### 3.2 Money Flow Index (MFI)
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
**Implementation Details:**
- Function: `talib.MFI(high, low, close, volume, timeperiod)`
- Volume-weighted RSI
- Signal Generation:
```python
def generate_mfi_signals(data, indicator, params):
    mfi = indicator.function(data['High'], data['Low'], 
                           data['Close'], data['Volume'], **params)
    return np.where(mfi < 20, 1,  # Oversold
                   np.where(mfi > 80, -1, 0))  # Overbought
```

## 4. Volatility Indicators

### 4.1 Bollinger Bands
```python
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
)
```
**Implementation Details:**
- Function: `talib.BBANDS(close, timeperiod, nbdevup, nbdevdn)`
- Returns: (upperband, middleband, lowerband)
- Signal Generation:
```python
def generate_bbands_signals(data, indicator, params):
    upper, middle, lower = indicator.function(data['Close'], **params)
    return np.where(data['Close'] < lower, 1,  # Price below lower band
                   np.where(data['Close'] > upper, -1, 0))  # Price above upper band
```

## 5. Additional Indicators

### 5.1 Average Directional Index (ADX)
```python
'ADX': IndicatorConfig(
    name='ADX',
    function=talib.ADX,
    params={'timeperiod': [7, 14, 21, 28]},
    standard_params={'timeperiod': 14},
    signal_thresholds=[20, 25, 30],
    comparison_type='threshold'
)
```
**Implementation Details:**
- Function: `talib.ADX(high, low, close, timeperiod)`
- Measures trend strength
- Values > 25 indicate strong trend

### 5.2 Rate of Change (ROC)
```python
'ROC': IndicatorConfig(
    name='ROC',
    function=talib.ROC,
    params={'timeperiod': [5, 10, 15, 20]},
    standard_params={'timeperiod': 10},
    signal_thresholds=[-5, 5],
    comparison_type='range'
)
```
**Implementation Details:**
- Function: `talib.ROC(close, timeperiod)`
- Measures price change rate
- Signal Generation based on momentum thresholds
