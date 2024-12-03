# DSE Trading Strategy Optimization

## Overview
A high-performance computing framework for optimizing and backtesting trading strategies on the Dhaka Stock Exchange (DSE). This project leverages 10 years of historical data across 400 DSE-listed stocks to identify robust, market-specific trading strategies using technical analysis.

## Key Features
- Multi-indicator strategy optimization using 12 technical indicators
- Parallel processing for efficient strategy backtesting
- Comprehensive performance metrics and analysis
- Support for both single-stock and market-wide analysis
- Flexible parameter optimization with standard and comprehensive modes
- Robust data validation and processing

## Technical Indicators
The framework implements the following technical indicators:
- Trend: SMA, EMA, MACD
- Momentum: RSI, Stochastic, CCI
- Volume: OBV, MFI
- Volatility: Bollinger Bands
- Additional: Williams %R, ROC, ADX
  
## Strategy Combination Methodology
1. Signal Generation Architecture
The code implements strategy combination through a multi-layered approach:

Individual Indicator Signals

Each technical indicator (RSI, MACD, etc.) generates its own signals (-1, 0, 1)
Signals are normalized across different indicators using consistent thresholds
Implementation in generate_indicator_signals method:

pythonCopydef generate_indicator_signals(self, data: pd.DataFrame, indicator: IndicatorConfig, params: dict) -> pd.Series

Signal Combination Method

Multiple indicators are combined using a weighted average approach
Current implementation in generate_combined_strategy:

pythonCopycombined_signals = signals.mean(axis=1)
return pd.Series(np.where(combined_signals > 0.3, 1,
                         np.where(combined_signals < -0.3, -1, 0)),
                 index=data.index)


## 2. Strategy Optimization Process
The code employs several layers of optimization:

Parameter Optimization

Grid search across parameter spaces for each indicator
Options for both standard and comprehensive parameter testing


Indicator Combination Testing

Tests combinations of 1 to 9 indicators
Uses itertools.combinations for systematic combination generation


Performance Evaluation

Comprehensive metrics including:

Risk-adjusted returns (Sharpe Ratio)
Maximum drawdown
Win rate
Profit factor


Composite scoring system for strategy ranking

## Requirements
- Python 3.x
- Required packages:
  - pandas
  - numpy
  - talib
  - concurrent.futures
  - logging

## Installation
1. Clone the repository
2. Install required packages:
```bash
pip install pandas numpy ta-lib
```
3. Ensure you have the DSE historical data in CSV format

## Usage
Run the main script:
```bash
python stockdatabacktesthpc.py
```

The script provides two analysis modes:
1. Single stock analysis
2. Market-wide analysis

And two parameter optimization modes:
1. Standard parameters (faster)
2. Multiple parameter combinations (comprehensive)

## Output
The framework generates three main reports:
1. Master Strategy Summary - Overall performance and consistency of strategies
2. Top Strategies by Stock - Best performing strategies for each stock
3. Stock Performance Summary - Performance metrics across different stocks

Reports are saved in CSV format with timestamps in the specified output directory.

## Project Structure
- `stockdatabacktesthpc.py` - Main implementation file
- Data processing modules:
  - Technical indicator generation
  - Strategy generation and testing
  - Performance analysis and reporting

## Performance Metrics
- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Trade Count
- Consistency Score

## Notes
- The framework uses parallel processing to optimize performance
- Results are saved with timestamps for easy tracking
- Comprehensive error handling and logging are implemented
- Strategy parameters can be customized based on requirements
- The current implementation provides a solid foundation for strategy development and testing. Its main strengths lie in computational efficiency and flexibility.
- However, for production use, it would benefit from:
- More sophisticated signal combination methods
- Enhanced risk management
- Robust validation framework
- Market regime consideration

