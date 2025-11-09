# 📈 Pattern_FindR - Professional Trading Strategy Discovery

**Pattern_FindR** is an advanced algorithmic trading strategy discovery platform that combines deep learning pattern recognition, hyperparameter optimization, and comprehensive backtesting to identify profitable trading strategies automatically.

## 🚀 Features

### 🤖 **AI-Powered Pattern Recognition**
- **CNN Deep Learning Models** - Automatically detect chart patterns (head & shoulders, triangles, flags, etc.)
- **Real-time Pattern Detection** - Generate buy/sell signals from live price data
- **Synthetic Training Data** - Generate unlimited training examples for robust model training

### 📊 **Advanced Strategy Optimization**
- **Optuna Hyperparameter Optimization** - Find optimal strategy parameters automatically
- **Multi-Core Parallelization** - Utilize all CPU cores for faster optimization
- **50+ Technical Indicators** - RSI, MACD, Bollinger Bands, Stochastic, and more
- **Intelligent Thresholds** - Dynamic buy/sell thresholds for each indicator

### 💰 **Professional Backtesting**
- **Configurable Starting Capital** - Test strategies with any capital amount ($1K - $10M)
- **Detailed Trade Analytics** - Position sizes, P&L, win rates, drawdowns
- **Portfolio Evolution Charts** - Visualize performance over time
- **Benchmark Comparisons** - Compare against buy-and-hold baseline

### 💾 **Strategy Management**
- **Persistent Storage** - Save top-performing strategies for future reference
- **Advanced Filtering** - Search, sort, and filter saved strategies
- **Performance Metrics** - Track returns, win rates, profit factors, and more
- **Strategy Comparison** - Compare multiple strategies side-by-side

## 🛠️ Installation

### Prerequisites
- **Python 3.8+**
- **4GB+ RAM** (8GB+ recommended for large optimizations)
- **Multi-core CPU** (for parallel optimization)

### 1. Clone the Repository
```bash
git clone https://github.com/jnaggud/pattern_findr.git
cd pattern_findr
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize Deep Learning Models (Optional)
```bash
# Set up and train CNN models for pattern recognition
python setup_dl_models.py
```

## 🎯 Quick Start Guide

### 1. Launch the Application
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

### 2. Configure Settings
- **Stock Symbol**: Enter any valid ticker (e.g., AAPL, MSFT, TSLA)
- **Starting Capital**: Set your backtesting capital ($100K default)
- **Optimization Method**: Choose parallelization method (joblib recommended)
- **Number of Trials**: Set optimization iterations (5000+ for best results)

### 3. Run Strategy Discovery
1. Click **"Find and Optimize Top Strategies"**
2. Wait for optimization to complete (progress shown in sidebar)
3. Review top-performing strategies with detailed metrics
4. Analyze trade logs and portfolio evolution charts
5. Save promising strategies for future reference

### 4. Strategy Analysis
Each optimized strategy includes:
- **Performance Metrics**: Return %, win rate, profit factor, max drawdown
- **Interactive Charts**: Price action with entry/exit points + portfolio evolution
- **Trade Log**: Detailed trade history with position sizes and P&L
- **Parameter Details**: Exact indicator settings for reproduction

## 📱 User Interface Guide

### Sidebar Controls
- **📈 Data Settings**: Symbol selection and data loading
- **🎯 Optimization**: Trial count and method selection
- **💰 Portfolio Settings**: Starting capital configuration
- **⚡ Performance**: Multi-core utilization controls
- **🏁 Benchmarking**: Performance testing options

### Main Dashboard
- **Strategy Results**: Top strategies ranked by performance
- **Interactive Charts**: Dual-panel price and portfolio charts
- **Trade Analytics**: Comprehensive trade logs with running totals
- **Strategy Storage**: Save and manage winning strategies

### Saved Strategies Viewer
- **Search & Filter**: Find strategies by name, return %, indicators
- **Advanced Sorting**: Sort by return, win rate, profit factor, etc.
- **Strategy Management**: View details, compare performance, delete strategies

## 🔧 Configuration Options

### Optimization Settings
```python
# Number of optimization trials
trials = 5000  # Recommended: 5000-10000

# Parallelization methods
methods = ['standard', 'joblib', 'advanced']

# Starting capital range
capital = 1000 to 10,000,000  # $1K to $10M
```

### Technical Indicators Available
- **Momentum**: RSI, Stochastic, Williams %R, ROC
- **Trend**: MACD, EMA, SMA, ADX, Parabolic SAR
- **Volatility**: Bollinger Bands, ATR, Standard Deviation
- **Volume**: OBV, Volume SMA, VWAP
- **Pattern Recognition**: CNN-based chart patterns

## 💡 Best Practices

### Optimization Strategy
1. **Start Small**: Begin with 1000 trials to test setup
2. **Scale Up**: Use 5000+ trials for production strategies
3. **Parallel Processing**: Enable multi-core for faster results
4. **Multiple Runs**: Run optimization multiple times for robustness

### Strategy Validation
1. **Out-of-Sample Testing**: Test on different time periods
2. **Multiple Symbols**: Validate across different stocks/sectors
3. **Risk Management**: Focus on drawdown and risk-adjusted returns
4. **Walk-Forward Analysis**: Use rolling optimization windows

### Performance Optimization
1. **Use SSD Storage**: Faster I/O for data processing
2. **Adequate RAM**: 8GB+ for large optimizations
3. **Multi-Core CPU**: Enables significant speedup
4. **Close Other Apps**: Maximize available resources

## 📊 Understanding Results

### Key Metrics
- **Total Return %**: Overall strategy performance
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Max Drawdown**: Largest portfolio decline from peak
- **Sharpe Ratio**: Risk-adjusted return measurement

### Trade Analysis
- **Position Size**: Dollar amount invested per trade
- **Shares/Contracts**: Number of units traded
- **Trade P&L**: Individual trade profit/loss
- **Running Total**: Cumulative profit across all trades

## 🔍 Troubleshooting

### Common Issues
1. **Slow Optimization**: Reduce trials or enable parallelization
2. **Memory Errors**: Reduce batch size or close other applications
3. **No Profitable Strategies**: Try different symbols or increase trials
4. **Import Errors**: Ensure all dependencies are installed

### Performance Tips
- Use joblib optimization method for best speed
- Enable benchmark testing to find fastest method
- Monitor CPU utilization in system monitor
- Consider cloud computing for very large optimizations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style requirements
- Testing procedures
- Feature request process
- Bug reporting guidelines

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**Important**: This software is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions. The authors are not responsible for any financial losses incurred through the use of this software.

## 📞 Support

- **Issues**: Report bugs via GitHub Issues
- **Feature Requests**: Submit via GitHub Discussions
- **Documentation**: Check the wiki for detailed guides
- **Community**: Join our Discord for real-time help

---

**Happy Trading! 🚀📈**
