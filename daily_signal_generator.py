#!/usr/bin/env python3
"""
Daily Signal Generator for Manual Trading
Runs daily to generate buy/sell signals for a portfolio of tickers
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
import yfinance as yf
from indicators import get_all_indicators
from optimization import universal_strategy
from backtester import Backtester

class DailySignalGenerator:
    def __init__(self, portfolio_file='portfolio_config.json', output_dir='daily_signals'):
        """
        Initialize the daily signal generator
        
        Args:
            portfolio_file: JSON file with portfolio configuration
            output_dir: Directory to save signal reports
        """
        self.portfolio_file = portfolio_file
        self.output_dir = output_dir
        self.signals_today = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load portfolio configuration
        self.load_portfolio_config()
        
    def load_portfolio_config(self):
        """Load portfolio tickers and strategies from config file"""
        if not os.path.exists(self.portfolio_file):
            print(f"⚠️  Portfolio config not found. Creating default: {self.portfolio_file}")
            self.create_default_config()
        
        with open(self.portfolio_file, 'r') as f:
            self.config = json.load(f)
        
        print(f"📊 Loaded portfolio with {len(self.config['tickers'])} tickers")
        
    def create_default_config(self):
        """Create a default portfolio configuration"""
        default_config = {
            "tickers": ["SPY", "MSTY", "MSTR"],
            "default_strategy": None,  # Will use best saved strategy if None
            "lookback_days": 252,  # 1 year of data for indicators
            "signal_threshold": 0.7,  # Confidence threshold for signals
            "max_positions": 5,  # Maximum number of concurrent positions
            "notification_email": None  # Optional: email for signal alerts
        }
        
        with open(self.portfolio_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        self.config = default_config
        
    def get_market_data(self, ticker, period='1y', interval='1d'):
        """Download fresh market data for a ticker"""
        try:
            print(f"📥 Downloading {ticker} data...")
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data.empty:
                print(f"❌ No data for {ticker}")
                return None
            
            # Standardize columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data = data.reset_index()
            data.columns = [str(col).lower() for col in data.columns]
            data['date'] = pd.to_datetime(data['date'])
            
            print(f"✅ Downloaded {len(data)} days of {ticker} data")
            return data
            
        except Exception as e:
            print(f"❌ Error downloading {ticker}: {e}")
            return None
    
    def load_best_strategy(self, ticker=None):
        """Load the best performing saved strategy"""
        strategies_dir = 'saved_strategies'
        
        if not os.path.exists(strategies_dir):
            print("⚠️  No saved strategies found. Run optimization first!")
            return None
        
        # Load all strategies
        strategies = []
        for filename in os.listdir(strategies_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(strategies_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        strategy = json.load(f)
                        strategies.append(strategy)
                except Exception as e:
                    print(f"⚠️  Could not load {filename}: {e}")
        
        if not strategies:
            print("⚠️  No valid strategies found")
            return None
        
        # Find best strategy by return
        best_strategy = max(strategies, key=lambda x: x['performance']['total_return_pct'])
        print(f"✅ Loaded strategy: {best_strategy['name']} ({best_strategy['performance']['total_return_pct']:.2f}% return)")
        
        return best_strategy
    
    def generate_signals_for_ticker(self, ticker):
        """Generate buy/sell signals for a specific ticker"""
        print(f"\n{'='*60}")
        print(f"🎯 Analyzing {ticker}")
        print(f"{'='*60}")
        
        # Get market data
        data = self.get_market_data(ticker, period='1y', interval='1d')
        if data is None:
            return None
        
        # Calculate indicators
        print(f"📊 Calculating indicators...")
        try:
            enriched_data = get_all_indicators(data)
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return None
        
        # Load strategy
        strategy = self.load_best_strategy(ticker)
        if strategy is None:
            print(f"⚠️  No strategy available for {ticker}")
            return None
        
        # Generate signals using the strategy
        try:
            signals = universal_strategy(enriched_data, strategy['parameters'])
            
            # Check the most recent signal (TODAY)
            if len(signals) == 0:
                print(f"⚠️  No signals generated for {ticker}")
                return None
            
            last_signal = signals.iloc[-1]
            last_date = enriched_data.index[-1]
            last_close = enriched_data['close'].iloc[-1]
            
            # Determine signal type
            signal_type = None
            if last_signal == 1:
                signal_type = "BUY"
            elif last_signal == -1:
                signal_type = "SELL"
            else:
                signal_type = "HOLD"
            
            # Get recent price action
            price_change_5d = ((enriched_data['close'].iloc[-1] / enriched_data['close'].iloc[-5]) - 1) * 100 if len(enriched_data) >= 5 else 0
            price_change_1d = ((enriched_data['close'].iloc[-1] / enriched_data['close'].iloc[-2]) - 1) * 100 if len(enriched_data) >= 2 else 0
            
            signal_data = {
                'ticker': ticker,
                'date': last_date.strftime('%Y-%m-%d'),
                'signal': signal_type,
                'price': round(last_close, 2),
                'price_change_1d': round(price_change_1d, 2),
                'price_change_5d': round(price_change_5d, 2),
                'strategy_name': strategy['name'],
                'strategy_return': round(strategy['performance']['total_return_pct'], 2),
                'active_indicators': len(strategy['active_indicators']),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Print signal
            signal_emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"
            print(f"\n{signal_emoji} SIGNAL: {signal_type}")
            print(f"   Price: ${last_close:.2f} ({price_change_1d:+.2f}% today)")
            print(f"   Strategy: {strategy['name']}")
            print(f"   Historical Return: {strategy['performance']['total_return_pct']:.2f}%")
            
            return signal_data
            
        except Exception as e:
            print(f"❌ Error generating signals: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_daily_scan(self):
        """Run the daily signal scan for all portfolio tickers"""
        print("\n" + "="*60)
        print(f"🚀 DAILY SIGNAL GENERATOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        self.signals_today = []
        
        # Process each ticker
        for ticker in self.config['tickers']:
            signal = self.generate_signals_for_ticker(ticker)
            if signal:
                self.signals_today.append(signal)
        
        # Generate report
        self.generate_report()
        
        # Print summary
        self.print_summary()
        
        return self.signals_today
    
    def generate_report(self):
        """Generate and save daily signal report"""
        if not self.signals_today:
            print("\n⚠️  No signals generated today")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.signals_today)
        
        # Save to CSV
        date_str = datetime.now().strftime('%Y-%m-%d')
        csv_filename = os.path.join(self.output_dir, f'signals_{date_str}.csv')
        df.to_csv(csv_filename, index=False)
        print(f"\n💾 Saved signals to: {csv_filename}")
        
        # Save to JSON for programmatic access
        json_filename = os.path.join(self.output_dir, f'signals_{date_str}.json')
        with open(json_filename, 'w') as f:
            json.dump(self.signals_today, f, indent=2)
        print(f"💾 Saved signals to: {json_filename}")
        
    def print_summary(self):
        """Print a summary of today's signals"""
        if not self.signals_today:
            return
        
        print("\n" + "="*60)
        print("📋 TODAY'S TRADING SIGNALS SUMMARY")
        print("="*60)
        
        buy_signals = [s for s in self.signals_today if s['signal'] == 'BUY']
        sell_signals = [s for s in self.signals_today if s['signal'] == 'SELL']
        hold_signals = [s for s in self.signals_today if s['signal'] == 'HOLD']
        
        print(f"\n🟢 BUY Signals: {len(buy_signals)}")
        for signal in buy_signals:
            print(f"   • {signal['ticker']}: ${signal['price']:.2f} ({signal['price_change_1d']:+.2f}% today)")
            print(f"     Strategy: {signal['strategy_name']} ({signal['strategy_return']:.2f}% hist. return)")
        
        print(f"\n🔴 SELL Signals: {len(sell_signals)}")
        for signal in sell_signals:
            print(f"   • {signal['ticker']}: ${signal['price']:.2f} ({signal['price_change_1d']:+.2f}% today)")
            print(f"     Strategy: {signal['strategy_name']} ({signal['strategy_return']:.2f}% hist. return)")
        
        print(f"\n⚪ HOLD Signals: {len(hold_signals)}")
        for signal in hold_signals:
            print(f"   • {signal['ticker']}: ${signal['price']:.2f} ({signal['price_change_1d']:+.2f}% today)")
        
        print("\n" + "="*60)
        print("💡 Manual Action Required:")
        if buy_signals:
            print("   • Consider BUYING:", ", ".join([s['ticker'] for s in buy_signals]))
        if sell_signals:
            print("   • Consider SELLING:", ", ".join([s['ticker'] for s in sell_signals]))
        if not buy_signals and not sell_signals:
            print("   • No action needed today (all HOLD)")
        print("="*60 + "\n")


def main():
    """Main entry point for daily signal generation"""
    generator = DailySignalGenerator()
    signals = generator.run_daily_scan()
    return signals


if __name__ == "__main__":
    main()
