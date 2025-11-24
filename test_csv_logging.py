#!/usr/bin/env python3
"""
Test script to demonstrate the comprehensive CSV logging system
"""

import yfinance as yf
import pandas as pd
from detailed_logger import create_comprehensive_strategy_log
from indicators import get_all_indicators
from optimization import universal_strategy

def test_csv_logging():
    """Test the CSV logging system with a simple example"""
    print("🧪 TESTING CSV LOGGING SYSTEM")
    print("="*50)
    
    # Load some test data
    ticker = "SPY"
    spy = yf.Ticker(ticker)
    data = spy.history(period="3mo", interval="1d")
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    
    print(f"📊 Loaded {len(data)} days of {ticker} data")
    
    # Add indicators
    enriched_data = get_all_indicators(data)
    print(f"📈 Added indicators, now have {len(enriched_data.columns)} columns")
    
    # Create a simple test strategy
    test_params = {
        'buy_score_threshold': 2,
        'sell_score_threshold': 2,
        'signal_persistence_days': 1,
        'use_RSI_14': True,
        'use_MACD_12_26_9': True,
        'use_WILLR_14': True,
        'RSI_14_buy': 30,
        'RSI_14_sell': 70,
        'MACD_12_26_9_buy': -1.0,
        'MACD_12_26_9_sell': 1.0,
        'WILLR_14_buy': -80,
        'WILLR_14_sell': -20,
        'use_trend_filter': False,
        'enable_position_sizing': True,
        'max_position_pct': 0.5
    }
    
    print(f"🎯 Test strategy parameters:")
    print(f"   Buy threshold: {test_params['buy_score_threshold']}")
    print(f"   RSI buy: < {test_params['RSI_14_buy']}")
    print(f"   MACD buy: > {test_params['MACD_12_26_9_buy']}")
    print(f"   WillR buy: < {test_params['WILLR_14_buy']}")
    
    # Generate signals
    signals = universal_strategy(enriched_data, test_params)
    print(f"📊 Generated {(signals == 1).sum()} buy signals, {(signals == -1).sum()} sell signals")
    
    # Create buy/sell score series (simplified)
    buy_scores = pd.Series(0, index=enriched_data.index)
    sell_scores = pd.Series(0, index=enriched_data.index)
    
    # Create comprehensive log
    try:
        log_filepath = create_comprehensive_strategy_log(
            ticker=ticker,
            data=enriched_data,
            strategy_params=test_params,
            signals=signals,
            buy_scores_series=buy_scores,
            sell_scores_series=sell_scores,
            trades_df=None,  # No actual trades for this test
            trial_number=999,
            return_pct=15.5
        )
        
        print(f"✅ Successfully created test log!")
        print(f"📁 Log file: {log_filepath}")
        
        # Show some sample data from the log
        if log_filepath:
            df = pd.read_csv(log_filepath)
            print(f"\n📋 CSV STRUCTURE:")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            
            print(f"\n🔍 SAMPLE COLUMNS:")
            interesting_cols = [col for col in df.columns if col in [
                'date', 'close', 'signal', 'signal_text', 'buy_score', 'sell_score',
                'ind_RSI_14', 'ind_WILLR_14', 'ind_MACD_12_26_9', 'position_shares',
                'total_portfolio_value', 'trade_action'
            ]]
            for col in interesting_cols[:10]:  # Show first 10 interesting columns
                print(f"   ✓ {col}")
            
            print(f"\n📊 SIGNAL SUMMARY FROM CSV:")
            print(f"   Buy signals: {(df['signal'] == 1).sum()}")
            print(f"   Sell signals: {(df['signal'] == -1).sum()}")
            print(f"   Hold days: {(df['signal'] == 0).sum()}")
            
            # Show a few days with signals
            signal_days = df[df['signal'] != 0].head(3)
            if len(signal_days) > 0:
                print(f"\n🎯 SAMPLE SIGNAL DAYS:")
                for _, row in signal_days.iterrows():
                    print(f"   {row['date']}: {row['signal_text']} at ${row['close']:.2f}")
                    print(f"      RSI: {row.get('ind_RSI_14', 'N/A')}, Buy Score: {row['buy_score']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_csv_logging()
