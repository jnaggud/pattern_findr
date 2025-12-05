import pandas as pd
import numpy as np
import os
import glob

def run_deep_analysis():
    # 1. Find latest file
    list_of_files = glob.glob('analysis/*.csv')
    if not list_of_files:
        print("No analysis files found.")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"🔍 ANALYZING FILE: {latest_file}")
    
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    
    # 2. Feature Engineering
    df['composite_slope'] = df['composite_indicator'].diff()
    df['slope_roc'] = df['composite_slope'].diff()
    
    # Slope Turns
    slope = df['composite_slope'].values
    slope_prev = np.roll(slope, 1)
    slope_prev[0] = 0
    
    turn_up = (slope_prev < 0) & (slope > 0)
    turn_down = (slope_prev > 0) & (slope < 0)
    
    # Calculate Returns
    horizons = [1, 3, 5, 10, 20]
    for d in horizons:
        df[f'ret_{d}d'] = df['close'].pct_change(d).shift(-d) * 100
        
    buys = df[turn_up].copy()
    sells = df[turn_down].copy()
    
    print("\n" + "="*60)
    print("🕵️‍♂️ DEEP DIVE ANALYST REPORT")
    print("="*60)
    
    # --- SECTION 1: THE GOLDEN ZONES (EXPECTANCY ANALYSIS) ---
    print("\n💎 SECTION 1: THE GOLDEN ZONES (Expectancy Analysis)")
    print("Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)")
    print("-" * 60)
    
    bins = np.arange(-1.0, 1.1, 0.1)
    buys['comp_bin'] = pd.cut(buys['composite_indicator'], bins=bins)
    
    stats = []
    for bin_name, group in buys.groupby('comp_bin'):
        if len(group) < 5: continue
        
        # 5d Metrics
        wins = group[group['ret_5d'] > 0]['ret_5d']
        losses = group[group['ret_5d'] < 0]['ret_5d']
        
        win_rate = len(wins) / len(group)
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        profit_factor = (wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else 999
        
        stats.append({
            'Bin': bin_name,
            'Count': len(group),
            'WinRate': win_rate * 100,
            'Expectancy': expectancy,
            'ProfitFactor': profit_factor,
            'AvgReturn': group['ret_5d'].mean()
        })
        
    stats_df = pd.DataFrame(stats).sort_values('Expectancy', ascending=False)
    print(stats_df.to_string(formatters={
        'WinRate': '{:.1f}%'.format,
        'Expectancy': '{:.2f}%'.format,
        'ProfitFactor': '{:.2f}'.format,
        'AvgReturn': '{:.2f}%'.format
    }))
    
    if not stats_df.empty:
        print("\n💡 INSIGHT: The highest expectancy is NOT always the highest win rate.")
        top_bin = stats_df.iloc[0]
        print(f"   -> BEST ZONE: {top_bin['Bin']} with Expectancy {top_bin['Expectancy']}")
    
    # --- SECTION 2: THE MOMENTUM ANOMALY (0.2 to 0.3) ---
    print("\n🚀 SECTION 2: THE MOMENTUM ANOMALY (0.2 to 0.3)")
    print("-" * 60)
    mom_buys = buys[(buys['composite_indicator'] >= 0.2) & (buys['composite_indicator'] <= 0.3)]
    
    if not mom_buys.empty:
        print(f"Found {len(mom_buys)} Momentum Trades.")
        print(f"Win Rate (5d): {(mom_buys['ret_5d']>0).mean()*100:.1f}%")
        print(f"Avg Return (5d): {mom_buys['ret_5d'].mean():.2f}%")
        print(f"Avg Return (20d): {mom_buys['ret_20d'].mean():.2f}% (Trend Continuation?)")
        
        # Check volatility context
        # Is it better when volatility is Low or High?
        # Calculate ATR proxy (High-Low range % of close)
        df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
        mom_buys['range_pct'] = df.loc[mom_buys.index, 'range_pct']
        
        low_vol_mom = mom_buys[mom_buys['range_pct'] < mom_buys['range_pct'].median()]
        high_vol_mom = mom_buys[mom_buys['range_pct'] >= mom_buys['range_pct'].median()]
        
        print(f"   -> Low Volatility Returns: {low_vol_mom['ret_5d'].mean():.2f}%")
        print(f"   -> High Volatility Returns: {high_vol_mom['ret_5d'].mean():.2f}%")
        print("   -> CONCLUSION: Do breakouts work better in calm or chaos?")
    
    # --- SECTION 3: THE NOV 13 DRAWDOWN AUTOPSY ---
    print("\n🩸 SECTION 3: AUTOPSY OF RECENT DRAWDOWN (Last 30 Days)")
    print("-" * 60)
    if not df.empty:
        recent_cutoff = df.index[-1] - pd.Timedelta(days=30)
        recent_buys = buys[buys.index >= recent_cutoff]
        
        if not recent_buys.empty:
            print(f"Recent Trades: {len(recent_buys)}")
            print(f"Recent Win Rate: {(recent_buys['ret_5d']>0).mean()*100:.1f}%")
            print(f"Recent Avg Return: {recent_buys['ret_5d'].mean():.2f}%")
            
            # What characterizes these losers?
            losers = recent_buys[recent_buys['ret_5d'] < 0]
            if not losers.empty:
                avg_losers_slope = losers['composite_slope'].mean()
                avg_losers_roc = losers['slope_roc'].mean()
                print(f"   -> Avg Slope at Entry (Losers): {avg_losers_slope:.4f}")
                print(f"   -> Avg Slope Acceleration (Losers): {avg_losers_roc:.4f}")
                
                # Compare to historical winners
                winners_hist = buys[buys['ret_5d'] > 0]
                if not winners_hist.empty:
                    print(f"   -> Historical Winners Slope: {winners_hist['composite_slope'].mean():.4f}")
                    print(f"   -> Historical Winners Accel: {winners_hist['slope_roc'].mean():.4f}")
                    
                    print("   -> HYPOTHESIS: If Recent Losers have weaker slope/acceleration, filter by magnitude!")

    # --- SECTION 4: "SMART FILTER" SIMULATION ---
    print("\n🧠 SECTION 4: 'SMART FILTER' SIMULATION")
    print("-" * 60)
    # Simulate the strategy: Buy if Comp < -0.6 OR (Comp > 0.2 AND Comp < 0.3)
    
    smart_buys = buys[
        (buys['composite_indicator'] < -0.6) | 
        ((buys['composite_indicator'] >= 0.2) & (buys['composite_indicator'] <= 0.3))
    ]
    
    if not smart_buys.empty:
        print(f"Smart Filter Count: {len(smart_buys)} (Original: {len(buys)})")
        print(f"Smart Filter Win Rate: {(smart_buys['ret_5d']>0).mean()*100:.1f}%")
        print(f"Smart Filter Avg Return: {smart_buys['ret_5d'].mean():.2f}%")
        
        wr = (smart_buys['ret_5d']>0).mean()
        avg_w = smart_buys[smart_buys['ret_5d']>0]['ret_5d'].mean() if wr > 0 else 0
        avg_l = abs(smart_buys[smart_buys['ret_5d']<0]['ret_5d'].mean()) if wr < 1 else 0
        
        exp = (wr * avg_w) - ((1-wr) * avg_l)
        
        print(f"Smart Filter Expectancy: {exp:.2f}%")
    
    print("\n" + "="*60)
    print("END OF REPORT")
    print("="*60)

if __name__ == "__main__":
    run_deep_analysis()
