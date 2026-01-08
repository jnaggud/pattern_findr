"""
Performance Dashboard for Trading Signals

Generates public performance reports and statistics for marketing.
Run with: streamlit run performance_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Strategy directories
VELOCITY_STRATEGIES_DIR = "velocity_strategies"

st.set_page_config(
    page_title="Trading Signal Performance",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .big-number {
        font-size: 48px;
        font-weight: bold;
        color: #00ff88;
    }
    .disclaimer {
        background: #2d2d2d;
        padding: 15px;
        border-radius: 5px;
        font-size: 12px;
        color: #888;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)


def load_trade_history(strategy_name: str) -> list:
    """Load trade history for a strategy."""
    safe_name = strategy_name.replace("/", "-").replace(":", "-").replace(" ", "_")
    history_path = f"velocity_trade_history_{safe_name}.json"

    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def load_all_strategies() -> list:
    """Load all saved velocity strategies."""
    strategies = []

    if not os.path.exists(VELOCITY_STRATEGIES_DIR):
        return strategies

    for item in os.listdir(VELOCITY_STRATEGIES_DIR):
        strategy_path = os.path.join(VELOCITY_STRATEGIES_DIR, item)
        config_file = os.path.join(strategy_path, "velocity_config.json")

        if os.path.isdir(strategy_path) and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # Load trade history
                strategy_name = config.get('strategy_name', item)
                history = load_trade_history(strategy_name)

                strategies.append({
                    'name': item,
                    'config': config,
                    'strategy_name': strategy_name,
                    'ticker': config.get('ticker', 'Unknown'),
                    'optimization_period': config.get('optimization_period', ''),
                    'history': history,
                    'trade_count': len(history),
                })
            except Exception as e:
                pass

    return strategies


def calculate_stats(history: list) -> dict:
    """Calculate performance statistics from trade history."""
    if not history:
        return {
            'total_trades': 0,
            'winners': 0,
            'losers': 0,
            'win_rate': 0,
            'total_return': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'best_trade': 0,
            'worst_trade': 0,
        }

    pnls = [t.get('pnl_pct', 0) for t in history]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    # Calculate equity curve for drawdown
    equity = [100]
    for pnl in pnls:
        equity.append(equity[-1] * (1 + pnl/100))

    # Max drawdown
    peak = equity[0]
    max_dd = 0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'total_trades': len(history),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(history) * 100 if history else 0,
        'total_return': (equity[-1] / equity[0] - 1) * 100,
        'avg_win': np.mean(winners) if winners else 0,
        'avg_loss': np.mean(losers) if losers else 0,
        'profit_factor': abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else 0,
        'max_drawdown': max_dd,
        'best_trade': max(pnls) if pnls else 0,
        'worst_trade': min(pnls) if pnls else 0,
        'equity_curve': equity,
    }


def main():
    st.title("📈 Trading Signal Performance Dashboard")
    st.markdown("**Live Performance Tracking for Velocity Trading Strategies**")

    # Load strategies
    strategies = load_all_strategies()

    if not strategies:
        st.warning("No strategies found. Deploy strategies first.")
        return

    # Filter to main 6 strategies
    main_strategies = [s for s in strategies if any(x in s['name'].lower() for x in ['spy_5y', 'spy_2y', 'spy_1y', 'btc_5y', 'btc_2y', 'btc_1y'])]

    if not main_strategies:
        main_strategies = strategies[:6]

    # Overall stats
    st.header("📊 Overall Performance")

    all_trades = []
    for s in main_strategies:
        all_trades.extend(s['history'])

    overall_stats = calculate_stats(all_trades)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Trades", overall_stats['total_trades'])
    with col2:
        st.metric("Win Rate", f"{overall_stats['win_rate']:.1f}%")
    with col3:
        st.metric("Total Return", f"{overall_stats['total_return']:.1f}%",
                  delta=f"{overall_stats['total_return']:.1f}%")
    with col4:
        st.metric("Profit Factor", f"{overall_stats['profit_factor']:.2f}")
    with col5:
        st.metric("Max Drawdown", f"-{overall_stats['max_drawdown']:.1f}%")

    st.divider()

    # Individual strategy performance
    st.header("🎯 Strategy Performance")

    # Create comparison table
    comparison_data = []
    for s in main_strategies:
        stats = calculate_stats(s['history'])
        comparison_data.append({
            'Strategy': s['strategy_name'],
            'Ticker': s['ticker'],
            'Period': s['optimization_period'].upper(),
            'Trades': stats['total_trades'],
            'Win Rate': f"{stats['win_rate']:.1f}%",
            'Return': f"{stats['total_return']:.1f}%",
            'Profit Factor': f"{stats['profit_factor']:.2f}",
            'Best Trade': f"{stats['best_trade']:.1f}%",
            'Worst Trade': f"{stats['worst_trade']:.1f}%",
        })

    if comparison_data:
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

    # Equity curves
    st.header("📈 Equity Curves")

    fig = go.Figure()

    colors = ['#00ff88', '#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3', '#f38181']

    for i, s in enumerate(main_strategies):
        stats = calculate_stats(s['history'])
        if stats.get('equity_curve') and len(stats['equity_curve']) > 1:
            fig.add_trace(go.Scatter(
                y=stats['equity_curve'],
                mode='lines',
                name=f"{s['ticker']} {s['optimization_period'].upper()}",
                line=dict(color=colors[i % len(colors)], width=2)
            ))

    fig.update_layout(
        title="Equity Curves by Strategy (Starting at $100)",
        xaxis_title="Trade #",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Recent trades
    st.header("📜 Recent Trades")

    recent_trades = sorted(all_trades, key=lambda x: x.get('exit_time', ''), reverse=True)[:20]

    if recent_trades:
        trades_display = []
        for t in recent_trades:
            trades_display.append({
                'Date': t.get('exit_time', '')[:10] if t.get('exit_time') else 'N/A',
                'Strategy': t.get('strategy_name', 'Unknown'),
                'Type': t.get('type', 'LONG'),
                'Entry': f"${t.get('entry_price', 0):.2f}",
                'Exit': f"${t.get('exit_price', 0):.2f}",
                'P&L': f"{t.get('pnl_pct', 0):+.2f}%",
                'Result': '✅' if t.get('pnl_pct', 0) > 0 else '❌',
            })

        df_trades = pd.DataFrame(trades_display)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)
    else:
        st.info("No trades recorded yet. Trades will appear here as positions are closed.")

    # Monthly returns calendar
    st.header("📅 Monthly Performance")

    if all_trades:
        # Group trades by month
        monthly_returns = {}
        for t in all_trades:
            exit_time = t.get('exit_time', '')
            if exit_time:
                month_key = exit_time[:7]  # YYYY-MM
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = []
                monthly_returns[month_key].append(t.get('pnl_pct', 0))

        # Calculate monthly totals
        monthly_data = []
        for month, pnls in sorted(monthly_returns.items()):
            total_return = sum(pnls)
            monthly_data.append({
                'Month': month,
                'Trades': len(pnls),
                'Return': total_return,
            })

        if monthly_data:
            df_monthly = pd.DataFrame(monthly_data)

            fig_monthly = px.bar(
                df_monthly,
                x='Month',
                y='Return',
                color='Return',
                color_continuous_scale=['#ff6b6b', '#ffe66d', '#00ff88'],
                title="Monthly Returns (%)"
            )
            fig_monthly.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_monthly, use_container_width=True)

    # Call to action
    st.divider()
    st.header("🚀 Get Access to Real-Time Signals")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### Free Tier
        **$0/month**
        - 1 Strategy (24hr delayed)
        - Daily summary updates
        - Community access

        [Join Free →](#)
        """)

    with col2:
        st.markdown("""
        ### Pro Tier
        **$49/month**
        - All SPY Strategies
        - Real-time signals
        - Entry/Exit alerts
        - Discord support

        [Subscribe →](#)
        """)

    with col3:
        st.markdown("""
        ### Premium Tier
        **$99/month**
        - All 6 Strategies
        - Real-time signals
        - Priority support
        - Strategy insights

        [Subscribe →](#)
        """)

    # Legal disclaimer
    st.markdown("""
    <div class="disclaimer">
    <strong>⚠️ DISCLAIMER</strong><br><br>
    <strong>NOT FINANCIAL ADVICE:</strong> The information provided on this dashboard and through our Discord signals
    is for educational and informational purposes only. It should not be considered as financial, investment,
    or trading advice.<br><br>
    <strong>RISK WARNING:</strong> Trading stocks, ETFs, cryptocurrencies, and other financial instruments involves
    substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.
    You could lose some or all of your investment.<br><br>
    <strong>NO GUARANTEES:</strong> We make no guarantees about the accuracy, completeness, or timeliness of the
    signals or information provided. Historical returns shown are backtested results and may not reflect actual
    trading performance.<br><br>
    <strong>INDEPENDENT DECISION:</strong> You are solely responsible for your own investment decisions.
    Always do your own research and consider consulting with a licensed financial advisor before making
    any investment decisions.<br><br>
    <strong>HYPOTHETICAL PERFORMANCE:</strong> Many of the results shown may be based on hypothetical or
    backtested performance, which has inherent limitations and may not reflect actual trading results.
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(f"_Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")


if __name__ == "__main__":
    main()
