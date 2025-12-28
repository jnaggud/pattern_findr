"""
Delayed Signal Bot for Free Tier

Monitors trade history and posts signals to the free tier channel with a 24-hour delay.
Run continuously: python delayed_signal_bot.py

This ensures free tier users get real signals, just delayed, which:
1. Proves the system works (builds trust)
2. Creates FOMO for real-time signals (drives upgrades)
3. Provides value without giving away the edge
"""

import json
import os
import time
import io
from datetime import datetime, timedelta
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Try to import yfinance for chart generation
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Charts will not be generated.")

# Discord webhook for FREE tier delayed signals channel
FREE_TIER_WEBHOOK = ""

# Delay in hours before posting signals to free tier
SIGNAL_DELAY_HOURS = 24

# Check interval in seconds (every hour)
CHECK_INTERVAL = 3600

# Track which signals have been posted
POSTED_SIGNALS_FILE = "posted_delayed_signals.json"

# Legal disclaimer
LEGAL_DISCLAIMER = (
    "\n\n_⚠️ **Disclaimer:** This signal is delayed 24 hours. "
    "Past performance does not guarantee future results. "
    "Not financial advice. Trade at your own risk._"
)


def load_posted_signals() -> set:
    """Load set of already posted signal IDs."""
    if os.path.exists(POSTED_SIGNALS_FILE):
        try:
            with open(POSTED_SIGNALS_FILE, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()


def save_posted_signals(posted: set):
    """Save posted signal IDs."""
    with open(POSTED_SIGNALS_FILE, 'w') as f:
        json.dump(list(posted), f)


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


def get_strategy_label(strategy_name: str) -> str:
    """Convert strategy name to display label."""
    if 'SPY' in strategy_name.upper():
        if '5y' in strategy_name.lower():
            return 'SPY 5Y'
        elif '2y' in strategy_name.lower():
            return 'SPY 2Y'
        elif '1y' in strategy_name.lower():
            return 'SPY 1Y'
    elif 'BTC' in strategy_name.upper():
        if '5y' in strategy_name.lower():
            return 'BTC 5Y'
        elif '2y' in strategy_name.lower():
            return 'BTC 2Y'
        elif '1y' in strategy_name.lower():
            return 'BTC 1Y'
    return strategy_name


def get_ticker_from_strategy(strategy_name: str) -> str:
    """Get ticker symbol from strategy name."""
    if 'SPY' in strategy_name.upper():
        return 'SPY'
    elif 'BTC' in strategy_name.upper():
        return 'BTC-USD'
    return 'SPY'


def generate_trade_chart(trade: dict, strategy_name: str) -> io.BytesIO:
    """Generate a simple chart showing the trade entry/exit."""
    if not YFINANCE_AVAILABLE:
        return None

    try:
        ticker = get_ticker_from_strategy(strategy_name)
        label = get_strategy_label(strategy_name)

        # Parse trade times
        entry_time_str = trade.get('entry_time', '')
        exit_time_str = trade.get('exit_time', '')

        if not entry_time_str or not exit_time_str:
            return None

        entry_time = datetime.strptime(entry_time_str[:19], '%Y-%m-%d %H:%M:%S')
        exit_time = datetime.strptime(exit_time_str[:19], '%Y-%m-%d %H:%M:%S')

        # Fetch data for chart (30 days around the trade)
        start_date = entry_time - timedelta(days=15)
        end_date = exit_time + timedelta(days=5)

        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if df.empty:
            return None

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.lower()

        # Create chart
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot price
        ax.plot(df.index, df['close'], color='white', linewidth=1.5, label='Price')

        # Mark entry and exit
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        pnl = trade.get('pnl_pct', 0)

        # Entry marker
        ax.axhline(y=entry_price, color='#00ff00', linestyle='--', alpha=0.7, label=f'Entry ${entry_price:.2f}')

        # Exit marker
        exit_color = '#00ff00' if pnl > 0 else '#ff4444'
        ax.axhline(y=exit_price, color=exit_color, linestyle='--', alpha=0.7, label=f'Exit ${exit_price:.2f}')

        # Shade the trade period
        ax.axvspan(entry_time, exit_time, alpha=0.2, color='#00ff00' if pnl > 0 else '#ff4444')

        # Labels and title
        pnl_emoji = "✅" if pnl > 0 else "❌"
        ax.set_title(f'{ticker} - {label} Trade Result: {pnl:+.2f}% {pnl_emoji}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add trade info text
        info_text = f"Entry: {entry_time.strftime('%Y-%m-%d')} @ ${entry_price:.2f}\nExit: {exit_time.strftime('%Y-%m-%d')} @ ${exit_price:.2f}"
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor='#1a1a2e')
        buf.seek(0)
        plt.close(fig)

        return buf

    except Exception as e:
        print(f"   ⚠️ Could not generate chart: {e}")
        return None


def get_strategy_stats(strategy_name: str) -> dict:
    """Get cumulative stats for a strategy from trade history."""
    history = load_trade_history(strategy_name)

    if not history:
        return None

    total_trades = len(history)
    winners = [t for t in history if t.get('pnl_pct', 0) > 0]
    win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum(t.get('pnl_pct', 0) for t in history)

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
    }


def find_delayed_signals() -> list:
    """Find signals that should be posted (24hr+ old, not yet posted)."""
    posted = load_posted_signals()
    signals_to_post = []

    # Define strategies to monitor (only SPY for free tier - BTC is premium only)
    strategies = [
        'velocity_SPY_5y',
        'velocity_SPY_2y',
        'velocity_SPY_1y',
    ]

    cutoff_time = datetime.now() - timedelta(hours=SIGNAL_DELAY_HOURS)

    for strategy_name in strategies:
        history = load_trade_history(strategy_name)

        for trade in history:
            # Create unique signal ID
            signal_id = f"{strategy_name}_{trade.get('entry_time', '')}_{trade.get('exit_time', '')}"

            # Skip if already posted
            if signal_id in posted:
                continue

            # Check if entry is old enough to post
            entry_time_str = trade.get('entry_time', '')
            if entry_time_str:
                try:
                    entry_time = datetime.strptime(entry_time_str[:19], '%Y-%m-%d %H:%M:%S')

                    # Only post if entry was 24+ hours ago
                    if entry_time <= cutoff_time:
                        signals_to_post.append({
                            'signal_id': signal_id,
                            'trade': trade,
                            'strategy_name': strategy_name,
                            'label': get_strategy_label(strategy_name),
                        })
                except:
                    pass

    return signals_to_post


def format_signal_message(signal: dict) -> str:
    """Format a delayed signal for Discord."""
    trade = signal['trade']
    label = signal['label']
    strategy_name = signal['strategy_name']

    entry_price = trade.get('entry_price', 0)
    exit_price = trade.get('exit_price', 0)
    entry_time = trade.get('entry_time', '')[:16] if trade.get('entry_time') else 'N/A'
    exit_time = trade.get('exit_time', '')[:16] if trade.get('exit_time') else 'N/A'
    pnl = trade.get('pnl_pct', 0)
    exit_reason = trade.get('exit_reason', 'Signal Exit')

    pnl_emoji = "✅" if pnl > 0 else "❌"

    # Get strategy stats
    stats = get_strategy_stats(strategy_name)
    stats_section = ""
    if stats:
        stats_section = (
            f"---\n"
            f"📊 **Strategy Performance:**\n"
            f"• Trades: {stats['total_trades']} | Win Rate: {stats['win_rate']:.0f}%\n"
            f"• Total Return: {stats['total_pnl']:.1f}%\n"
        )

    message = f"""⏰ **[DELAYED 24hr] JD Signal Result - {label}**

📈 **Entry:** ${entry_price:.2f}
📅 **Entry Time:** {entry_time}

📉 **Exit:** ${exit_price:.2f}
📅 **Exit Time:** {exit_time}

💰 **Result:** {pnl:+.2f}% {pnl_emoji}
📋 **Reason:** {exit_reason}
{stats_section}
---
_🔔 Want real-time signals? Upgrade to Pro for instant alerts!_
{LEGAL_DISCLAIMER}"""

    return message


def send_to_discord(message: str, chart_buf: io.BytesIO = None) -> bool:
    """Send message to Discord webhook with optional chart."""
    if not FREE_TIER_WEBHOOK or FREE_TIER_WEBHOOK == "YOUR_FREE_TIER_WEBHOOK_HERE":
        print(f"⚠️  No webhook configured. Would post:")
        print(message[:200] + "...")
        return True  # Return True to mark as "posted" for testing

    try:
        if chart_buf:
            # Send with image
            chart_buf.seek(0)
            files = {'file': ('trade_chart.png', chart_buf, 'image/png')}
            payload = {'content': message}
            response = requests.post(FREE_TIER_WEBHOOK, data=payload, files=files)
        else:
            # Send text only
            payload = {"content": message}
            response = requests.post(FREE_TIER_WEBHOOK, json=payload)

        if response.status_code in [200, 204]:
            return True
        else:
            print(f"❌ Discord error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_delayed_bot():
    """Main bot loop."""
    print("🕐 Delayed Signal Bot Starting...")
    print(f"   Delay: {SIGNAL_DELAY_HOURS} hours")
    print(f"   Check interval: {CHECK_INTERVAL} seconds")
    print(f"   Webhook: {'Configured' if FREE_TIER_WEBHOOK != 'YOUR_FREE_TIER_WEBHOOK_HERE' else 'NOT SET'}")
    print()

    while True:
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for delayed signals...")

            signals = find_delayed_signals()
            posted = load_posted_signals()

            if signals:
                print(f"   Found {len(signals)} signals to post")

                for signal in signals:
                    message = format_signal_message(signal)

                    # Generate chart for the trade
                    chart_buf = None
                    try:
                        print(f"   📊 Generating chart for {signal['label']}...")
                        chart_buf = generate_trade_chart(signal['trade'], signal['strategy_name'])
                        if chart_buf:
                            print(f"   ✅ Chart generated")
                    except Exception as e:
                        print(f"   ⚠️ Chart generation failed: {e}")

                    if send_to_discord(message, chart_buf):
                        posted.add(signal['signal_id'])
                        print(f"   ✅ Posted: {signal['label']} ({signal['trade'].get('pnl_pct', 0):+.2f}%)")

                        # Small delay between posts
                        time.sleep(2)

                save_posted_signals(posted)
            else:
                print("   No new signals to post")

            print(f"   Next check in {CHECK_INTERVAL // 60} minutes...")
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    run_delayed_bot()
