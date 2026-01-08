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

# Import the SAME chart generator and backtest functions as the paid channels
try:
    from velocity_live_trader import (
        generate_velocity_chart,
        run_historical_backtest,
        calculate_composite_oscillator,
        calculate_velocity_signals,
        load_config,
        load_locked_backtest
    )
    VELOCITY_CHARTS_AVAILABLE = True
except ImportError as e:
    VELOCITY_CHARTS_AVAILABLE = False
    print(f"Warning: Could not import velocity charts: {e}")

# Discord webhook for FREE tier delayed signals channel
FREE_TIER_WEBHOOK = ""

# Secondary webhook for Haus Hedge server (delayed signals)
HAUS_HEDGE_DELAYED_WEBHOOK = ""

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


def find_strategy_config(strategy_name: str) -> dict:
    """Find and load the config for a given strategy name."""
    strategies_dir = "velocity_strategies"

    if not os.path.exists(strategies_dir):
        return None

    # Search for matching strategy directory
    for item in os.listdir(strategies_dir):
        strategy_path = os.path.join(strategies_dir, item)
        config_file = os.path.join(strategy_path, "velocity_config.json")

        if os.path.isdir(strategy_path) and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                # Match by strategy_name in config
                if config.get('strategy_name') == strategy_name:
                    return config
            except:
                pass

    return None


def generate_trade_chart(trade: dict, strategy_name: str) -> io.BytesIO:
    """Generate professional velocity chart matching paid channels."""

    # Try to use velocity charts if available
    if VELOCITY_CHARTS_AVAILABLE:
        try:
            ticker = get_ticker_from_strategy(strategy_name)
            label = get_strategy_label(strategy_name)

            # Find strategy config
            config = find_strategy_config(strategy_name)
            if not config:
                print(f"   ⚠️ Could not find config for {strategy_name}, using simple chart")
                return generate_simple_chart(trade, strategy_name)

            # Load the locked backtest (prevents repainting - uses same markers as live trader)
            locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)
            if locked_backtest:
                print(f"   🔒 Using locked backtest for consistent markers")

            # Fetch data (200 days for full chart context)
            print(f"   📊 Fetching {ticker} data for professional chart...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=200)

            df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
            if df.empty:
                return generate_simple_chart(trade, strategy_name)

            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.lower()

            # Calculate oscillators and run backtest for stats
            df = calculate_composite_oscillator(df, config)
            backtest = run_historical_backtest(df, config)

            # Generate the SAME professional chart as paid channels
            # Use locked_backtest for markers to prevent repainting
            chart_buf = generate_velocity_chart(
                df, backtest, config, ticker,
                title_suffix=f" - {label} (24hr Delayed)",
                locked_backtest=locked_backtest
            )

            print(f"   ✅ Professional velocity chart generated")
            return chart_buf

        except Exception as e:
            print(f"   ⚠️ Velocity chart failed: {e}, falling back to simple chart")
            import traceback
            traceback.print_exc()
            return generate_simple_chart(trade, strategy_name)

    # Fallback to simple chart
    return generate_simple_chart(trade, strategy_name)


def generate_simple_chart(trade: dict, strategy_name: str) -> io.BytesIO:
    """Generate a simple fallback chart showing the trade entry/exit."""
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
    """Get cumulative stats for a strategy from FULL BACKTEST (not just live trades)."""

    # Try to get full backtest stats (much more impressive for FOMO)
    if VELOCITY_CHARTS_AVAILABLE:
        try:
            config = find_strategy_config(strategy_name)
            if config:
                ticker = config.get('ticker', 'SPY')

                # Fetch full historical data for backtest
                end_date = datetime.now()
                start_date = end_date - timedelta(days=config.get('data_period_days', 365))

                df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
                if not df.empty:
                    # Handle MultiIndex columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = df.columns.str.lower()

                    # Run full backtest
                    df = calculate_composite_oscillator(df, config)
                    backtest = run_historical_backtest(df, config)

                    # Stats are directly in backtest dict (not under 'stats' key)
                    num_trades = backtest.get('num_trades', 0)
                    if num_trades > 0:
                        # Calculate avg win/loss from exits
                        exits = backtest.get('exits', [])
                        winners = [e for e in exits if e.get('pnl', 0) > 0]
                        losers = [e for e in exits if e.get('pnl', 0) < 0]
                        avg_win = sum(e['pnl'] for e in winners) / len(winners) if winners else 0
                        avg_loss = sum(e['pnl'] for e in losers) / len(losers) if losers else 0

                        return {
                            'total_trades': num_trades,
                            'win_rate': backtest.get('win_rate', 0),
                            'total_pnl': backtest.get('total_return', 0),
                            'avg_win': avg_win,
                            'avg_loss': avg_loss,
                        }
        except Exception as e:
            print(f"   ⚠️ Could not get backtest stats: {e}")

    # Fallback to live trade history
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


def find_latest_signal() -> dict:
    """Find the most recent completed trade across all strategies (for startup post)."""
    strategies = [
        'velocity_SPY_5y',
        'velocity_SPY_2y',
        'velocity_SPY_1y',
    ]

    latest = None
    latest_time = None

    for strategy_name in strategies:
        history = load_trade_history(strategy_name)

        for trade in history:
            exit_time_str = trade.get('exit_time', '')
            if exit_time_str:
                try:
                    exit_time = datetime.strptime(exit_time_str[:19], '%Y-%m-%d %H:%M:%S')
                    if latest_time is None or exit_time > latest_time:
                        latest_time = exit_time
                        latest = {
                            'signal_id': f"{strategy_name}_{trade.get('entry_time', '')}_{exit_time_str}",
                            'trade': trade,
                            'strategy_name': strategy_name,
                            'label': get_strategy_label(strategy_name),
                        }
                except:
                    pass

    return latest


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
    """Format a delayed signal for Discord with FOMO messaging."""
    trade = signal['trade']
    label = signal['label']
    strategy_name = signal['strategy_name']

    entry_price = trade.get('entry_price', 0)
    exit_price = trade.get('exit_price', 0)
    entry_time = trade.get('entry_time', '')[:16] if trade.get('entry_time') else 'N/A'
    exit_time = trade.get('exit_time', '')[:16] if trade.get('exit_time') else 'N/A'
    pnl = trade.get('pnl_pct', 0)
    exit_reason = trade.get('exit_reason', 'Signal Exit')
    pnl_dollars = trade.get('pnl_dollars', pnl * 100)  # Assume $10k position if not specified

    pnl_emoji = "✅" if pnl > 0 else "❌"

    # Calculate missed profit messaging
    if pnl > 0:
        missed_msg = (
            f"💸 **You missed +{pnl:.2f}% profit!**\n"
            f"_Pro members got this signal 24 hours ago and captured this gain._\n"
        )
    else:
        missed_msg = (
            f"⚠️ **Pro members exited this trade 24 hours ago.**\n"
            f"_Real-time signals help cut losses faster._\n"
        )

    # Get strategy stats for cumulative missed profits (FULL BACKTEST)
    stats = get_strategy_stats(strategy_name)
    stats_section = ""
    if stats:
        # Determine backtest period from strategy name
        if '5y' in strategy_name.lower():
            period_label = "5-Year"
        elif '2y' in strategy_name.lower():
            period_label = "2-Year"
        elif '1y' in strategy_name.lower():
            period_label = "1-Year"
        else:
            period_label = "Historical"

        # Calculate what they would have made with paid signals
        missed_total = stats['total_pnl']
        avg_win = stats.get('avg_win', 0)
        avg_loss = stats.get('avg_loss', 0)

        if missed_total > 0:
            stats_section = (
                f"---\n"
                f"📊 **{period_label} Backtest Performance:**\n"
                f"• **{stats['total_trades']} Trades** | **{stats['win_rate']:.0f}% Win Rate**\n"
                f"• Avg Win: +{avg_win:.1f}% | Avg Loss: {avg_loss:.1f}%\n"
                f"• **Total Return: +{missed_total:.1f}%** 💰\n"
                f"• _On a $100,000 account: **+${missed_total * 1000:,.0f} profit**_\n"
            )
        else:
            stats_section = (
                f"---\n"
                f"📊 **{period_label} Backtest Performance:**\n"
                f"• {stats['total_trades']} Trades | {stats['win_rate']:.0f}% Win Rate\n"
                f"• Total Return: {missed_total:.1f}%\n"
            )

    message = f"""⏰ **[DELAYED 24hr] Signal Result - {label}**

{missed_msg}
📈 **Entry:** ${entry_price:.2f}
📅 **Entry Time:** {entry_time}

📉 **Exit:** ${exit_price:.2f}
📅 **Exit Time:** {exit_time}

💰 **Result:** {pnl:+.2f}% {pnl_emoji}
📋 **Exit Reason:** {exit_reason}
{stats_section}
---
🚀 **Stop watching from the sidelines!**
_Upgrade to Pro for REAL-TIME signals and never miss another trade._
{LEGAL_DISCLAIMER}"""

    return message


def send_to_discord(message: str, chart_buf: io.BytesIO = None) -> bool:
    """Send message to Discord webhook with optional chart. Posts to both servers."""
    if not FREE_TIER_WEBHOOK or FREE_TIER_WEBHOOK == "YOUR_FREE_TIER_WEBHOOK_HERE":
        print(f"⚠️  No webhook configured. Would post:")
        print(message[:200] + "...")
        return True  # Return True to mark as "posted" for testing

    def post_to_webhook(url: str, msg: str, chart: io.BytesIO = None) -> bool:
        """Helper to post to a single webhook."""
        try:
            if chart:
                chart.seek(0)
                files = {'file': ('trade_chart.png', chart, 'image/png')}
                payload = {'content': msg}
                response = requests.post(url, data=payload, files=files)
            else:
                payload = {"content": msg}
                response = requests.post(url, json=payload)

            if response.status_code in [200, 204]:
                return True
            else:
                print(f"❌ Discord error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    # Send to primary webhook (your server)
    primary_success = post_to_webhook(FREE_TIER_WEBHOOK, message, chart_buf)

    # Send to secondary webhook (Haus Hedge server)
    if HAUS_HEDGE_DELAYED_WEBHOOK:
        if chart_buf:
            chart_buf.seek(0)
        secondary_success = post_to_webhook(HAUS_HEDGE_DELAYED_WEBHOOK, message, chart_buf)
        if secondary_success:
            print(f"   📤 Also posted to Haus Hedge server")

    return primary_success


def run_delayed_bot():
    """Main bot loop."""
    print("🕐 Delayed Signal Bot Starting...")
    print(f"   Delay: {SIGNAL_DELAY_HOURS} hours")
    print(f"   Check interval: {CHECK_INTERVAL} seconds")
    print(f"   Webhook: {'Configured' if FREE_TIER_WEBHOOK != 'YOUR_FREE_TIER_WEBHOOK_HERE' else 'NOT SET'}")
    print()

    # Send startup message to both servers
    startup_msg = f"""**Delayed Signal Bot Online**

This channel receives trading signals with a 24-hour delay.
Signals are from our velocity-based strategies monitoring SPY and BTC.

Want real-time signals? Upgrade to premium!

_Bot started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EST_"""

    send_to_discord(startup_msg)
    print("   📤 Startup message sent to both servers")

    # Post the most recent signal as an example
    try:
        latest_signal = find_latest_signal()
        if latest_signal:
            print(f"   📊 Posting latest signal: {latest_signal['label']}")
            message = format_signal_message(latest_signal)
            chart_buf = None
            try:
                chart_buf = generate_trade_chart(latest_signal['trade'], latest_signal['strategy_name'])
            except Exception as e:
                print(f"   ⚠️ Chart generation failed: {e}")
            send_to_discord(message, chart_buf)
            print(f"   ✅ Latest signal posted")
    except Exception as e:
        print(f"   ⚠️ Could not post latest signal: {e}")

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
