"""
Weekly Performance Report Bot

Automatically generates and posts weekly performance reports to Discord.
Run weekly via cron or manually: python weekly_report_bot.py

Cron example (every Sunday at 6 PM):
0 18 * * 0 cd /path/to/Pattern_FindR && python weekly_report_bot.py
"""

import json
import os
from datetime import datetime, timedelta
import requests

# Discord webhook for announcements channel
ANNOUNCEMENTS_WEBHOOK = os.environ.get("DISCORD_ANNOUNCEMENTS_WEBHOOK_URL", "")

# Strategy directories
VELOCITY_STRATEGIES_DIR = "velocity_strategies"


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


def get_weekly_trades(history: list, days: int = 7) -> list:
    """Filter trades from the last N days."""
    cutoff = datetime.now() - timedelta(days=days)
    weekly_trades = []

    for trade in history:
        exit_time = trade.get('exit_time', '')
        if exit_time:
            try:
                trade_date = datetime.strptime(exit_time[:19], '%Y-%m-%d %H:%M:%S')
                if trade_date >= cutoff:
                    weekly_trades.append(trade)
            except:
                pass

    return weekly_trades


def calculate_stats(trades: list) -> dict:
    """Calculate performance stats from trades."""
    if not trades:
        return {
            'signals': 0,
            'winners': 0,
            'losers': 0,
            'win_rate': 0,
            'total_return': 0,
        }

    winners = [t for t in trades if t.get('pnl_pct', 0) > 0]
    losers = [t for t in trades if t.get('pnl_pct', 0) <= 0]
    total_return = sum(t.get('pnl_pct', 0) for t in trades)

    return {
        'signals': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(trades) * 100 if trades else 0,
        'total_return': total_return,
    }


def load_all_strategies() -> dict:
    """Load all strategy configs."""
    strategies = {}

    # Define the 6 main strategies
    strategy_folders = {
        'SPY 5Y': 'velocity_SPY_5y',
        'SPY 2Y': 'velocity_SPY_2y',
        'SPY 1Y': 'velocity_SPY_1y',
        'BTC 5Y': 'velocity_BTC_5y',
        'BTC 2Y': 'velocity_BTC_2y',
        'BTC 1Y': 'velocity_BTC_1y',
    }

    for label, folder_prefix in strategy_folders.items():
        # Find matching folder
        if os.path.exists(VELOCITY_STRATEGIES_DIR):
            for item in os.listdir(VELOCITY_STRATEGIES_DIR):
                if item.startswith(folder_prefix) and os.path.isdir(os.path.join(VELOCITY_STRATEGIES_DIR, item)):
                    config_path = os.path.join(VELOCITY_STRATEGIES_DIR, item, 'velocity_config.json')
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        strategies[label] = {
                            'config': config,
                            'strategy_name': config.get('strategy_name', ''),
                        }
                        break

    return strategies


def generate_weekly_report() -> str:
    """Generate the weekly performance report message."""
    strategies = load_all_strategies()

    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    report = f"""═══════════════════════════════════════
📊 **WEEKLY PERFORMANCE REPORT**
Week of {start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}
═══════════════════════════════════════

**SPY STRATEGIES**
"""

    spy_total = {'signals': 0, 'winners': 0, 'losers': 0, 'return': 0}
    btc_total = {'signals': 0, 'winners': 0, 'losers': 0, 'return': 0}

    # SPY strategies
    for label in ['SPY 5Y', 'SPY 2Y', 'SPY 1Y']:
        if label in strategies:
            strategy_name = strategies[label]['strategy_name']
            history = load_trade_history(strategy_name)
            weekly = get_weekly_trades(history)
            stats = calculate_stats(weekly)

            spy_total['signals'] += stats['signals']
            spy_total['winners'] += stats['winners']
            spy_total['losers'] += stats['losers']
            spy_total['return'] += stats['total_return']

            report += f"""
📈 {label}
→ Signals: {stats['signals']}
→ Winners: {stats['winners']} | Losers: {stats['losers']}
→ Win Rate: {stats['win_rate']:.0f}%
→ Weekly Return: {stats['total_return']:+.1f}%
"""

    report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**BTC STRATEGIES** (Premium)
"""

    # BTC strategies
    for label in ['BTC 5Y', 'BTC 2Y', 'BTC 1Y']:
        if label in strategies:
            strategy_name = strategies[label]['strategy_name']
            history = load_trade_history(strategy_name)
            weekly = get_weekly_trades(history)
            stats = calculate_stats(weekly)

            btc_total['signals'] += stats['signals']
            btc_total['winners'] += stats['winners']
            btc_total['losers'] += stats['losers']
            btc_total['return'] += stats['total_return']

            report += f"""
₿ {label}
→ Signals: {stats['signals']}
→ Winners: {stats['winners']} | Losers: {stats['losers']}
→ Win Rate: {stats['win_rate']:.0f}%
→ Weekly Return: {stats['total_return']:+.1f}%
"""

    # Combined stats
    total_signals = spy_total['signals'] + btc_total['signals']
    total_winners = spy_total['winners'] + btc_total['winners']
    total_losers = spy_total['losers'] + btc_total['losers']
    total_return = spy_total['return'] + btc_total['return']
    overall_win_rate = (total_winners / total_signals * 100) if total_signals > 0 else 0

    report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**COMBINED STATS**

Total Signals: {total_signals}
Total Winners: {total_winners} ✅
Total Losers: {total_losers} ❌
Overall Win Rate: {overall_win_rate:.0f}%
Combined Return: {total_return:+.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_Past performance does not guarantee future results. Trade responsibly._

═══════════════════════════════════════"""

    return report


def send_to_discord(webhook_url: str, message: str):
    """Send message to Discord webhook."""
    if not webhook_url or webhook_url == "YOUR_ANNOUNCEMENTS_WEBHOOK_HERE":
        print("⚠️  No webhook configured. Printing report instead:")
        print(message)
        return False

    try:
        payload = {"content": message}
        response = requests.post(webhook_url, json=payload)
        if response.status_code in [200, 204]:
            print("✅ Weekly report sent to Discord!")
            return True
        else:
            print(f"❌ Discord error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("📊 Generating Weekly Performance Report...")
    report = generate_weekly_report()
    send_to_discord(ANNOUNCEMENTS_WEBHOOK, report)


if __name__ == "__main__":
    main()
