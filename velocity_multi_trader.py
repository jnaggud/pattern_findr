"""
Velocity Multi-Strategy Trader

Runs multiple velocity strategies in a single process with interactive selection.
Uses velocity_core.py for all shared logic.

Features:
- Interactive strategy selection at startup
- Shared data fetches per ticker (BTC, SPY fetched once each)
- Independent state/backtest files per strategy
- Single main loop processing all selected strategies

Usage:
    python velocity_multi_trader.py
"""

import os
import time
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd

# Import all shared logic from velocity_core
from velocity_core import (
    # Constants
    DEFAULT_DISCORD_WEBHOOK,
    HAUS_HEDGE_WEBHOOKS,
    VELOCITY_STRATEGIES_DIR,
    # Config
    load_config,
    # Data functions
    fetch_price_data,
    fetch_realtime_price,
    # Oscillator functions
    calculate_composite_oscillator,
    calculate_velocity_signals,
    # State management
    get_state_file_path,
    load_trade_state,
    save_trade_state,
    log_closed_trade,
    # Locked backtest
    save_locked_backtest,
    load_locked_backtest,
    append_to_locked_backtest,
    detect_and_add_missed_signals,
    # Backtest
    run_historical_backtest,
    # Discord
    send_discord_alert,
    generate_velocity_chart,
    build_position_section,
)


# ============================================================================
# AVAILABLE STRATEGIES
# ============================================================================

AVAILABLE_STRATEGIES = [
    {
        "name": "velocity_BTC_1y",
        "display_name": "BTC-USD 1Y",
        "ticker": "BTC-USD",
        "lookback": "1y",
        "config_path": "velocity_strategies/velocity_BTC_1y_20251222_103111/velocity_config.json"
    },
    {
        "name": "velocity_BTC_2y",
        "display_name": "BTC-USD 2Y",
        "ticker": "BTC-USD",
        "lookback": "2y",
        "config_path": "velocity_strategies/velocity_BTC_2y_20251222_104158/velocity_config.json"
    },
    {
        "name": "velocity_BTC_5y",
        "display_name": "BTC-USD 5Y",
        "ticker": "BTC-USD",
        "lookback": "5y",
        "config_path": "velocity_strategies/velocity_BTC_5y_20251222_105550/velocity_config.json"
    },
    {
        "name": "velocity_SPY_1y",
        "display_name": "SPY 1Y",
        "ticker": "SPY",
        "lookback": "1y",
        "config_path": "velocity_strategies/velocity_SPY_1y_20251222_101525/velocity_config.json"
    },
    {
        "name": "velocity_SPY_2y",
        "display_name": "SPY 2Y",
        "ticker": "SPY",
        "lookback": "2y",
        "config_path": "velocity_strategies/velocity_SPY_2y_20251222_100334/velocity_config.json"
    },
    {
        "name": "velocity_SPY_5y",
        "display_name": "SPY 5Y",
        "ticker": "SPY",
        "lookback": "5y",
        "config_path": "velocity_strategies/velocity_SPY_5y_20251222_094920/velocity_config.json"
    },
]


# ============================================================================
# INTERACTIVE SELECTION
# ============================================================================

def select_strategies_interactive() -> list:
    """
    Interactive strategy selection at startup.
    Returns list of selected strategy dicts.
    """
    print(f"\n{'='*60}")
    print("VELOCITY MULTI-TRADER - Strategy Selection")
    print(f"{'='*60}\n")

    # Load current position for each strategy
    for strat in AVAILABLE_STRATEGIES:
        try:
            trade_state = load_trade_state(strategy_name=strat['name'], ticker=strat['ticker'])
            position = trade_state.get('position')
            if position:
                entry_price = trade_state.get('entry_price', 0)
                strat['position_display'] = f"{position.upper()} @ ${entry_price:.2f}"
            else:
                strat['position_display'] = "None"
        except:
            strat['position_display'] = "Unknown"

    # Display available strategies
    print("Available Strategies:")
    for i, strat in enumerate(AVAILABLE_STRATEGIES, 1):
        print(f"  [{i}] {strat['display_name']:12} - Position: {strat['position_display']}")

    print(f"\nEnter strategy numbers (comma-separated), 'all', or 'q' to quit:")

    while True:
        user_input = input("> ").strip().lower()

        if user_input == 'q':
            return []

        if user_input == 'all':
            selected = AVAILABLE_STRATEGIES.copy()
            break

        try:
            # Parse comma-separated numbers
            indices = [int(x.strip()) for x in user_input.split(',')]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(AVAILABLE_STRATEGIES):
                    selected.append(AVAILABLE_STRATEGIES[idx - 1])
                else:
                    print(f"Invalid number: {idx}. Please enter 1-{len(AVAILABLE_STRATEGIES)}")
                    selected = None
                    break
            if selected is not None:
                break
        except ValueError:
            print("Invalid input. Enter numbers like '1,2,3' or 'all' or 'q'")

    # Confirm selection
    if selected:
        print(f"\nSelected {len(selected)} strategies:")
        for strat in selected:
            print(f"  - {strat['display_name']}")
        print(f"\nPress Enter to start, or 'q' to quit:")
        confirm = input().strip().lower()
        if confirm == 'q':
            return []

    return selected


# ============================================================================
# STRATEGY INITIALIZATION
# ============================================================================

def initialize_strategy(strat: dict, ticker_data: dict) -> bool:
    """
    Initialize a single strategy - load config, run backtest, sync state.
    Returns True if successful.
    """
    strategy_name = strat['name']
    ticker = strat['ticker']
    config_path = strat['config_path']

    print(f"\n--- Initializing {strategy_name} ---")

    try:
        # Load config
        if not os.path.exists(config_path):
            print(f"   Config not found: {config_path}")
            return False

        config = load_config(config_path)
        strat['config'] = config

        # Get data for this ticker (already fetched)
        df = ticker_data.get(ticker)
        if df is None or df.empty:
            print(f"   No data for {ticker}")
            return False

        # Calculate oscillator
        df = calculate_composite_oscillator(df.copy(), config)
        strat['df'] = df

        # Run backtest
        backtest = run_historical_backtest(df, config)
        strat['backtest'] = backtest

        # Load trade state
        trade_state = load_trade_state(strategy_name=strategy_name, ticker=ticker)
        strat['trade_state'] = trade_state

        # Load or create locked backtest
        locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)
        if locked_backtest is None:
            locked_backtest = save_locked_backtest(backtest, strategy_name=strategy_name,
                                                    ticker=ticker, trade_state=trade_state)
        strat['locked_backtest'] = locked_backtest

        # Get webhook URL
        webhook_url = config.get('discord_webhook') or DEFAULT_DISCORD_WEBHOOK
        strat['webhook_url'] = webhook_url

        # Print summary
        position = trade_state.get('position', 'None')
        print(f"   Trades: {backtest['num_trades']} | Win Rate: {backtest['win_rate']:.0f}% | "
              f"Return: {backtest['total_return']:.1f}% | Position: {position}")

        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN LOOP
# ============================================================================

def process_strategy(strat: dict, current_price: float, df_fresh: pd.DataFrame) -> None:
    """
    Process a single strategy - check for exits and entries.
    """
    config = strat['config']
    strategy_name = strat['name']
    ticker = strat['ticker']
    trade_state = strat['trade_state']
    webhook_url = strat['webhook_url']

    # Calculate signals on fresh data
    df = calculate_composite_oscillator(df_fresh.copy(), config)
    df = calculate_velocity_signals(df, config)

    # Get last completed bar (not forming bar)
    last_bar = df.iloc[-2]  # Yesterday's bar for daily
    signal_date = df.index[-2]

    # Get current position
    position = trade_state.get('position')
    entry_price = trade_state.get('entry_price', 0)

    # Risk parameters
    stop_loss_pct = config.get('stop_loss_pct', 5.0)
    take_profit_pct = config.get('take_profit_pct', 10.0)
    exit_on_opposite = config.get('exit_on_opposite_signal', True)
    exit_on_midline = config.get('exit_on_midline_cross', False)

    strategy_label = f"{ticker} {strat['lookback'].upper()}"

    # ========================================
    # CHECK EXITS (if in position) - LONG ONLY
    # ========================================
    if position == 'long':
        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        exit_reason = None
        osc = last_bar.get('osc_smooth', 0)

        # Check exit conditions
        if stop_loss_pct > 0 and pnl_pct <= -stop_loss_pct:
            exit_reason = "Stop Loss"
        elif take_profit_pct > 0 and pnl_pct >= take_profit_pct:
            exit_reason = "Take Profit"
        elif exit_on_midline and osc > 0:
            exit_reason = "Midline Cross"
        elif exit_on_opposite and last_bar.get('sell_signal', False):
            exit_reason = "Opposite Signal"

        if exit_reason:
            pnl_emoji = "+" if pnl_pct >= 0 else ""
            pnl_dollars = current_price - entry_price

            # Log trade
            entry_time = trade_state.get('entry_time', '')
            log_closed_trade(
                ticker=ticker,
                position_type=position.upper(),
                entry_price=entry_price,
                exit_price=current_price,
                entry_time=entry_time,
                exit_time=str(datetime.now()),
                exit_reason=exit_reason,
                pnl_pct=pnl_pct,
                strategy_name=strategy_name
            )

            # Append to locked backtest
            append_to_locked_backtest(
                exit_trade={
                    'date': str(signal_date),
                    'price': current_price,
                    'pnl': pnl_pct,
                    'reason': exit_reason,
                    'entry_price': entry_price,
                    'entry_date': entry_time
                },
                strategy_name=strategy_name,
                ticker=ticker
            )

            # Send Discord alert
            exit_msg = (
                f"**[{strategy_label}] {position.upper()} EXIT** ({exit_reason})\n"
                f"Entry: ${entry_price:.2f} -> Exit: ${current_price:.2f}\n"
                f"**P&L: {pnl_emoji}{pnl_pct:.2f}%** (${pnl_dollars:+,.0f})"
            )
            send_discord_alert(webhook_url, exit_msg, strategy_name=strategy_name)
            print(f"   [{strategy_label}] EXIT: {exit_reason} | P&L: {pnl_emoji}{pnl_pct:.2f}%")

            # Clear position
            trade_state['position'] = None
            trade_state['entry_price'] = None
            trade_state['entry_time'] = None
            save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)
            strat['trade_state'] = trade_state
            return

    # ========================================
    # CHECK ENTRIES (if no position)
    # ========================================
    if position is None:
        buy_signal = last_bar.get('buy_signal', False)
        sell_signal = last_bar.get('sell_signal', False)

        # Check if we've already processed this signal
        last_signal = trade_state.get('last_signal_time', '')
        signal_str = str(signal_date)[:10]
        if last_signal and signal_str <= last_signal[:10]:
            return  # Already processed

        if buy_signal:
            # LONG entry
            trade_state['position'] = 'long'
            trade_state['entry_price'] = current_price
            trade_state['entry_time'] = str(datetime.now())
            trade_state['last_signal_time'] = signal_str
            save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)
            strat['trade_state'] = trade_state

            # Append to locked backtest
            append_to_locked_backtest(
                entry={'date': str(signal_date), 'price': current_price, 'position': 'long'},
                strategy_name=strategy_name,
                ticker=ticker
            )

            # Send Discord alert
            buy_msg = (
                f"**[{strategy_label}] LONG ENTRY**\n"
                f"Signal: {signal_str}\n"
                f"Entry: ${current_price:.2f}"
            )
            send_discord_alert(webhook_url, buy_msg, strategy_name=strategy_name)
            print(f"   [{strategy_label}] LONG ENTRY @ ${current_price:.2f}")

        elif sell_signal:
            # LONG-ONLY STRATEGY: Sell signals are ignored when not in position
            # They are only used to EXIT existing long positions (handled above)
            print(f"   [{strategy_label}] Sell signal ignored (LONG-only strategy, no position)")
            trade_state['last_signal_time'] = signal_str
            save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)
            strat['trade_state'] = trade_state


def run_multi_trader():
    """Main multi-strategy trading loop."""

    # Interactive selection
    selected_strategies = select_strategies_interactive()

    if not selected_strategies:
        print("No strategies selected. Exiting.")
        return

    print(f"\n{'='*60}")
    print(f"STARTING MULTI-TRADER WITH {len(selected_strategies)} STRATEGIES")
    print(f"{'='*60}")

    # Group strategies by ticker for data sharing
    strategies_by_ticker = defaultdict(list)
    for strat in selected_strategies:
        strategies_by_ticker[strat['ticker']].append(strat)

    print(f"\nTickers: {list(strategies_by_ticker.keys())}")

    # ========================================
    # STARTUP: Fetch data and initialize
    # ========================================
    print("\nFetching initial data...")
    ticker_data = {}
    for ticker in strategies_by_ticker.keys():
        # Get the max days needed for any strategy on this ticker
        max_days = 365 * 5  # 5 years to cover all strategies
        print(f"   Fetching {ticker} ({max_days} days)...")
        df = fetch_price_data(ticker, days=max_days, interval="1d")
        if not df.empty:
            ticker_data[ticker] = df
            print(f"   {ticker}: {len(df)} bars loaded")
        else:
            print(f"   WARNING: No data for {ticker}")

    # Initialize each strategy
    print("\nInitializing strategies...")
    for strat in selected_strategies:
        success = initialize_strategy(strat, ticker_data)
        if not success:
            strat['enabled'] = False
        else:
            strat['enabled'] = True
            strat['error_count'] = 0

    # Filter to enabled strategies
    enabled_strategies = [s for s in selected_strategies if s.get('enabled', False)]
    print(f"\n{len(enabled_strategies)} strategies initialized successfully")

    if not enabled_strategies:
        print("No strategies could be initialized. Exiting.")
        return

    # ========================================
    # MAIN LOOP
    # ========================================
    check_interval_seconds = 900  # 15 minutes

    print(f"\nEntering main loop (check every {check_interval_seconds // 60} min)")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
            loop_start = datetime.now()
            print(f"\n[{loop_start.strftime('%Y-%m-%d %H:%M:%S')}] Update Cycle")

            # ========================================
            # PHASE 1: FETCH FRESH DATA (once per ticker)
            # ========================================
            ticker_prices = {}
            ticker_fresh_data = {}

            for ticker in strategies_by_ticker.keys():
                try:
                    # Fetch fresh price data (200 days for signals)
                    df_fresh = fetch_price_data(ticker, days=200, interval="1d")
                    if not df_fresh.empty:
                        ticker_fresh_data[ticker] = df_fresh

                    # Get real-time price
                    realtime_price = fetch_realtime_price(ticker)
                    if realtime_price:
                        ticker_prices[ticker] = realtime_price
                    elif not df_fresh.empty:
                        ticker_prices[ticker] = df_fresh['close'].iloc[-1]

                    print(f"   {ticker}: ${ticker_prices.get(ticker, 0):.2f}")

                except Exception as e:
                    print(f"   ERROR fetching {ticker}: {e}")

            # ========================================
            # PHASE 2: PROCESS EACH STRATEGY
            # ========================================
            for strat in enabled_strategies:
                ticker = strat['ticker']
                strategy_name = strat['name']

                try:
                    if ticker not in ticker_fresh_data:
                        continue

                    current_price = ticker_prices.get(ticker, 0)
                    df_fresh = ticker_fresh_data[ticker]

                    process_strategy(strat, current_price, df_fresh)

                    # Reset error count on success
                    strat['error_count'] = 0

                except Exception as e:
                    strat['error_count'] = strat.get('error_count', 0) + 1
                    print(f"   ERROR in {strategy_name}: {e}")

                    # Disable after 10 consecutive errors
                    if strat['error_count'] >= 10:
                        strat['enabled'] = False
                        print(f"   DISABLED {strategy_name} after 10 errors")

            # ========================================
            # PHASE 3: SLEEP UNTIL NEXT CYCLE
            # ========================================
            elapsed = (datetime.now() - loop_start).total_seconds()
            sleep_time = max(0, check_interval_seconds - elapsed)

            # Show status
            active_count = sum(1 for s in enabled_strategies if s.get('enabled', False))
            print(f"\n   {active_count} strategies active. Next check in {int(sleep_time)}s...")

            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nShutting down multi-trader...")

            # Send shutdown notification
            for strat in enabled_strategies:
                try:
                    msg = f"**[{strat['display_name']}] Multi-Trader Stopped**"
                    send_discord_alert(strat.get('webhook_url', DEFAULT_DISCORD_WEBHOOK),
                                      msg, strategy_name=strat['name'])
                except:
                    pass

            break

        except Exception as e:
            print(f"\nLoop error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)


if __name__ == "__main__":
    run_multi_trader()
