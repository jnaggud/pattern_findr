"""
Discord notification system for velocity trading.

Sends trading alerts, status updates, and charts to Discord webhooks.
"""

import os
import io
import json
import requests
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class DiscordConfig:
    """Discord webhook configuration."""
    webhook_url: str
    username: str = "Velocity Trader"
    avatar_url: str = None
    mention_on_signal: bool = True


def send_discord_alert(
    webhook_url: str,
    message: str,
    chart: io.BytesIO = None,
    csv_buf: io.BytesIO = None,
    embed: Dict = None,
    username: str = "Velocity Trader",
    mention: bool = False
) -> bool:
    """
    Send an alert to Discord.

    Args:
        webhook_url: Discord webhook URL
        message: Message text
        chart: Optional chart image as BytesIO
        csv_buf: Optional CSV file as BytesIO
        embed: Optional Discord embed dict
        username: Bot username
        mention: Whether to include @here mention

    Returns:
        True if sent successfully
    """
    if not webhook_url:
        print("   No Discord webhook URL configured")
        return False

    try:
        # Prepare the message
        if mention and "@here" not in message:
            message = f"@here\n{message}"

        data = {
            "username": username,
            "content": message,
            # CRITICAL: Enable @here and @everyone mentions
            # Discord webhooks suppress these by default for safety
            "allowed_mentions": {
                "parse": ["everyone"]  # Allows both @here and @everyone
            }
        }

        if embed:
            data["embeds"] = [embed]

        files = {}

        # Add chart if provided
        if chart is not None:
            chart.seek(0)
            files["file"] = ("chart.png", chart, "image/png")

        # Add CSV if provided
        if csv_buf is not None:
            csv_buf.seek(0)
            files["file2"] = ("trades.csv", csv_buf, "text/csv")

        if files:
            # Multipart request with files
            response = requests.post(
                webhook_url,
                data={"payload_json": json.dumps(data)},
                files=files,
                timeout=30
            )
        else:
            # JSON request without files
            response = requests.post(
                webhook_url,
                json=data,
                timeout=30
            )

        if response.status_code in [200, 204]:
            return True
        else:
            print(f"   Discord error: {response.status_code} - {response.text[:200]}")
            return False

    except Exception as e:
        print(f"   Discord send failed: {e}")
        return False


def send_entry_alert(
    webhook_url: str,
    strategy_name: str,
    ticker: str,
    entry_price: float,
    signal_time: str,
    stop_loss_pct: float = 2.0,
    take_profit_pct: float = 5.0,
    stats: Dict = None,
    chart: io.BytesIO = None,
    is_delayed: bool = False,
    enhanced_stats: Dict = None
) -> bool:
    """
    Send entry signal alert.

    Args:
        webhook_url: Discord webhook URL
        strategy_name: Strategy label
        ticker: Trading symbol
        entry_price: Entry price
        signal_time: Signal timestamp
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        stats: Optional strategy statistics (legacy format)
        chart: Optional chart image
        is_delayed: True if entering on delayed signal
        enhanced_stats: Optional enhanced stats dict with all_time and recent_2day

    Returns:
        True if sent successfully
    """
    # Calculate SL/TP prices
    sl_price = entry_price * (1 - stop_loss_pct / 100)
    tp_price = entry_price * (1 + take_profit_pct / 100)

    # Position status
    if is_delayed:
        position_status = "Position: LONG (late entry)"
        signal_note = " (delayed signal)"
    else:
        position_status = "Position: LONG"
        signal_note = ""

    # Build stats section - use enhanced stats if available
    stats_section = ""
    if enhanced_stats:
        all_time = enhanced_stats.get('all_time', {})
        recent = enhanced_stats.get('recent_2day', {})
        all_trades = all_time.get('num_trades', 0)
        all_wr = all_time.get('win_rate', 0)
        all_ret = all_time.get('total_return', 0)
        all_dd = all_time.get('max_drawdown', 0)
        recent_trades = recent.get('num_trades', 0)
        recent_wr = recent.get('win_rate', 0)
        recent_ret = recent.get('total_return', 0)
        recent_dd = recent.get('max_drawdown', 0)
        stats_section = (
            f"---\n"
            f"📊 **All-Time** ({all_trades} trades): {all_wr:.0f}% WR | {all_ret:.1f}% Return | DD {all_dd:.1f}%\n"
            f"📈 **Recent 2 Days** ({recent_trades} trades): {recent_wr:.0f}% WR | {recent_ret:.1f}% Return | DD {recent_dd:.1f}%\n"
        )
    elif stats:
        stats_section = (
            f"---\n"
            f"**Strategy Stats:**\n"
            f"Trades: {stats.get('num_trades', 0)} | "
            f"Win Rate: {stats.get('win_rate', 0):.0f}%\n"
            f"Total Return: {stats.get('total_return', 0):.1f}% | "
            f"PF: {stats.get('profit_factor', 0):.1f}\n"
        )

    message = (
        f"@here\n"
        f"🟢 **[{strategy_name}] ENTRY LONG**{signal_note}\n"
        f"---\n"
        f"**Signal Time:** {signal_time}\n"
        f"**Entry Price:** ${entry_price:,.2f}\n"
        f"**{position_status}**\n"
        f"{stats_section}"
        f"---\n"
        f"_SL: {stop_loss_pct:.1f}% (${sl_price:,.2f}) | "
        f"TP: {take_profit_pct:.1f}% (${tp_price:,.2f})_"
    )

    return send_discord_alert(webhook_url, message, chart=chart)


def send_exit_alert(
    webhook_url: str,
    strategy_name: str,
    ticker: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    pnl_dollars: float,
    exit_reason: str,
    entry_time: str,
    exit_time: str,
    cumulative_stats: Dict = None,
    chart: io.BytesIO = None,
    csv_buf: io.BytesIO = None,
    enhanced_stats: Dict = None
) -> bool:
    """
    Send exit signal alert.

    Args:
        webhook_url: Discord webhook URL
        strategy_name: Strategy label
        ticker: Trading symbol
        entry_price: Entry price
        exit_price: Exit price
        pnl_pct: P&L percentage
        pnl_dollars: P&L in dollars
        exit_reason: Reason for exit
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        cumulative_stats: Optional cumulative statistics (legacy format)
        chart: Optional chart image
        csv_buf: Optional trade log CSV
        enhanced_stats: Optional enhanced stats dict with all_time and recent_2day

    Returns:
        True if sent successfully
    """
    # P&L emoji
    if pnl_pct > 0:
        pnl_emoji = ""
    else:
        pnl_emoji = ""

    # Build stats section - use enhanced stats if available
    stats_section = ""
    if enhanced_stats:
        all_time = enhanced_stats.get('all_time', {})
        recent = enhanced_stats.get('recent_2day', {})
        all_trades = all_time.get('num_trades', 0)
        all_wr = all_time.get('win_rate', 0)
        all_ret = all_time.get('total_return', 0)
        all_dd = all_time.get('max_drawdown', 0)
        recent_trades = recent.get('num_trades', 0)
        recent_wr = recent.get('win_rate', 0)
        recent_ret = recent.get('total_return', 0)
        recent_dd = recent.get('max_drawdown', 0)
        stats_section = (
            f"---\n"
            f"📊 **All-Time** ({all_trades} trades): {all_wr:.0f}% WR | {all_ret:.1f}% Return | DD {all_dd:.1f}%\n"
            f"📈 **Recent 2 Days** ({recent_trades} trades): {recent_wr:.0f}% WR | {recent_ret:.1f}% Return | DD {recent_dd:.1f}%\n"
        )
    elif cumulative_stats:
        stats_section = (
            f"---\n"
            f"**Cumulative Performance:**\n"
            f"Trades: {cumulative_stats.get('total_trades', 0)} "
            f"({cumulative_stats.get('winners', 0)}W / {cumulative_stats.get('losers', 0)}L)\n"
            f"Win Rate: {cumulative_stats.get('win_rate', 0):.0f}%\n"
            f"Total P&L: {cumulative_stats.get('total_pnl_pct', 0):+.2f}% "
            f"(${cumulative_stats.get('total_pnl_dollars', 0):+,.0f})\n"
        )

    message = (
        f"@here\n"
        f"🔴 **[{strategy_name}] EXIT LONG** {pnl_emoji}\n"
        f"---\n"
        f"**Exit Reason:** {exit_reason}\n"
        f"**Entry:** ${entry_price:,.2f} @ {entry_time}\n"
        f"**Exit:** ${exit_price:,.2f} @ {exit_time}\n"
        f"**P&L:** {pnl_pct:+.2f}% (${pnl_dollars:+,.2f})\n"
        f"**Position:** Flat\n"
        f"{stats_section}"
    )

    return send_discord_alert(webhook_url, message, chart=chart, csv_buf=csv_buf)


def send_status_update(
    webhook_url: str,
    strategy_name: str,
    ticker: str,
    current_price: float,
    position: Optional[Dict],
    stats: Dict,
    title: str = "Status Update",
    chart: io.BytesIO = None
) -> bool:
    """
    Send periodic status update.

    Args:
        webhook_url: Discord webhook URL
        strategy_name: Strategy label
        ticker: Trading symbol
        current_price: Current price
        position: Current position dict or None
        stats: Strategy statistics
        title: Update title
        chart: Optional chart image

    Returns:
        True if sent successfully
    """
    # Position section
    if position:
        entry_price = position.get('entry_price', 0)
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        position_section = (
            f"**Position:** LONG @ ${entry_price:,.2f}\n"
            f"**Current P&L:** {pnl_pct:+.2f}%\n"
        )
    else:
        position_section = "**Position:** Flat\n"

    message = (
        f"**[{strategy_name}] {title}**\n"
        f"---\n"
        f"**{ticker}:** ${current_price:,.2f}\n"
        f"{position_section}"
        f"---\n"
        f"**Strategy Performance:**\n"
        f"Trades: {stats.get('num_trades', 0)} | "
        f"Win Rate: {stats.get('win_rate', 0):.0f}%\n"
        f"Total Return: {stats.get('total_return', 0):.1f}% | "
        f"PF: {stats.get('profit_factor', 0):.1f}\n"
    )

    return send_discord_alert(webhook_url, message, chart=chart, mention=False)


def send_error_alert(
    webhook_url: str,
    strategy_name: str,
    error_message: str,
    error_type: str = "Error"
) -> bool:
    """
    Send error notification.

    Args:
        webhook_url: Discord webhook URL
        strategy_name: Strategy label
        error_message: Error description
        error_type: Type of error

    Returns:
        True if sent successfully
    """
    message = (
        f"**[{strategy_name}] {error_type}**\n"
        f"---\n"
        f"{error_message[:1000]}\n"
        f"---\n"
        f"_Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
    )

    return send_discord_alert(webhook_url, message, mention=False)


def send_startup_notification(
    webhook_url: str,
    strategy_name: str,
    ticker: str,
    interval: str,
    position: Optional[Dict],
    stats: Dict,
    chart: io.BytesIO = None,
    enhanced_stats: Dict = None,
    csv_buf: io.BytesIO = None
) -> bool:
    """
    Send startup notification when trader starts.

    Args:
        webhook_url: Discord webhook URL
        strategy_name: Strategy label
        ticker: Trading symbol
        interval: Bar interval
        position: Current position or None
        stats: Strategy statistics
        chart: Optional chart image
        enhanced_stats: Optional enhanced stats from get_enhanced_stats()
        csv_buf: Optional trade log CSV as BytesIO

    Returns:
        True if sent successfully
    """
    # Use enhanced format if available
    if enhanced_stats:
        return send_enhanced_startup_notification(
            webhook_url=webhook_url,
            enhanced_stats=enhanced_stats,
            interval=interval,
            chart=chart,
            csv_buf=csv_buf
        )

    # Legacy format (fallback)
    if position:
        pos_status = f"LONG @ ${position.get('entry_price', 0):,.2f}"
    else:
        pos_status = "Flat"

    message = (
        f"**[{strategy_name}] Trader Started**\n"
        f"---\n"
        f"**Instrument:** {ticker} ({interval})\n"
        f"**Position:** {pos_status}\n"
        f"---\n"
        f"**Historical Performance:**\n"
        f"Trades: {stats.get('num_trades', 0)} | "
        f"Win Rate: {stats.get('win_rate', 0):.0f}%\n"
        f"Total Return: {stats.get('total_return', 0):.1f}% | "
        f"PF: {stats.get('profit_factor', 0):.1f}\n"
        f"---\n"
        f"_Monitoring active. Signals will be posted here._"
    )

    return send_discord_alert(webhook_url, message, chart=chart, mention=False)


def send_enhanced_startup_notification(
    webhook_url: str,
    enhanced_stats: Dict,
    interval: str,
    chart: io.BytesIO = None,
    csv_buf: io.BytesIO = None
) -> bool:
    """
    Send startup notification with enhanced stats format.

    Format:
    Strategy: velocity_ES=F_any_reversal_sl.72
    Ticker: ES=F
    Current Price: $6,933.75
    Signal Type: any_reversal
    Risk: SL=0.7% ($6,883.31), TP=8.9% ($7,551.65)
    ---
    📊 Tracked Stats (458 trades):
    • Win Rate: 82% | Return: 64.0%
    • Profit Factor: 18.2
    📈 Recent 2 Days (30 trades):
    • Win Rate: 87% | Return: 5.3%
    • Profit Factor: 69.1
    ---
    ⚪ Position: None
    """
    strategy_name = enhanced_stats.get('strategy_name', 'Unknown')
    ticker = enhanced_stats.get('ticker', '?')
    signal_type = enhanced_stats.get('signal_type', '?')
    current_price = enhanced_stats.get('current_price')

    sl_pct = enhanced_stats.get('sl_pct', 0)
    tp_pct = enhanced_stats.get('tp_pct', 0)
    sl_price = enhanced_stats.get('sl_price')
    tp_price = enhanced_stats.get('tp_price')

    all_time = enhanced_stats.get('all_time', {})
    recent = enhanced_stats.get('recent_2day', {})
    position = enhanced_stats.get('position')

    # Build price string
    price_str = f"${current_price:,.2f}" if current_price else "N/A"

    # Build risk string
    if sl_price and tp_price:
        risk_str = f"SL={sl_pct:.1f}% (${sl_price:,.2f}), TP={tp_pct:.1f}% (${tp_price:,.2f})"
    else:
        risk_str = f"SL={sl_pct:.1f}%, TP={tp_pct:.1f}%"

    # Build position string
    if position and position.get('is_long'):
        entry_price = position.get('entry_price', 0)
        pos_str = f"🟢 Position: LONG @ ${entry_price:,.2f}"
    else:
        pos_str = "⚪ Position: None"

    # Build all-time stats
    all_trades = all_time.get('num_trades', 0)
    all_wr = all_time.get('win_rate', 0)
    all_ret = all_time.get('total_return', 0)
    all_pf = all_time.get('profit_factor', 0)
    all_dd = all_time.get('max_drawdown', 0)

    # Build recent stats
    recent_trades = recent.get('num_trades', 0)
    recent_wr = recent.get('win_rate', 0)
    recent_ret = recent.get('total_return', 0)
    recent_pf = recent.get('profit_factor', 0)
    recent_dd = recent.get('max_drawdown', 0)

    # Add gap risk note for daily strategies
    is_daily = interval in ['1d', 'd', 'daily']
    dd_note = " _(includes gap risk)_" if is_daily and all_dd > sl_pct else ""

    message = (
        f"**[{strategy_name}] Trader Started**\n"
        f"---\n"
        f"**Strategy:** {strategy_name}\n"
        f"**Ticker:** {ticker} ({interval})\n"
        f"**Current Price:** {price_str}\n"
        f"**Signal Type:** {signal_type}\n"
        f"**Risk:** {risk_str}\n"
        f"---\n"
        f"📊 **Tracked Stats** ({all_trades} trades):\n"
        f"• Win Rate: {all_wr:.0f}% | Return: {all_ret:.1f}% | Max DD: {all_dd:.1f}%{dd_note}\n"
        f"• Profit Factor: {all_pf:.1f}\n"
        f"📈 **Recent 2 Days** ({recent_trades} trades):\n"
        f"• Win Rate: {recent_wr:.0f}% | Return: {recent_ret:.1f}% | Max DD: {recent_dd:.1f}%\n"
        f"• Profit Factor: {recent_pf:.1f}\n"
        f"---\n"
        f"{pos_str}\n"
        f"---\n"
        f"_Monitoring active. Signals will be posted here._"
    )

    return send_discord_alert(webhook_url, message, chart=chart, csv_buf=csv_buf, mention=False)


def send_regime_change_alert(
    webhook_url: str,
    strategy_name: str,
    old_regime: str,
    new_regime: str,
    adx: float,
    adx_threshold: float,
    plus_di: float,
    minus_di: float,
    is_tradeable: bool,
    is_high_vol: bool = False,
    detected_at: str = None,
) -> bool:
    """
    Send regime change notification to Discord.

    Posted when the market regime transitions (e.g., uptrend -> chop).

    Args:
        webhook_url: Discord webhook URL
        strategy_name: Strategy label
        old_regime: Previous regime name (e.g., 'uptrend')
        new_regime: Current regime name (e.g., 'chop')
        adx: Current ADX value
        adx_threshold: ADX threshold used for classification
        plus_di: Current +DI value
        minus_di: Current -DI value
        is_tradeable: Whether the new regime is tradeable
        is_high_vol: Whether currently in high-volatility state
        detected_at: Timestamp string for when change was detected

    Returns:
        True if sent successfully
    """
    if not detected_at:
        detected_at = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

    if is_tradeable:
        trading_status = "ACTIVE -- accepting signals"
    else:
        trading_status = f"PAUSED -- {new_regime} regime (not tradeable)"

    hv_note = " | High Volatility: Yes" if is_high_vol else ""

    message = (
        f"**[{strategy_name}] Regime Change**\n"
        f"---\n"
        f"**{old_regime.upper()}** --> **{new_regime.upper()}**\n"
        f"ADX: {adx:.1f} (threshold: {adx_threshold:.0f}) | +DI: {plus_di:.1f} | -DI: {minus_di:.1f}{hv_note}\n"
        f"Trading {trading_status}\n"
        f"---\n"
        f"_Detected at {detected_at}_"
    )

    return send_discord_alert(webhook_url, message, mention=False)


def send_regime_status_update(
    webhook_url: str,
    strategy_name: str,
    current_regime: str,
    adx: float,
    plus_di: float,
    minus_di: float,
    is_high_vol: bool,
    trading_status: str,
    position_str: str,
    regime_distribution: Dict,
    interval_hours: int = 4,
) -> bool:
    """
    Send periodic regime status update to Discord.

    Posted every N hours to keep subscribers informed of market conditions.

    Args:
        webhook_url: Discord webhook URL
        strategy_name: Strategy label
        current_regime: Current regime name
        adx: Current ADX value
        plus_di: Current +DI value
        minus_di: Current -DI value
        is_high_vol: Whether in high-volatility state
        trading_status: Human-readable trading status (e.g., "ACTIVE -- accepting signals")
        position_str: Position description (e.g., "LONG @ $6,491.75 (+2.1%)")
        regime_distribution: Dict mapping regime name to {'count': int, 'pct': float}
        interval_hours: Hours between updates (for "next update" note)

    Returns:
        True if sent successfully
    """
    # ADX strength description
    if adx > 40:
        adx_desc = "Very strong trend"
    elif adx > 25:
        adx_desc = "Strong trend"
    elif adx > 20:
        adx_desc = "Moderate trend"
    else:
        adx_desc = "Weak/no trend"

    hv_str = "Yes" if is_high_vol else "No"

    # Build regime distribution section
    dist_lines = []
    for name in ['uptrend', 'downtrend', 'chop']:
        info = regime_distribution.get(name, {})
        count = info.get('count', 0)
        pct = info.get('pct', 0)
        dist_lines.append(f"  {name.capitalize()}: {pct:.0f}% ({count} bars)")
    dist_section = "\n".join(dist_lines)

    message = (
        f"**[{strategy_name}] Market Status**\n"
        f"---\n"
        f"**Current Regime:** {current_regime.upper()}\n"
        f"ADX: {adx:.1f} (+DI: {plus_di:.1f}, -DI: {minus_di:.1f}) -- {adx_desc}\n"
        f"High Volatility: {hv_str}\n"
        f"Trading Status: {trading_status}\n"
        f"Position: {position_str}\n"
        f"---\n"
        f"**Regime Distribution (24h):**\n"
        f"{dist_section}\n"
        f"---\n"
        f"_Next update in ~{interval_hours} hours_"
    )

    return send_discord_alert(webhook_url, message, mention=False)


def create_trade_log_csv_buffer(csv_content: str) -> io.BytesIO:
    """
    Create a BytesIO buffer from CSV string for Discord attachment.

    Args:
        csv_content: CSV string from get_trade_log_csv()

    Returns:
        BytesIO buffer ready for Discord upload
    """
    buf = io.BytesIO()
    buf.write(csv_content.encode('utf-8'))
    buf.seek(0)
    return buf
