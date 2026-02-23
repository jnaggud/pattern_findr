"""
Unified Backtest Engine for Velocity Trading.

Single source of truth for trade simulation. Used by:
- Walk-forward optimization (walkforward_v8_production.py)
- Production rebuild (position_manager.py) — future migration
- Streamlit charts (velocity_live_trader.py) — future migration

Design:
- prepare_backtest_arrays(): DataFrame + config -> numpy arrays
- run_backtest(): numpy arrays + params -> stats dict or trade list
- Pure numpy in hot loop for Optuna speed (10K trials, 32 workers)
- Optional callables for novel filter entry rejection and ML exit
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable, Tuple, List


def prepare_backtest_arrays(
    df: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Convert DataFrame + config into numpy arrays for run_backtest().

    Expects df to already have oscillator calculated (via calculate_composite_oscillator).
    Handles: smoothing, wavelet denoising, velocity/accel/jerk, signal generation.

    Args:
        df: DataFrame with OHLCV + oscillator columns (osc_smooth or composite_smooth)
        config: Strategy config dict

    Returns:
        Dict of numpy arrays ready for run_backtest()
    """
    result = df.copy()

    # --- Oscillator smoothing ---
    vel_smoothing = config.get('vel_smoothing', 1)
    osc_col = 'composite_smooth' if 'composite_smooth' in result.columns else 'osc_smooth'
    if osc_col not in result.columns:
        osc_col = 'composite_oscillator' if 'composite_oscillator' in result.columns else None
    if osc_col is None:
        raise ValueError("No oscillator column found in DataFrame")

    if vel_smoothing > 1:
        result['osc_smooth'] = result[osc_col].rolling(window=vel_smoothing, center=False).mean()
        result['osc_smooth'] = result['osc_smooth'].bfill()
    else:
        result['osc_smooth'] = result[osc_col]

    # --- Wavelet denoising ---
    if config.get('use_wavelet_denoise', False):
        try:
            import pywt
            raw = result['osc_smooth'].values.copy()
            family = config.get('wavelet_family', 'db4')
            level = config.get('wavelet_level', 2)
            mode = config.get('wavelet_threshold_mode', 'hard')
            coeffs = pywt.wavedec(raw, family, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(raw)))
            denoised = [coeffs[0]] + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
            osc_denoised = pywt.waverec(denoised, family)[:len(raw)]
            result['osc_smooth'] = pd.Series(osc_denoised, index=result.index)
        except Exception:
            pass  # Denoising failed, use original

    # --- Velocity, acceleration, jerk ---
    osc_smooth = result['osc_smooth'].values
    velocity = np.zeros(len(osc_smooth))
    velocity[1:] = np.diff(osc_smooth)

    acceleration = np.zeros(len(velocity))
    acceleration[1:] = np.diff(velocity)

    jerk = np.zeros(len(acceleration))
    jerk[1:] = np.diff(acceleration)

    # --- Signal generation (matches velocity_trading/indicators/velocity.py) ---
    signal_type = config.get('signal_type', 'any_reversal')
    oversold = config.get('oversold_threshold', -0.3)
    overbought = config.get('overbought_threshold', 0.3)
    extreme_mult = config.get('extreme_zone_mult', 1.5)

    in_oversold = osc_smooth < oversold
    in_overbought = osc_smooth > overbought
    extreme_oversold = osc_smooth < (oversold * extreme_mult)
    extreme_overbought = osc_smooth > (overbought * extreme_mult)

    # Velocity zero-crossings
    vel_cross_up = np.zeros(len(velocity), dtype=bool)
    vel_cross_up[1:] = (velocity[1:] > 0) & (velocity[:-1] <= 0)
    vel_cross_down = np.zeros(len(velocity), dtype=bool)
    vel_cross_down[1:] = (velocity[1:] < 0) & (velocity[:-1] >= 0)

    # Strong momentum (rolling std of velocity)
    vel_series = pd.Series(velocity)
    vel_std = vel_series.rolling(10, min_periods=1).std().fillna(vel_series.std()).values
    strong_momentum_up = velocity > vel_std * 1.5
    strong_momentum_down = velocity < -vel_std * 1.5

    # Build signals based on signal_type
    if signal_type == 'velocity_crossover_and_zone':
        buy_signals = vel_cross_up & in_oversold
        sell_signals = vel_cross_down & in_overbought
    elif signal_type == 'velocity_crossover_or_zone':
        buy_signals = vel_cross_up | extreme_oversold
        sell_signals = vel_cross_down | extreme_overbought
    elif signal_type == 'zone_only':
        buy_signals = extreme_oversold & (velocity > 0)
        sell_signals = extreme_overbought & (velocity < 0)
    elif signal_type == 'momentum':
        buy_signals = strong_momentum_up & (osc_smooth < 0)
        sell_signals = strong_momentum_down & (osc_smooth > 0)
    elif signal_type == 'any_reversal':
        buy_signals = vel_cross_up | extreme_oversold | (strong_momentum_up & in_oversold)
        sell_signals = vel_cross_down | extreme_overbought | (strong_momentum_down & in_overbought)
    elif signal_type == 'double_bottom':
        vel_cross_up_count = pd.Series(vel_cross_up.astype(float)).rolling(10).sum().values
        buy_signals = (vel_cross_up_count >= 2) & in_oversold
        vel_cross_down_count = pd.Series(vel_cross_down.astype(float)).rolling(10).sum().values
        sell_signals = (vel_cross_down_count >= 2) & in_overbought
    elif signal_type == 'breakout':
        osc_prev = np.zeros_like(osc_smooth)
        osc_prev[1:] = osc_smooth[:-1]
        buy_signals = (osc_smooth > oversold) & (osc_prev <= oversold)
        sell_signals = (osc_smooth < overbought) & (osc_prev >= overbought)
    elif signal_type == 'divergence':
        close_vals = result['close'].values if 'close' in result.columns else result['Close'].values
        close_series = pd.Series(close_vals)
        osc_series = pd.Series(osc_smooth)
        price_lower_low = close_vals < close_series.rolling(5).min().shift(1).values
        osc_higher_low = osc_smooth > osc_series.rolling(5).min().shift(1).values
        buy_signals = price_lower_low & osc_higher_low & in_oversold
        price_higher_high = close_vals > close_series.rolling(5).max().shift(1).values
        osc_lower_high = osc_smooth < osc_series.rolling(5).max().shift(1).values
        sell_signals = price_higher_high & osc_lower_high & in_overbought
    else:
        # Default: velocity_crossover_and_zone
        buy_signals = vel_cross_up & in_oversold
        sell_signals = vel_cross_down & in_overbought

    # --- Apply require_accel filter ---
    if config.get('require_accel', False):
        accel_threshold = config.get('accel_threshold', 0.0)
        buy_accel = acceleration > 0
        sell_accel = acceleration < 0
        if accel_threshold > 0:
            buy_accel = buy_accel & (np.abs(acceleration) >= accel_threshold)
            sell_accel = sell_accel & (np.abs(acceleration) >= accel_threshold)
        buy_signals = buy_signals & buy_accel
        sell_signals = sell_signals & sell_accel

    # --- Apply velocity magnitude filter ---
    vel_threshold = config.get('vel_threshold', 0.0)
    if vel_threshold > 0:
        vel_mag = np.abs(velocity) >= vel_threshold
        buy_signals = buy_signals & vel_mag
        sell_signals = sell_signals & vel_mag

    # --- RSI filter ---
    rsi_filter = config.get('rsi_filter', 'none')
    if rsi_filter != 'none' and 'RSI' in result.columns:
        rsi = result['RSI'].values
        rsi_os = config.get('rsi_oversold', 30)
        rsi_ob = config.get('rsi_overbought', 70)
        if rsi_filter == 'oversold_only':
            buy_signals = buy_signals & (rsi < rsi_os)
        elif rsi_filter == 'overbought_only':
            sell_signals = sell_signals & (rsi > rsi_ob)
        elif rsi_filter == 'both':
            buy_signals = buy_signals & (rsi < rsi_os)
            sell_signals = sell_signals & (rsi > rsi_ob)

    # --- MACD confirmation filter ---
    if config.get('use_macd_confirm', False) and 'MACD_histogram' in result.columns:
        macd_h = result['MACD_histogram'].values
        macd_improving = macd_h > np.roll(macd_h, 1)
        macd_declining = macd_h < np.roll(macd_h, 1)
        buy_signals = buy_signals & macd_improving
        sell_signals = sell_signals & macd_declining

    # --- Bollinger Band filter ---
    if config.get('use_bb_filter', False) and 'BB_lower' in result.columns:
        _close_col = 'close' if 'close' in result.columns else 'Close'
        close_vals = result[_close_col].values
        buy_signals = buy_signals & (close_vals < result['BB_lower'].values)
        sell_signals = sell_signals & (close_vals > result['BB_upper'].values)

    # --- V2 indicator filters (Regime, Fragility, Entropy) ---
    if config.get('use_regime_filter', False) and 'RSC' in result.columns:
        regime_thresh = config.get('regime_threshold', 0.0)
        buy_signals = buy_signals & (result['RSC'].values > regime_thresh)

    if config.get('use_fragility_filter', False) and 'MFI2' in result.columns:
        frag_thresh = config.get('fragility_threshold', 0.5)
        buy_signals = buy_signals & (result['MFI2'].values < frag_thresh)

    if config.get('use_entropy_filter', False) and 'SEI' in result.columns:
        entropy_thresh = config.get('entropy_threshold', 0.7)
        buy_signals = buy_signals & (result['SEI'].values < entropy_thresh)

    # --- Volatility regime filter (ATR percentile) ---
    if config.get('use_vol_regime_filter', False) and 'VOL_REGIME' in result.columns:
        vol_thresh = config.get('vol_regime_percentile_threshold', 0.25)
        buy_signals = buy_signals & (result['VOL_REGIME'].values > vol_thresh)

    # --- Money Flow Velocity filter ---
    if config.get('use_mfv_filter', False):
        mfv_mode = config.get('mfv_mode', 'velocity')
        mfv_thresh = config.get('mfv_threshold', 0.0)

        if mfv_mode == 'velocity' and 'MFV_VEL' in result.columns:
            mfv_vel = result['MFV_VEL'].values
            buy_signals = buy_signals & (mfv_vel > mfv_thresh)
            sell_signals = sell_signals & (mfv_vel < -mfv_thresh)
        elif mfv_mode == 'flow' and 'MFV_FLOW' in result.columns:
            mfv_flow = result['MFV_FLOW'].values
            buy_signals = buy_signals & (mfv_flow > mfv_thresh)
            sell_signals = sell_signals & (mfv_flow < -mfv_thresh)
        elif mfv_mode == 'both' and 'MFV_FLOW' in result.columns and 'MFV_VEL' in result.columns:
            mfv_flow = result['MFV_FLOW'].values
            mfv_vel = result['MFV_VEL'].values
            buy_signals = buy_signals & (mfv_flow > mfv_thresh) & (mfv_vel > mfv_thresh)
            sell_signals = sell_signals & (mfv_flow < -mfv_thresh) & (mfv_vel < -mfv_thresh)

    # --- Options Influence Zone filter ---
    if config.get('use_options_zone_filter', False):
        oz_mode = config.get('options_zone_mode', 'gamma')
        oz_thresh = config.get('options_zone_threshold', 0.0)
        zone_col_map = {
            'gamma': 'OPTIONS_GAMMA_ZONE',
            'max_pain': 'OPTIONS_MP_ZONE',
            'wall': 'OPTIONS_WALL_ZONE',
            'combined': 'OPTIONS_COMBINED_ZONE',
        }
        zone_col = zone_col_map.get(oz_mode, 'OPTIONS_GAMMA_ZONE')
        if zone_col in result.columns:
            zone = result[zone_col].values
            if oz_mode == 'gamma':
                buy_signals = buy_signals & (zone > oz_thresh)
                sell_signals = sell_signals & (zone > oz_thresh)
            elif oz_mode in ('max_pain', 'wall'):
                buy_signals = buy_signals & (zone < -oz_thresh)
                sell_signals = sell_signals & (zone > oz_thresh)
            elif oz_mode == 'combined':
                buy_signals = buy_signals & (zone > oz_thresh)
                sell_signals = sell_signals & (zone < -oz_thresh)

    # --- KNN Pattern Matcher filter ---
    if config.get('use_knn_filter', False):
        knn_h = config.get('knn_horizon', 8)
        knn_pt = config.get('knn_prob_threshold', 0.55)
        knn_ct = config.get('knn_confidence_threshold', 0.1)
        prob_col = f'knn_prob_up_{knn_h}'
        conf_col = f'knn_confidence_{knn_h}'
        if prob_col in result.columns and conf_col in result.columns:
            knn_prob = result[prob_col].values
            knn_conf = result[conf_col].values
            buy_signals = buy_signals & (knn_prob > knn_pt) & (knn_conf > knn_ct)
            sell_signals = sell_signals & (knn_prob < (1 - knn_pt)) & (knn_conf > knn_ct)

    # Get close/high/low arrays
    close_col = 'close' if 'close' in result.columns else 'Close'
    high_col = 'high' if 'high' in result.columns else 'High'
    low_col = 'low' if 'low' in result.columns else 'Low'

    return {
        'close': result[close_col].values.astype(float),
        'high': result[high_col].values.astype(float),
        'low': result[low_col].values.astype(float),
        'buy_signals': buy_signals.astype(bool),
        'sell_signals': sell_signals.astype(bool),
        'osc_smooth': osc_smooth.astype(float),
        'velocity': velocity.astype(float),
        'acceleration': acceleration.astype(float),
        'jerk': jerk.astype(float),
        'index': result.index,
    }


def run_backtest(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    buy_signals: np.ndarray,
    sell_signals: np.ndarray,
    osc_smooth: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
    # Trade parameters
    stop_loss_pct: float = 5.0,
    take_profit_pct: float = 10.0,
    min_hold_bars: int = 1,
    min_bars_between: int = 1,
    # Exit toggles
    exit_on_opposite: bool = True,
    exit_on_midline: bool = False,
    # Trailing stop
    use_trailing_stop: bool = False,
    trailing_stop_pct: float = 1.0,
    trailing_stop_activation_pct: float = 0.3,
    # Break-even stop
    use_breakeven_stop: bool = False,
    breakeven_trigger_pct: float = 0.3,
    breakeven_offset_pct: float = 0.05,
    # Accel exit
    use_accel_exit: bool = False,
    accel_exit_type: str = 'sign_reversal',
    accel_exit_threshold: float = 0.0,
    accel_exit_min_pnl: float = 0.5,
    accel_exit_lookback: int = 1,
    use_jerk_confirm: bool = False,
    jerk_confirm_threshold: float = 0.0,
    # Optional callables
    entry_filter_fn: Optional[Callable[[int], bool]] = None,
    exit_model_fn: Optional[Callable] = None,
    # Mode
    return_trades: bool = False,
    # Unused (accepted so **arrays works cleanly)
    velocity: np.ndarray = None,
    index: object = None,
) -> dict:
    """
    Unified trade simulation engine. Pure numpy for speed.

    Args:
        close/high/low: Price arrays
        buy_signals/sell_signals: Boolean signal arrays
        osc_smooth: Oscillator values (for midline cross)
        acceleration: Acceleration values (for accel exit)
        jerk: Jerk values (for jerk confirmation)
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        min_hold_bars: Minimum bars before signal-based exits
        min_bars_between: Minimum bars between trades
        exit_on_opposite: Exit on opposite signal
        exit_on_midline: Exit on midline cross (osc > 0)
        use_trailing_stop: Enable trailing stop from high watermark
        trailing_stop_pct: Trail distance from high watermark (%)
        trailing_stop_activation_pct: Min profit to activate trailing stop (%)
        use_breakeven_stop: Enable break-even stop
        breakeven_trigger_pct: Min profit to move SL to breakeven (%)
        breakeven_offset_pct: Buffer above entry price for breakeven (%)
        use_accel_exit: Enable acceleration exit
        accel_exit_*: Accel exit parameters
        use_jerk_confirm: Require jerk confirmation for accel exit
        jerk_confirm_threshold: Jerk threshold
        entry_filter_fn: Optional callable(bar_idx) -> bool for novel filter entry rejection
        exit_model_fn: Optional callable(entry_price, entry_bar, current_bar, hwm) -> (bool, float, str)
        return_trades: If True, return full trade list. If False, return stats only.

    Returns:
        Dict with stats (and optionally trade list)
    """
    n = len(close)
    if n == 0:
        return _empty_result(return_trades)

    # State
    in_position = False
    entry_price = 0.0
    entry_bar = 0
    high_watermark = 0.0
    last_exit_bar = -min_bars_between  # Allow immediate first trade

    # Tracking
    trades = [] if return_trades else None
    pnls = []
    maes = []
    mfes = []
    min_price_in_trade = 0.0
    max_price_in_trade = 0.0

    # Equity tracking for drawdown
    equity = 1.0
    peak_equity = 1.0
    max_drawdown = 0.0

    for i in range(n):
        if not in_position:
            # --- ENTRY ---
            if (buy_signals[i]
                    and (i - last_exit_bar) >= min_bars_between):
                # Optional novel filter check
                if entry_filter_fn is not None:
                    if not entry_filter_fn(i):
                        continue  # Rejected by filter

                in_position = True
                entry_price = close[i]
                entry_bar = i
                high_watermark = high[i]
                min_price_in_trade = low[i]
                max_price_in_trade = high[i]
        else:
            # --- IN POSITION: track MAE/MFE ---
            min_price_in_trade = min(min_price_in_trade, low[i])
            max_price_in_trade = max(max_price_in_trade, high[i])
            high_watermark = max(high_watermark, high[i])

            bars_held = i - entry_bar
            close_pnl = (close[i] - entry_price) / entry_price * 100
            low_pnl = (low[i] - entry_price) / entry_price * 100
            high_pnl = (high[i] - entry_price) / entry_price * 100

            exit_reason = None
            exit_price = close[i]

            # Priority 1: Stop Loss (ALWAYS fires, ignores min_hold_bars)
            if low_pnl <= -stop_loss_pct:
                exit_reason = 'Stop Loss'
                exit_price = entry_price * (1 - stop_loss_pct / 100)

            # Priority 2: Take Profit (ALWAYS fires, ignores min_hold_bars)
            elif high_pnl >= take_profit_pct:
                exit_reason = 'Take Profit'
                exit_price = entry_price * (1 + take_profit_pct / 100)

            # Priority 3: Trailing Stop (ALWAYS fires once activated, ignores min_hold_bars)
            if not exit_reason and use_trailing_stop:
                hwm_pnl = (high_watermark - entry_price) / entry_price * 100
                if hwm_pnl >= trailing_stop_activation_pct:
                    trail_level = high_watermark * (1 - trailing_stop_pct / 100)
                    if low[i] <= trail_level:
                        exit_reason = 'Trailing Stop'
                        exit_price = trail_level

            # Priority 4: Break-Even Stop (ALWAYS fires once activated, ignores min_hold_bars)
            if not exit_reason and use_breakeven_stop:
                # Check if trade ever reached the trigger profit level
                hwm_pnl = (high_watermark - entry_price) / entry_price * 100
                if hwm_pnl >= breakeven_trigger_pct:
                    # SL is now at entry + small offset
                    be_price = entry_price * (1 + breakeven_offset_pct / 100)
                    # Price must have actually reached be_price before we can exit there
                    if high_watermark >= be_price and low[i] <= be_price:
                        exit_reason = 'Break-Even Stop'
                        exit_price = be_price

            # All remaining exits require min_hold_bars
            if not exit_reason and bars_held >= min_hold_bars:

                # Priority 5: Accel Exit
                if (not exit_reason and use_accel_exit
                        and i >= accel_exit_lookback):
                    pnl_ok = close_pnl >= accel_exit_min_pnl or close_pnl < 0
                    if pnl_ok:
                        accel_cond = False
                        if accel_exit_type == 'sign_reversal':
                            accel_vals = acceleration[i - accel_exit_lookback + 1:i + 1]
                            accel_cond = np.all(accel_vals < 0)
                        elif accel_exit_type == 'magnitude':
                            accel_cond = acceleration[i] < -accel_exit_threshold
                        elif accel_exit_type == 'both':
                            accel_vals = acceleration[i - accel_exit_lookback + 1:i + 1]
                            accel_cond = np.all(accel_vals < 0) and abs(acceleration[i]) > accel_exit_threshold

                        jerk_ok = True
                        if use_jerk_confirm and jerk_confirm_threshold > 0:
                            jerk_ok = jerk[i] < -jerk_confirm_threshold

                        if accel_cond and jerk_ok:
                            exit_reason = 'Accel Exit'

                # Priority 6: ML Exit (via callable)
                if not exit_reason and exit_model_fn is not None:
                    try:
                        should_exit, prob, reason = exit_model_fn(
                            entry_price, entry_bar, i, high_watermark
                        )
                        if should_exit:
                            exit_reason = f'ML Exit ({close_pnl:.2f}%, p={prob:.2f})'
                    except Exception:
                        pass

                # Priority 7: Midline Cross
                if not exit_reason and exit_on_midline:
                    if osc_smooth[i] > 0:
                        exit_reason = 'Midline Cross'

                # Priority 8: Opposite Signal
                if not exit_reason and exit_on_opposite and sell_signals[i]:
                    exit_reason = 'Opposite Signal'

            # --- EXECUTE EXIT ---
            if exit_reason:
                pnl = (exit_price - entry_price) / entry_price * 100
                mae = (entry_price - min_price_in_trade) / entry_price * 100
                mfe = (max_price_in_trade - entry_price) / entry_price * 100

                pnls.append(pnl)
                maes.append(mae)
                mfes.append(mfe)

                # Update equity and drawdown
                equity *= (1 + pnl / 100)
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, dd)

                if return_trades:
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'mae': mae,
                        'mfe': mfe,
                        'bars_held': bars_held,
                    })

                in_position = False
                last_exit_bar = i

    # --- Build result ---
    n_trades = len(pnls)
    if n_trades == 0:
        return _empty_result(return_trades)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001

    total_return = (equity - 1) * 100

    result = {
        'total_return': total_return,
        'n_trades': n_trades,
        'win_rate': len(wins) / n_trades * 100,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'max_drawdown': max_drawdown,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else (999.99 if gross_profit > 0 else 0),
        'avg_mae': np.mean(maes) if maes else 0,
        'max_mae': max(maes) if maes else 0,
        'avg_mfe': np.mean(mfes) if mfes else 0,
    }

    if return_trades:
        result['trades'] = trades
        # Open position info
        if in_position:
            result['open_position'] = {
                'entry_bar': entry_bar,
                'entry_price': entry_price,
            }
        else:
            result['open_position'] = None

    return result


def _empty_result(return_trades: bool) -> dict:
    """Return empty result when no trades."""
    result = {
        'total_return': 0.0,
        'n_trades': 0,
        'win_rate': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'max_drawdown': 0.0,
        'profit_factor': 0.0,
        'avg_mae': 0.0,
        'max_mae': 0.0,
        'avg_mfe': 0.0,
    }
    if return_trades:
        result['trades'] = []
        result['open_position'] = None
    return result
