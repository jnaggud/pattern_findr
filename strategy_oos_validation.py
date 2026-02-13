#!/usr/bin/env python3
"""
Out-of-Sample Validation for Optimal Strategy Combination [14, 4, 12]

Three-way split:
  - Train (60%): Optuna optimization runs here
  - Validation (20%): Used to confirm strategy selection (already done in search)
  - Holdout (20%): TRUE out-of-sample — no decisions ever used this data

The holdout set is the most recent data, ensuring temporal validity.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import optuna
from datetime import datetime, timedelta
from copy import deepcopy

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ['QUIET_WORKERS'] = '1'

import builtins
_original_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = _flush_print

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
from novel_indicators import (
    calculate_arwo, calculate_dco, calculate_vcmo, calculate_ics,
    calculate_mji, calculate_prf, calculate_ewaf, calculate_kfif,
)
from oscillator_indicators import create_composite_oscillator_features
import optuna_worker

# Import strategy functions from the harness
from strategy_test_harness import (
    fetch_data, fetch_cross_asset_data, compute_oscillators_standard,
    precompute_indicators, run_optimization, evaluate_on_test, select_best,
    compute_gt_score, apply_kalman, apply_wavelet, compute_hmm_mask,
    compute_ou_thresholds, apply_instance_selection, build_xgboost_mask,
    apply_smote_to_xgb_mask, build_transformer_mask, apply_nolaw_filter,
    compute_adaptive_stoch_vol, train_drl_stop_loss, meta_learn_adaptation,
    compute_cross_asset_correlation_filter, compute_copula_tail_filter,
    train_vae_anomaly_mask, online_adaptive_weights, apply_diffusion_denoise,
    compute_transfer_entropy, STRATEGY_NAMES,
)


def apply_strategies_to_data(strategy_ids, osc_dict, close, high, low, vol, cross_asset_data):
    """Apply a list of strategies to oscillators/masks. Returns modified osc + mask."""
    current_osc = deepcopy(osc_dict)
    mask = np.ones(len(close), dtype=bool)
    min_mask_pct = 0.10
    use_gt_score = False
    use_drawdown_penalty = False

    def safe_apply_mask(current, new, label=''):
        candidate = current & new
        pct = candidate.sum() / len(candidate)
        if pct < min_mask_pct:
            print(f"    WARNING: {label} mask would drop to {pct:.0%} — keeping previous ({current.sum()/len(current):.0%})")
            return current
        return candidate

    for sid in strategy_ids:
        name = STRATEGY_NAMES.get(sid, f"#{sid}")

        if sid == 1:
            result, err = apply_kalman(current_osc)
            if not err:
                current_osc = result
                print(f"    #{sid} {name}: Applied Kalman to {len(result)} oscillators")
            else:
                print(f"    #{sid} {name}: SKIPPED: {err}")

        elif sid == 2:
            result, err = apply_wavelet(current_osc)
            if not err:
                current_osc = result
                print(f"    #{sid} {name}: Applied wavelet denoising")
            else:
                print(f"    #{sid} {name}: SKIPPED: {err}")

        elif sid == 3:
            hmm_mask, msg = compute_hmm_mask(close)
            if hmm_mask is not None:
                mask = safe_apply_mask(mask, hmm_mask, name)
                print(f"    #{sid} {name}: {msg}")
            else:
                print(f"    #{sid} {name}: SKIPPED: {msg}")

        elif sid == 4:
            osc_vals = current_osc.get('composite_smooth', close * 0)
            entry, exit_lvl, err = compute_ou_thresholds(osc_vals)
            if not err:
                for oname, vals in current_osc.items():
                    near_entry = np.abs(vals - entry) < 0.1
                    near_exit = np.abs(vals - exit_lvl) < 0.1
                    boost = np.ones(len(vals))
                    boost[near_entry] = 1.3
                    boost[near_exit] = 1.3
                    current_osc[oname] = np.clip(vals * boost, -1, 1)
                print(f"    #{sid} {name}: entry={entry:.4f}, exit={exit_lvl:.4f}")
            else:
                print(f"    #{sid} {name}: SKIPPED: {err}")

        elif sid == 5:
            use_gt_score = True
            print(f"    #{sid} {name}: Enabled GT-Score re-ranking")

        elif sid == 6:
            filtered, msg = apply_instance_selection(close, current_osc)
            current_osc = filtered
            print(f"    #{sid} {name}: {msg}")

        elif sid == 7:
            osc_vals = current_osc.get('mji', current_osc.get('composite_smooth', close * 0))
            xgb_mask, model, msg = build_xgboost_mask(close, osc_vals, current_osc, vol, high, low)
            if xgb_mask is not None:
                mask = safe_apply_mask(mask, xgb_mask, name)
                print(f"    #{sid} {name}: {msg}")
            else:
                print(f"    #{sid} {name}: SKIPPED: {msg}")

        elif sid == 8:
            print(f"    #{sid} {name}: Applied (no-op in isolation)")

        elif sid == 9:
            asv = compute_adaptive_stoch_vol(close)
            for oname, vals in current_osc.items():
                vol_scale = 1.0 / (1.0 + asv * 10)
                current_osc[oname] = np.clip(vals * vol_scale, -1, 1)
            print(f"    #{sid} {name}: mean vol={np.mean(asv):.6f}")

        elif sid == 10:
            result, err = apply_nolaw_filter(current_osc)
            if not err:
                current_osc = result
                print(f"    #{sid} {name}: Applied NoLAW to {len(result)} oscillators")
            else:
                print(f"    #{sid} {name}: SKIPPED: {err}")

        elif sid == 11:
            osc_vals = current_osc.get('composite_smooth', close * 0)
            stops, Q = train_drl_stop_loss(close, osc_vals, high, low)
            tight_mask = stops > 0.75
            mask = safe_apply_mask(mask, tight_mask, name)
            print(f"    #{sid} {name}: mean stop={np.mean(stops):.2f}%")

        elif sid == 12:
            osc_vals = current_osc.get('mji', current_osc.get('composite_smooth', close * 0))
            smote_mask, model, msg = apply_smote_to_xgb_mask(close, osc_vals, current_osc, vol, high, low)
            if smote_mask is not None:
                mask = safe_apply_mask(mask, smote_mask, name)
                print(f"    #{sid} {name}: {msg}")
            else:
                print(f"    #{sid} {name}: SKIPPED: {msg}")

        elif sid == 13:
            for oname, vals in current_osc.items():
                adapted, _ = meta_learn_adaptation(close, vals)
                current_osc[oname] = adapted
            print(f"    #{sid} {name}: Applied meta-learned adaptive smoothing")

        elif sid == 14:
            osc_vals = current_osc.get('composite_smooth', close * 0)
            try:
                tfm_mask, msg = build_transformer_mask(close, osc_vals, vol, high, low)
                if tfm_mask is not None:
                    mask = safe_apply_mask(mask, tfm_mask, name)
                    pct = tfm_mask.sum() / len(tfm_mask) * 100
                    print(f"    #{sid} {name}: {pct:.0f}% high-confidence bars")
                else:
                    print(f"    #{sid} {name}: SKIPPED: {msg}")
            except Exception as e:
                print(f"    #{sid} {name}: SKIPPED: {e}")

        elif sid == 15:
            returns = np.zeros(len(close))
            returns[1:] = np.diff(close) / close[:-1]
            ca_mask, msg = compute_cross_asset_correlation_filter(returns, cross_asset_data)
            mask = safe_apply_mask(mask, ca_mask, name)
            print(f"    #{sid} {name}: {msg}")

        elif sid == 16:
            returns = np.zeros(len(close))
            returns[1:] = np.diff(close) / close[:-1]
            vix_data = cross_asset_data.get('^VIX', None)
            cop_mask, msg = compute_copula_tail_filter(returns, vix_data)
            mask = safe_apply_mask(mask, cop_mask, name)
            print(f"    #{sid} {name}: {msg}")

        elif sid == 17:
            osc_vals = current_osc.get('composite_smooth', close * 0)
            try:
                vae_mask, msg = train_vae_anomaly_mask(close, osc_vals, vol)
                if vae_mask is not None:
                    mask = safe_apply_mask(mask, vae_mask, name)
                    print(f"    #{sid} {name}: {msg}")
                else:
                    print(f"    #{sid} {name}: SKIPPED: {msg}")
            except Exception as e:
                print(f"    #{sid} {name}: SKIPPED: {e}")

        elif sid == 18:
            use_drawdown_penalty = True
            print(f"    #{sid} {name}: Enabled drawdown penalty")

        elif sid == 19:
            for oname, vals in current_osc.items():
                adapted, _ = online_adaptive_weights(close, vals)
                current_osc[oname] = adapted
            print(f"    #{sid} {name}: Applied online adaptive weights")

        elif sid == 20:
            result, err = apply_diffusion_denoise(current_osc)
            if not err:
                current_osc = result
                print(f"    #{sid} {name}: Applied diffusion denoising")
            else:
                print(f"    #{sid} {name}: SKIPPED: {err}")

    return current_osc, mask, use_gt_score, use_drawdown_penalty


def get_masked_oscillators(osc_dict, mask):
    """Apply bar mask — zero out non-tradeable bars."""
    if mask is None or mask.all():
        return osc_dict
    filtered = {}
    for name, vals in osc_dict.items():
        modified = vals.copy()
        mask_len = min(len(mask), len(modified))
        modified[:mask_len][~mask[:mask_len]] = 0.0
        filtered[name] = modified
    return filtered


def run_split_evaluation(ticker, interval, strategy_ids, n_trials, n_workers, days=59):
    """
    Run 3-way split evaluation:
    - Train (60%): optimization
    - Validation (20%): confirms strategy quality
    - Holdout (20%): TRUE out-of-sample
    """
    print("=" * 100)
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 100)
    print(f"Ticker: {ticker}, Interval: {interval}")
    print(f"Strategies: {strategy_ids} ({', '.join(STRATEGY_NAMES.get(s, f'#{s}') for s in strategy_ids)})")
    print(f"Trials: {n_trials:,}, Workers: {n_workers}")
    print(f"Split: 60% Train / 20% Validation / 20% Holdout\n")

    # Fetch data
    print("Fetching data...")
    df = fetch_data(ticker, interval, days)
    print(f"  {len(df)} bars total")

    # 3-way split
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    holdout_df = df.iloc[val_end:].copy()

    print(f"  Train:    {len(train_df)} bars ({train_df.index[0].strftime('%Y-%m-%d %H:%M')} to {train_df.index[-1].strftime('%Y-%m-%d %H:%M')})")
    print(f"  Valid:    {len(val_df)} bars ({val_df.index[0].strftime('%Y-%m-%d %H:%M')} to {val_df.index[-1].strftime('%Y-%m-%d %H:%M')})")
    print(f"  Holdout:  {len(holdout_df)} bars ({holdout_df.index[0].strftime('%Y-%m-%d %H:%M')} to {holdout_df.index[-1].strftime('%Y-%m-%d %H:%M')})")

    bh_train = (train_df['close'].iloc[-1] / train_df['close'].iloc[0] - 1) * 100
    bh_val = (val_df['close'].iloc[-1] / val_df['close'].iloc[0] - 1) * 100
    bh_holdout = (holdout_df['close'].iloc[-1] / holdout_df['close'].iloc[0] - 1) * 100
    print(f"  Buy & Hold — Train: {bh_train:.2f}%, Val: {bh_val:.2f}%, Holdout: {bh_holdout:.2f}%")

    # Fetch cross-asset data
    print("\nFetching cross-asset data...")
    cross_asset_data = fetch_cross_asset_data(interval, days)
    print(f"  Available: {list(cross_asset_data.keys())}")

    # Compute oscillators for each split
    print("\nComputing oscillators for each split...")
    train_osc_raw = compute_oscillators_standard(train_df)
    val_osc_raw = compute_oscillators_standard(val_df)
    holdout_osc_raw = compute_oscillators_standard(holdout_df)
    print(f"  {len(train_osc_raw)} oscillator types: {list(train_osc_raw.keys())}")

    # =========================================================================
    # BASELINE: No strategies, raw optimization
    # =========================================================================
    print("\n" + "=" * 100)
    print("BASELINE (no strategies)")
    print("=" * 100)

    close_train = train_df['close'].values
    high_train = train_df['high'].values
    low_train = train_df['low'].values
    vol_train = train_df['volume'].values

    default_osc = train_osc_raw.get('composite_smooth', close_train * 0)

    print("  Running baseline optimization on train...")
    baseline_results = run_optimization(
        close_train, default_osc, train_osc_raw,
        n_trials=n_trials, n_workers=n_workers,
        optimize_metric='risk_adjusted', label="BASELINE",
        high_prices=high_train, low_prices=low_train, volume=vol_train,
    )

    baseline_top = select_best(baseline_results, top_n=5, metric='risk_adjusted')
    baseline_best = baseline_top[0] if baseline_top else None

    if baseline_best:
        bl_val_result = evaluate_on_test(baseline_best, val_df, val_osc_raw)
        bl_holdout_result = evaluate_on_test(baseline_best, holdout_df, holdout_osc_raw)

        print(f"\n  BASELINE Results:")
        print(f"    Train:   Return={baseline_best.get('total_return', 0):.2f}%, "
              f"WR={baseline_best.get('win_rate', 0):.1f}%, "
              f"Trades={baseline_best.get('num_trades', 0)}, "
              f"PF={baseline_best.get('profit_factor', 0):.2f}, "
              f"MaxDD={baseline_best.get('max_drawdown', 0):.2f}%")
        if bl_val_result:
            print(f"    Valid:   Return={bl_val_result.get('total_return', 0):.2f}%, "
                  f"WR={bl_val_result.get('win_rate', 0):.1f}%, "
                  f"Trades={bl_val_result.get('num_trades', 0)}, "
                  f"PF={bl_val_result.get('profit_factor', 0):.2f}, "
                  f"MaxDD={bl_val_result.get('max_drawdown', 0):.2f}%")
        if bl_holdout_result:
            print(f"    HOLDOUT: Return={bl_holdout_result.get('total_return', 0):.2f}%, "
                  f"WR={bl_holdout_result.get('win_rate', 0):.1f}%, "
                  f"Trades={bl_holdout_result.get('num_trades', 0)}, "
                  f"PF={bl_holdout_result.get('profit_factor', 0):.2f}, "
                  f"MaxDD={bl_holdout_result.get('max_drawdown', 0):.2f}%")
    else:
        print("  BASELINE: No valid results!")
        bl_val_result = bl_holdout_result = None

    # =========================================================================
    # STRATEGY COMBINATION: Apply [14, 4, 12]
    # =========================================================================
    print("\n" + "=" * 100)
    print(f"STRATEGY COMBINATION: {strategy_ids}")
    print("=" * 100)

    # Apply strategies to train oscillators/masks
    print("\n  Applying strategies to TRAIN data:")
    train_osc_mod, train_mask, use_gt, use_dd = apply_strategies_to_data(
        strategy_ids, train_osc_raw, close_train, high_train, low_train, vol_train, cross_asset_data
    )
    print(f"    Train mask: {train_mask.sum()}/{len(train_mask)} bars ({train_mask.sum()/len(train_mask)*100:.1f}%)")

    # Apply strategies to validation oscillators/masks
    close_val = val_df['close'].values
    high_val = val_df['high'].values
    low_val = val_df['low'].values
    vol_val = val_df['volume'].values

    print("\n  Applying strategies to VALIDATION data:")
    val_osc_mod, val_mask, _, _ = apply_strategies_to_data(
        strategy_ids, val_osc_raw, close_val, high_val, low_val, vol_val, cross_asset_data
    )
    print(f"    Val mask: {val_mask.sum()}/{len(val_mask)} bars ({val_mask.sum()/len(val_mask)*100:.1f}%)")

    # Apply strategies to holdout oscillators/masks
    close_holdout = holdout_df['close'].values
    high_holdout = holdout_df['high'].values
    low_holdout = holdout_df['low'].values
    vol_holdout = holdout_df['volume'].values

    print("\n  Applying strategies to HOLDOUT data:")
    holdout_osc_mod, holdout_mask, _, _ = apply_strategies_to_data(
        strategy_ids, holdout_osc_raw, close_holdout, high_holdout, low_holdout, vol_holdout, cross_asset_data
    )
    print(f"    Holdout mask: {holdout_mask.sum()}/{len(holdout_mask)} bars ({holdout_mask.sum()/len(holdout_mask)*100:.1f}%)")

    # Apply masks to oscillators
    train_osc_masked = get_masked_oscillators(train_osc_mod, train_mask)
    val_osc_masked = get_masked_oscillators(val_osc_mod, val_mask)
    holdout_osc_masked = get_masked_oscillators(holdout_osc_mod, holdout_mask)

    # Run optimization on modified train data
    print("\n  Running optimization on strategy-enhanced train data...")
    default_osc_mod = train_osc_masked.get('composite_smooth', close_train * 0)

    strat_results = run_optimization(
        close_train, default_osc_mod, train_osc_masked,
        n_trials=n_trials, n_workers=n_workers,
        optimize_metric='risk_adjusted', label="STRATEGY",
        high_prices=high_train, low_prices=low_train, volume=vol_train,
        use_drawdown_penalty=use_dd, max_drawdown_threshold=5.0, drawdown_penalty_weight=0.5,
    )

    if use_gt:
        for r in strat_results:
            r['gt_score'] = compute_gt_score(r)
        strat_top = select_best(strat_results, top_n=5, metric='gt_score')
    else:
        strat_top = select_best(strat_results, top_n=5, metric='risk_adjusted')

    strat_best = strat_top[0] if strat_top else None

    if strat_best:
        # Evaluate on validation
        val_result = evaluate_on_test(strat_best, val_df, val_osc_masked)
        # Evaluate on holdout — TRUE OUT-OF-SAMPLE
        holdout_result = evaluate_on_test(strat_best, holdout_df, holdout_osc_masked)

        print(f"\n  STRATEGY [{', '.join(str(s) for s in strategy_ids)}] Results:")
        print(f"    Train:   Return={strat_best.get('total_return', 0):.2f}%, "
              f"WR={strat_best.get('win_rate', 0):.1f}%, "
              f"Trades={strat_best.get('num_trades', 0)}, "
              f"PF={strat_best.get('profit_factor', 0):.2f}, "
              f"MaxDD={strat_best.get('max_drawdown', 0):.2f}%")
        if val_result:
            print(f"    Valid:   Return={val_result.get('total_return', 0):.2f}%, "
                  f"WR={val_result.get('win_rate', 0):.1f}%, "
                  f"Trades={val_result.get('num_trades', 0)}, "
                  f"PF={val_result.get('profit_factor', 0):.2f}, "
                  f"MaxDD={val_result.get('max_drawdown', 0):.2f}%")
        if holdout_result:
            print(f"    HOLDOUT: Return={holdout_result.get('total_return', 0):.2f}%, "
                  f"WR={holdout_result.get('win_rate', 0):.1f}%, "
                  f"Trades={holdout_result.get('num_trades', 0)}, "
                  f"PF={holdout_result.get('profit_factor', 0):.2f}, "
                  f"MaxDD={holdout_result.get('max_drawdown', 0):.2f}%")

        # Also eval top 5 on holdout for robustness
        top5_holdout = []
        for params in strat_top[:5]:
            hr = evaluate_on_test(params, holdout_df, holdout_osc_masked)
            if hr:
                top5_holdout.append(hr.get('total_return', 0))
        avg_top5_holdout = np.mean(top5_holdout) if top5_holdout else 0
        print(f"    Top5 Avg Holdout: {avg_top5_holdout:.2f}%")
    else:
        print("  STRATEGY: No valid results!")
        val_result = holdout_result = None
        avg_top5_holdout = 0

    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)

    bl_train_ret = baseline_best.get('total_return', 0) if baseline_best else 0
    bl_val_ret = bl_val_result.get('total_return', 0) if bl_val_result else 0
    bl_holdout_ret = bl_holdout_result.get('total_return', 0) if bl_holdout_result else 0

    st_train_ret = strat_best.get('total_return', 0) if strat_best else 0
    st_val_ret = val_result.get('total_return', 0) if val_result else 0
    st_holdout_ret = holdout_result.get('total_return', 0) if holdout_result else 0

    header = f"{'':>20} {'Train':>12} {'Validation':>12} {'HOLDOUT':>12}"
    print(header)
    print("-" * 60)
    print(f"{'Buy & Hold':>20} {bh_train:>11.2f}% {bh_val:>11.2f}% {bh_holdout:>11.2f}%")
    print(f"{'Baseline':>20} {bl_train_ret:>11.2f}% {bl_val_ret:>11.2f}% {bl_holdout_ret:>11.2f}%")
    print(f"{'Strategy [14,4,12]':>20} {st_train_ret:>11.2f}% {st_val_ret:>11.2f}% {st_holdout_ret:>11.2f}%")
    print("-" * 60)
    print(f"{'Delta vs Baseline':>20} {st_train_ret-bl_train_ret:>+11.2f}% {st_val_ret-bl_val_ret:>+11.2f}% {st_holdout_ret-bl_holdout_ret:>+11.2f}%")
    print(f"{'Delta vs B&H':>20} {st_train_ret-bh_train:>+11.2f}% {st_val_ret-bh_val:>+11.2f}% {st_holdout_ret-bh_holdout:>+11.2f}%")

    bl_val_wr = bl_val_result.get('win_rate', 0) if bl_val_result else 0
    bl_holdout_wr = bl_holdout_result.get('win_rate', 0) if bl_holdout_result else 0
    st_val_wr = val_result.get('win_rate', 0) if val_result else 0
    st_holdout_wr = holdout_result.get('win_rate', 0) if holdout_result else 0

    bl_val_trades = bl_val_result.get('num_trades', 0) if bl_val_result else 0
    bl_holdout_trades = bl_holdout_result.get('num_trades', 0) if bl_holdout_result else 0
    st_val_trades = val_result.get('num_trades', 0) if val_result else 0
    st_holdout_trades = holdout_result.get('num_trades', 0) if holdout_result else 0

    bl_val_dd = bl_val_result.get('max_drawdown', 0) if bl_val_result else 0
    bl_holdout_dd = bl_holdout_result.get('max_drawdown', 0) if bl_holdout_result else 0
    st_val_dd = val_result.get('max_drawdown', 0) if val_result else 0
    st_holdout_dd = holdout_result.get('max_drawdown', 0) if holdout_result else 0

    print(f"\n{'':>20} {'Baseline':>25} {'Strategy':>25}")
    print(f"{'':>20} {'Val':>12} {'Holdout':>12} {'Val':>12} {'Holdout':>12}")
    print("-" * 72)
    print(f"{'Win Rate':>20} {bl_val_wr:>11.1f}% {bl_holdout_wr:>11.1f}% {st_val_wr:>11.1f}% {st_holdout_wr:>11.1f}%")
    print(f"{'Trades':>20} {bl_val_trades:>12} {bl_holdout_trades:>12} {st_val_trades:>12} {st_holdout_trades:>12}")
    print(f"{'Max Drawdown':>20} {bl_val_dd:>11.2f}% {bl_holdout_dd:>11.2f}% {st_val_dd:>11.2f}% {st_holdout_dd:>11.2f}%")

    # Overfitting diagnostic
    print("\n" + "=" * 100)
    print("OVERFITTING DIAGNOSTIC")
    print("=" * 100)
    if st_train_ret > 0 and st_holdout_ret != 0:
        train_to_holdout_ratio = st_holdout_ret / st_train_ret
        print(f"  Strategy holdout/train ratio: {train_to_holdout_ratio:.2f}")
        if train_to_holdout_ratio > 0.5:
            print(f"  >> GOOD: Holdout retains >{train_to_holdout_ratio*100:.0f}% of train performance")
        elif train_to_holdout_ratio > 0.2:
            print(f"  >> MODERATE: Some overfitting, holdout retains {train_to_holdout_ratio*100:.0f}% of train")
        elif train_to_holdout_ratio > 0:
            print(f"  >> WEAK: Significant overfitting, holdout retains only {train_to_holdout_ratio*100:.0f}% of train")
        else:
            print(f"  >> OVERFIT: Holdout is negative while train is positive")
    if bl_holdout_ret != 0:
        improvement_ratio = (st_holdout_ret - bl_holdout_ret) / abs(bl_holdout_ret) if bl_holdout_ret != 0 else float('inf')
        print(f"  Strategy improvement over baseline on holdout: {st_holdout_ret - bl_holdout_ret:+.2f}%")
    print(f"  Top5 parameter avg on holdout: {avg_top5_holdout:.2f}% (lower spread = more robust)")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_data = {
        'ticker': ticker, 'interval': interval,
        'strategy_ids': strategy_ids,
        'strategy_names': [STRATEGY_NAMES.get(s, f'#{s}') for s in strategy_ids],
        'n_trials': n_trials, 'n_workers': n_workers,
        'split': {'train': len(train_df), 'validation': len(val_df), 'holdout': len(holdout_df)},
        'buy_and_hold': {'train': bh_train, 'validation': bh_val, 'holdout': bh_holdout},
        'baseline': {
            'train_return': bl_train_ret,
            'val_return': bl_val_ret, 'val_wr': bl_val_wr, 'val_trades': bl_val_trades, 'val_max_dd': bl_val_dd,
            'holdout_return': bl_holdout_ret, 'holdout_wr': bl_holdout_wr, 'holdout_trades': bl_holdout_trades, 'holdout_max_dd': bl_holdout_dd,
        },
        'strategy': {
            'train_return': st_train_ret,
            'val_return': st_val_ret, 'val_wr': st_val_wr, 'val_trades': st_val_trades, 'val_max_dd': st_val_dd,
            'holdout_return': st_holdout_ret, 'holdout_wr': st_holdout_wr, 'holdout_trades': st_holdout_trades, 'holdout_max_dd': st_holdout_dd,
            'top5_avg_holdout': avg_top5_holdout,
        },
        'timestamp': datetime.now().isoformat(),
    }
    filename = f'strategy_oos_validation_{ticker}_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {filename}")

    return save_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Out-of-Sample Strategy Validation')
    parser.add_argument('--ticker', default='ES=F')
    parser.add_argument('--interval', default='15m')
    parser.add_argument('--trials', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--days', type=int, default=59)
    parser.add_argument('--strategies', default='14,4,12', help='Comma-separated strategy IDs')
    args = parser.parse_args()

    n_workers = args.workers or min(os.cpu_count() or 8, 32)
    strategy_ids = [int(x.strip()) for x in args.strategies.split(',')]

    run_split_evaluation(args.ticker, args.interval, strategy_ids, args.trials, n_workers, args.days)
