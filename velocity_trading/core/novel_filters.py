"""
Novel Strategy Filters for Live Trading (v7+)

Implements the optimal strategy combination [14, 4, 12]:
  #14 - GradientBoosting entry quality scorer (trained on buy signal bars)
  #4  - OU Optimal Entry/Exit (Ornstein-Uhlenbeck threshold amplification)
  #12 - SMOTE + XGBoost entry quality scorer (trained on buy signal bars)

Entry models (#14, #12) are trained specifically on buy signal bars to learn
which entries are likely profitable vs unprofitable. This allows them to
discriminate AMONG buy signals rather than comparing against random bars.

Usage:
    filters = NovelStrategyFilters(config['novel_strategies'])
    filters.train(df)  # Train OU thresholds at startup

    # In _trading_cycle, after oscillator calculation:
    df = filters.apply_ou_boost(df)  # Modify oscillator values

    # After signals are computed:
    filters.train_exit_model(df)  # Trains entry models + exit model

    # In _check_entry, after buy_signal detected:
    if not filters.should_enter(df, bar_idx):
        return  # Reject entry
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Tuple

warnings.filterwarnings('ignore', category=RuntimeWarning)


class NovelStrategyFilters:
    """
    Manages the three novel strategy filters for live trading.

    Gated by config — only initialized when config has 'novel_strategies' key.
    """

    def __init__(self, novel_config: Dict):
        """
        Args:
            novel_config: The 'novel_strategies' dict from velocity_config.json.
                         Contains strategy_ids, strategy_names, etc.
        """
        self.strategy_ids = novel_config.get('strategy_ids', [14, 4, 12])
        self.trained = False

        # Models (trained after signals are available)
        self._gbm_model = None       # #14 GradientBoosting
        self._xgb_model = None       # #12 XGBoost
        self._ou_entry = None        # #4 OU entry threshold
        self._ou_exit = None         # #4 OU exit threshold

        # ML Exit Model (Strategy D from exit comparison)
        self.exit_model = NovelExitModel()

        # Config for each strategy (configurable from velocity_config.json)
        self._gbm_seq_len = 20
        self._gbm_conf_threshold = novel_config.get('gbm_conf_threshold', None)
        self._smote_conf_threshold = novel_config.get('smote_conf_threshold', None)
        self._ou_window = 300

        # Adaptive thresholds (computed during training from signal prediction distribution)
        self._gbm_adaptive_threshold = None
        self._smote_adaptive_threshold = None

        # Track training timestamps
        self._last_train_time = None
        self._entry_models_trained = False

    def train(self, df: pd.DataFrame, oscillator_col: str = 'JD_Osc'):
        """
        Train OU thresholds on historical data.

        GBM (#14) and XGB (#12) are deferred to train_exit_model() because
        they need buy_signal column to train on buy signal bars specifically.

        Args:
            df: DataFrame with OHLCV + oscillator columns
            oscillator_col: Name of the oscillator column to use
        """
        # Get oscillator values — try multiple column names
        osc_values = None
        for col in [oscillator_col, 'JD_Osc', 'osc_smooth', 'composite_smooth']:
            if col in df.columns:
                osc_values = df[col].values.astype(float)
                break
        if osc_values is None:
            print("   [NovelFilters] WARNING: No oscillator column found, using zeros")
            osc_values = np.zeros(len(df))

        osc_values = np.nan_to_num(osc_values, nan=0.0)

        n_trained = 0

        # Strategy #4: OU thresholds (doesn't need buy signals)
        if 4 in self.strategy_ids:
            try:
                self._compute_ou(osc_values)
                if self._ou_entry is not None:
                    n_trained += 1
                    print(f"   [NovelFilters] #4 OU thresholds: entry={self._ou_entry:.4f}, exit={self._ou_exit:.4f}")
            except Exception as e:
                print(f"   [NovelFilters] #4 OU failed: {e}")

        # #14 and #12 deferred to train_exit_model (needs buy_signal column)
        deferred = []
        if 14 in self.strategy_ids:
            deferred.append("#14 GBM")
        if 12 in self.strategy_ids:
            deferred.append("#12 XGB")
        if deferred:
            print(f"   [NovelFilters] {', '.join(deferred)} deferred to signal-based training")

        self.trained = n_trained > 0 or len(deferred) > 0
        self._last_train_time = pd.Timestamp.now()

    def train_exit_model(self, df: pd.DataFrame, stop_loss_pct: float = 2.5,
                         oscillator_col: str = 'JD_Osc'):
        """
        Train entry quality models and ML exit model.

        Must be called AFTER train() and AFTER signals are calculated,
        because it needs buy_signal column.

        Training flow:
        1. Find all buy signal bars
        2. Label each: was the resulting trade profitable? (using actual price action)
        3. Train GBM (#14) and XGB (#12) on buy signal bars
        4. Calibrate thresholds from buy-signal prediction distribution
        5. Filter entries using trained models
        6. Train exit model on filtered entries

        Args:
            df: DataFrame with OHLCV + indicators + buy_signal column
            stop_loss_pct: Stop loss percentage for training labels
            oscillator_col: Oscillator column name
        """
        # Step 1: Find all buy signal bars
        buy_signal_bars = []
        for i in range(len(df)):
            if df.iloc[i].get('buy_signal', False):
                buy_signal_bars.append(i)

        if len(buy_signal_bars) < 20:
            print(f"   [NovelFilters] Only {len(buy_signal_bars)} buy signals, "
                  f"skipping ML training")
            return

        buy_signal_bars = np.array(buy_signal_bars)

        # Step 2: Train entry models on buy signal bars
        close = df['Close'].values.astype(float)
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        volume = df['Volume'].values.astype(float) if 'Volume' in df.columns else np.ones(len(df))
        n = len(close)

        osc_values = None
        for col in [oscillator_col, 'JD_Osc', 'osc_smooth', 'composite_smooth']:
            if col in df.columns:
                osc_values = df[col].values.astype(float)
                break
        if osc_values is None:
            osc_values = np.zeros(n)
        osc_values = np.nan_to_num(osc_values, nan=0.0)

        # Compute trade outcome labels for each buy signal bar
        # Label = 1 if the trade from this entry would be profitable
        # IMPORTANT: Use fixed labeling parameters independent of live SL/TP.
        # Live SL/TP affects trade management, not "was this a good entry?"
        # Labeling SL=1.0% (wide, avoids noise), TP=0.20% (MFE median ~0.24%)
        labeling_sl = 1.0
        labeling_tp = 0.20
        labels = self._compute_trade_labels(
            close, buy_signal_bars, labeling_sl, labeling_tp, max_hold=50
        )

        n_winners = labels.sum()
        n_losers = len(labels) - n_winners
        print(f"   [NovelFilters] Buy signal outcomes: {len(labels)} signals, "
              f"{n_winners} winners ({100*n_winners/len(labels):.0f}%), "
              f"{n_losers} losers ({100*n_losers/len(labels):.0f}%)")

        # Compute optimal TP from actual forward price data
        self._compute_optimal_tp(close, buy_signal_bars, stop_loss_pct)

        # Step 3: Train GBM on buy signal bars
        if 14 in self.strategy_ids:
            try:
                self._train_gbm_on_signals(
                    df, close, osc_values, volume, high, low,
                    buy_signal_bars, labels
                )
            except Exception as e:
                print(f"   [NovelFilters] #14 GBM training failed: {e}")

        # Step 4: Train XGB on buy signal bars
        if 12 in self.strategy_ids:
            try:
                self._train_xgb_on_signals(
                    df, close, osc_values, volume, high, low,
                    buy_signal_bars, labels
                )
            except Exception as e:
                print(f"   [NovelFilters] #12 XGB training failed: {e}")

        self._entry_models_trained = True

        # Step 5: Find filtered entries for exit model training
        entry_bars = []
        for i in buy_signal_bars:
            should_enter, _ = self.should_enter(df, bar_idx=int(i))
            if should_enter:
                entry_bars.append(i)

        entry_bars = np.array(entry_bars)
        if len(entry_bars) < 5:
            print(f"   [ExitModel] Only {len(entry_bars)} filtered entries, skipping")
            return

        # Step 6: Train exit model
        print(f"   [ExitModel] Training on {len(entry_bars)} filtered entries...")
        self.exit_model.train(
            df, entry_bars,
            stop_loss_pct=stop_loss_pct,
            oscillator_col=oscillator_col
        )

    def _compute_trade_labels(self, close, signal_bars, stop_loss_pct,
                               tp_pct, max_hold=50):
        """
        Compute binary labels for buy signal bars based on trade outcome.

        For each signal bar, simulates a trade with SL/TP and labels:
        - 1 = trade would be profitable (hit TP or positive at max hold)
        - 0 = trade would be unprofitable (hit SL or negative at max hold)
        """
        n = len(close)
        labels = []
        for bar in signal_bars:
            entry_price = close[bar]
            label = 0  # Default: loser
            end_bar = min(bar + max_hold, n - 1)

            for j in range(bar + 1, end_bar + 1):
                pnl_pct = ((close[j] - entry_price) / entry_price) * 100
                if pnl_pct >= tp_pct:
                    label = 1  # Winner — hit TP
                    break
                if pnl_pct <= -stop_loss_pct:
                    label = 0  # Loser — hit SL
                    break
            else:
                # Reached max hold — label by final PnL
                final_pnl = ((close[end_bar] - entry_price) / entry_price) * 100
                label = 1 if final_pnl > 0 else 0

            labels.append(label)
        return np.array(labels)

    def _compute_optimal_tp(self, close, signal_bars, stop_loss_pct, max_hold=50):
        """
        Compute optimal SL and TP from actual forward price data.

        Phase 1: MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse
                 Excursion) distributions — unencumbered by any SL/TP.
        Phase 2: SL sweep with TP fixed at 3.0% (safety cap). Finds the SL
                 that maximizes E[R]/trade.
        Phase 3: TP sweep with the optimal SL from Phase 2.

        Stores results as self.optimal_sl_pct and self.optimal_tp_pct.
        """
        n = len(close)
        if len(signal_bars) < 20:
            return

        # ── Phase 1: MFE and MAE distributions (no SL/TP interference) ──
        mfe_list = []  # Peak profit per signal
        mae_list = []  # Peak drawdown per signal (stored as positive)
        for bar in signal_bars:
            entry_price = close[bar]
            end_bar = min(bar + max_hold, n - 1)
            if end_bar <= bar:
                continue

            mfe = 0.0
            mae = 0.0
            for j in range(bar + 1, end_bar + 1):
                pnl_pct = ((close[j] - entry_price) / entry_price) * 100
                if pnl_pct > mfe:
                    mfe = pnl_pct
                if pnl_pct < mae:
                    mae = pnl_pct

            mfe_list.append(mfe)
            mae_list.append(abs(mae))  # Positive for easier comparison

        mfe_arr = np.array(mfe_list)
        mae_arr = np.array(mae_list)

        print(f"   [SL/TP Analysis] {len(mfe_arr)} signals analyzed (max_hold={max_hold} bars):")
        print(f"   MFE (peak profit):   p25={np.percentile(mfe_arr, 25):.3f}%, "
              f"median={np.median(mfe_arr):.3f}%, "
              f"p75={np.percentile(mfe_arr, 75):.3f}%, "
              f"mean={mfe_arr.mean():.3f}%")
        print(f"   MAE (peak drawdown): p25={np.percentile(mae_arr, 25):.3f}%, "
              f"median={np.median(mae_arr):.3f}%, "
              f"p75={np.percentile(mae_arr, 75):.3f}%, "
              f"mean={mae_arr.mean():.3f}%")

        # ── Phase 2: SL sweep (TP fixed at 3.0% safety cap) ──
        safety_tp = 3.0
        sl_candidates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                         0.75, 1.00, 1.50, 2.00]
        best_sl = 0.50
        best_sl_expected = -999

        print(f"\n   [SL Sweep] TP fixed at {safety_tp:.1f}% (safety cap):")
        print(f"   {'SL%':>6} | {'WR':>6} | {'E[R]/trade':>10} | {'Trades':>6} | {'PF':>6}")
        print(f"   {'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}-+-{'-'*6}")

        for sl in sl_candidates:
            wins = 0
            losses = 0
            total_pnl = 0.0
            gross_wins = 0.0
            gross_losses = 0.0

            for bar in signal_bars:
                entry_price = close[bar]
                end_bar = min(bar + max_hold, n - 1)
                if end_bar <= bar:
                    continue

                outcome_pnl = 0.0
                for j in range(bar + 1, end_bar + 1):
                    pnl_pct = ((close[j] - entry_price) / entry_price) * 100
                    if pnl_pct >= safety_tp:
                        outcome_pnl = safety_tp
                        wins += 1
                        gross_wins += safety_tp
                        break
                    if pnl_pct <= -sl:
                        outcome_pnl = -sl
                        losses += 1
                        gross_losses += sl
                        break
                else:
                    outcome_pnl = ((close[end_bar] - entry_price) / entry_price) * 100
                    if outcome_pnl > 0:
                        wins += 1
                        gross_wins += outcome_pnl
                    else:
                        losses += 1
                        gross_losses += abs(outcome_pnl)

                total_pnl += outcome_pnl

            total_trades = wins + losses
            if total_trades == 0:
                continue
            wr = wins / total_trades
            expected_per_trade = total_pnl / total_trades
            pf = gross_wins / gross_losses if gross_losses > 0 else 999.99

            marker = ""
            if expected_per_trade > best_sl_expected:
                best_sl_expected = expected_per_trade
                best_sl = sl
                marker = " <-- best"

            print(f"   {sl:>5.2f}% | {wr:>5.1%} | {expected_per_trade:>+9.4f}% | {total_trades:>6} | {pf:>5.2f}{marker}")

        self.optimal_sl_pct = best_sl
        print(f"\n   [SL Sweep] Optimal SL: {best_sl:.2f}% "
              f"(E[R]={best_sl_expected:+.4f}%/trade, TP={safety_tp:.1f}%)")

        # ── Phase 3: TP sweep (SL fixed at optimal from Phase 2) ──
        tp_candidates = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00,
                         1.50, 2.00, 3.00, 5.00]
        best_tp = safety_tp
        best_tp_expected = -999

        print(f"\n   [TP Sweep] SL fixed at {best_sl:.2f}% (from SL sweep):")
        print(f"   {'TP%':>6} | {'WR':>6} | {'E[R]/trade':>10} | {'Trades':>6} | {'PF':>6}")
        print(f"   {'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}-+-{'-'*6}")

        for tp in tp_candidates:
            wins = 0
            losses = 0
            total_pnl = 0.0
            gross_wins = 0.0
            gross_losses = 0.0

            for bar in signal_bars:
                entry_price = close[bar]
                end_bar = min(bar + max_hold, n - 1)
                if end_bar <= bar:
                    continue

                outcome_pnl = 0.0
                for j in range(bar + 1, end_bar + 1):
                    pnl_pct = ((close[j] - entry_price) / entry_price) * 100
                    if pnl_pct >= tp:
                        outcome_pnl = tp
                        wins += 1
                        gross_wins += tp
                        break
                    if pnl_pct <= -best_sl:
                        outcome_pnl = -best_sl
                        losses += 1
                        gross_losses += best_sl
                        break
                else:
                    outcome_pnl = ((close[end_bar] - entry_price) / entry_price) * 100
                    if outcome_pnl > 0:
                        wins += 1
                        gross_wins += outcome_pnl
                    else:
                        losses += 1
                        gross_losses += abs(outcome_pnl)

                total_pnl += outcome_pnl

            total_trades = wins + losses
            if total_trades == 0:
                continue
            wr = wins / total_trades
            expected_per_trade = total_pnl / total_trades
            pf = gross_wins / gross_losses if gross_losses > 0 else 999.99

            marker = ""
            if expected_per_trade > best_tp_expected:
                best_tp_expected = expected_per_trade
                best_tp = tp
                marker = " <-- best"

            print(f"   {tp:>5.2f}% | {wr:>5.1%} | {expected_per_trade:>+9.4f}% | {total_trades:>6} | {pf:>5.2f}{marker}")

        self.optimal_tp_pct = best_tp
        print(f"\n   [SL/TP Analysis] RECOMMENDED: SL={best_sl:.2f}%, TP={best_tp:.2f}% "
              f"(E[R]={best_tp_expected:+.4f}%/trade)")

    def _train_gbm_on_signals(self, df, close, osc_values, volume, high, low,
                               signal_bars, labels, seq_len=20):
        """
        Train GradientBoosting (#14) on buy signal bars specifically.

        Instead of training on all bars to predict "will any bar go up?",
        trains on buy signal bars to predict "will THIS entry be profitable?"
        """
        from sklearn.ensemble import GradientBoostingClassifier

        n = len(close)
        returns_1 = np.zeros(n)
        returns_1[1:] = np.diff(close) / close[:-1]
        vol_20 = pd.Series(returns_1).rolling(20, min_periods=1).std().values
        velocity = np.zeros(n)
        velocity[1:] = np.diff(osc_values)
        accel = np.zeros(n)
        accel[1:] = np.diff(velocity)

        features_all = np.column_stack([osc_values, velocity, accel, returns_1, vol_20])
        features_all = np.nan_to_num(features_all, nan=0.0)

        # Build features only for buy signal bars
        X_seqs, y_labels = [], []
        for bar, label in zip(signal_bars, labels):
            if bar < seq_len + 10:
                continue  # Not enough history for this bar

            seq = features_all[bar - seq_len:bar]
            feat = np.concatenate([
                seq.mean(axis=0), seq.std(axis=0), seq[-1],
                np.polyfit(range(seq_len), seq[:, 0], 1)
            ])
            X_seqs.append(feat)
            y_labels.append(label)

        if len(X_seqs) < 30:
            print(f"   [NovelFilters] #14: only {len(X_seqs)} valid signal bars, skipping")
            return

        X, y = np.array(X_seqs), np.array(y_labels)

        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            print(f"   [NovelFilters] #14: insufficient class balance "
                  f"({y.sum()} wins, {len(y)-y.sum()} losses), skipping")
            return

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        model.fit(X, y)
        self._gbm_model = model
        self._gbm_seq_len = seq_len

        # Set threshold from signal prediction distribution
        train_probs = model.predict_proba(X)[:, 1]
        self._gbm_adaptive_threshold = float(np.percentile(train_probs, 25))
        print(f"   [NovelFilters] #14 GBM trained on {len(X)} buy signals "
              f"({y.sum()} winners, {len(y)-y.sum()} losers)")
        print(f"   [NovelFilters] #14 GBM signal scores: "
              f"min={train_probs.min():.3f}, p25={self._gbm_adaptive_threshold:.3f}, "
              f"median={np.median(train_probs):.3f}, max={train_probs.max():.3f}")

        effective = self._gbm_conf_threshold if self._gbm_conf_threshold is not None else self._gbm_adaptive_threshold
        print(f"   [NovelFilters] #14 GBM threshold: {effective:.3f} "
              f"({'config' if self._gbm_conf_threshold is not None else 'adaptive p25'})")

    def _train_xgb_on_signals(self, df, close, osc_values, volume, high, low,
                                signal_bars, labels):
        """
        Train SMOTE + XGBoost (#12) on buy signal bars specifically.

        Same approach as GBM — learns to discriminate good vs bad buy signals.
        """
        try:
            from xgboost import XGBClassifier
            from imblearn.over_sampling import BorderlineSMOTE
        except ImportError:
            print("   [NovelFilters] #12: xgboost or imblearn not installed")
            return

        n = len(close)
        returns_1 = np.zeros(n)
        returns_1[1:] = np.diff(close) / close[:-1]
        vol_20 = pd.Series(returns_1).rolling(20, min_periods=1).std().values

        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

        vol_sma = pd.Series(volume.astype(float)).rolling(20, min_periods=1).mean().values
        vol_ratio = volume / (vol_sma + 1e-10)

        price_series = pd.Series(close)
        rolling_min = price_series.rolling(20, min_periods=1).min().values
        rolling_max = price_series.rolling(20, min_periods=1).max().values
        price_range = rolling_max - rolling_min
        range_pos = np.where(price_range > 0, (close - rolling_min) / price_range, 0.5)

        # Build features only for buy signal bars
        features, valid_labels = [], []
        for bar, label in zip(signal_bars, labels):
            if bar < 10:
                continue
            vel = osc_values[bar] - osc_values[bar - 1] if bar > 0 else 0
            acc = vel - (osc_values[bar - 1] - osc_values[bar - 2]) if bar > 1 else 0
            features.append([
                osc_values[bar], vel, acc, vol_20[bar], atr[bar],
                vol_ratio[bar], range_pos[bar], returns_1[bar]
            ])
            valid_labels.append(label)

        X, y = np.array(features), np.array(valid_labels)
        if len(y) < 30 or y.sum() < 5 or (len(y) - y.sum()) < 5:
            print(f"   [NovelFilters] #12: insufficient data "
                  f"({len(y)} signals, {y.sum()} wins)")
            return

        # Apply SMOTE to balance classes
        try:
            k = min(5, min(int(y.sum()), int(len(y) - y.sum())) - 1)
            if k >= 1:
                smote = BorderlineSMOTE(random_state=42, k_neighbors=k)
                X_resampled, y_resampled = smote.fit_resample(X, y)
            else:
                X_resampled, y_resampled = X, y
        except Exception:
            X_resampled, y_resampled = X, y

        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
        model.fit(X_resampled, y_resampled)
        self._xgb_model = model

        # Set threshold from original (non-SMOTE) signal predictions
        train_probs = model.predict_proba(X)[:, 1]
        self._smote_adaptive_threshold = float(np.percentile(train_probs, 25))
        print(f"   [NovelFilters] #12 XGB trained on {len(X)} buy signals "
              f"({y.sum()} winners, {len(y)-y.sum()} losers)")
        print(f"   [NovelFilters] #12 XGB signal scores: "
              f"min={train_probs.min():.3f}, p25={self._smote_adaptive_threshold:.3f}, "
              f"median={np.median(train_probs):.3f}, max={train_probs.max():.3f}")

        effective = self._smote_conf_threshold if self._smote_conf_threshold is not None else self._smote_adaptive_threshold
        print(f"   [NovelFilters] #12 XGB threshold: {effective:.3f} "
              f"({'config' if self._smote_conf_threshold is not None else 'adaptive p25'})")

    def _compute_ou(self, osc_values, window=300):
        """Compute OU optimal entry/exit thresholds (#4)."""
        n = min(len(osc_values), window)
        vals = osc_values[-n:]
        dt = 1.0
        x = vals[:-1]
        dx = np.diff(vals)
        if np.std(x) < 1e-10:
            return
        b = np.sum(dx * (x - np.mean(x))) / np.sum((x - np.mean(x))**2)
        a = np.mean(dx) - b * np.mean(x)
        mu = -b / dt
        if mu <= 0:
            return
        theta = a / (mu * dt)
        residuals = dx - a - b * x
        sigma = np.std(residuals) / np.sqrt(dt)
        spread = sigma / np.sqrt(2 * mu)
        self._ou_entry = theta - 1.0 * spread
        self._ou_exit = theta + 0.5 * spread

    def apply_ou_boost(self, df: pd.DataFrame, oscillator_col: str = 'JD_Osc') -> pd.DataFrame:
        """
        Apply OU threshold amplification to oscillator values (#4).

        Call this in _trading_cycle() after calculate_composite_oscillator()
        but BEFORE calculate_velocity_signals().

        Boosts oscillator values near OU entry/exit thresholds by 1.3x,
        making signals near those levels more likely to fire.
        """
        if 4 not in self.strategy_ids or self._ou_entry is None:
            return df

        # Find the oscillator column
        col = None
        for c in [oscillator_col, 'JD_Osc', 'osc_smooth']:
            if c in df.columns:
                col = c
                break
        if col is None:
            return df

        vals = df[col].values.copy().astype(float)
        entry = self._ou_entry
        exit_lvl = self._ou_exit

        near_entry = np.abs(vals - entry) < 0.1
        near_exit = np.abs(vals - exit_lvl) < 0.1
        boost = np.ones(len(vals))
        boost[near_entry] = 1.3
        boost[near_exit] = 1.3
        df[col] = np.clip(vals * boost, -1, 1)

        return df

    def should_enter(self, df: pd.DataFrame, bar_idx: int = -1,
                     oscillator_col: str = 'JD_Osc') -> Tuple[bool, str]:
        """
        Check if the current bar passes all entry filters.

        Call this in _check_entry() after buy_signal is detected.

        Args:
            df: DataFrame with OHLCV + indicators
            bar_idx: Index of the bar to check (default -1 = most recent)
            oscillator_col: Oscillator column name

        Returns:
            Tuple of (should_enter: bool, reason: str)
        """
        if not self._entry_models_trained:
            return True, "entry models not yet trained"

        close = df['Close'].values.astype(float)
        n = len(close)
        idx = bar_idx if bar_idx >= 0 else n + bar_idx  # Convert negative index

        # Get oscillator values
        osc_values = None
        for col in [oscillator_col, 'JD_Osc', 'osc_smooth', 'composite_smooth']:
            if col in df.columns:
                osc_values = df[col].values.astype(float)
                break
        if osc_values is None:
            osc_values = np.zeros(n)
        osc_values = np.nan_to_num(osc_values, nan=0.0)

        reasons = []

        # Check #14: GradientBoosting confidence
        if 14 in self.strategy_ids and self._gbm_model is not None:
            conf = self._score_gbm(close, osc_values, df, idx)
            gbm_thresh = self._gbm_conf_threshold if self._gbm_conf_threshold is not None else self._gbm_adaptive_threshold
            if gbm_thresh is None:
                gbm_thresh = 0.5  # Ultimate fallback
            if conf is not None and conf < gbm_thresh:
                return False, f"GBM confidence too low ({conf:.2f} < {gbm_thresh:.2f})"
            if conf is not None:
                reasons.append(f"GBM={conf:.2f}")

        # Check #12: SMOTE+XGBoost confidence
        if 12 in self.strategy_ids and self._xgb_model is not None:
            conf = self._score_xgb(close, osc_values, df, idx)
            xgb_thresh = self._smote_conf_threshold if self._smote_conf_threshold is not None else self._smote_adaptive_threshold
            if xgb_thresh is None:
                xgb_thresh = 0.5  # Ultimate fallback
            if conf is not None and conf < xgb_thresh:
                return False, f"XGB confidence too low ({conf:.2f} < {xgb_thresh:.2f})"
            if conf is not None:
                reasons.append(f"XGB={conf:.2f}")

        return True, " | ".join(reasons) if reasons else "all filters passed"

    def _score_gbm(self, close, osc_values, df, idx) -> Optional[float]:
        """Score a single bar with GradientBoosting (#14)."""
        if self._gbm_model is None:
            return None

        n = len(close)
        seq_len = self._gbm_seq_len

        if idx < seq_len + 10:
            return None  # Not enough history

        returns_1 = np.zeros(n)
        returns_1[1:] = np.diff(close) / close[:-1]
        vol_20 = pd.Series(returns_1).rolling(20, min_periods=1).std().values
        velocity = np.zeros(n)
        velocity[1:] = np.diff(osc_values)
        accel = np.zeros(n)
        accel[1:] = np.diff(velocity)

        features_all = np.column_stack([osc_values, velocity, accel, returns_1, vol_20])
        features_all = np.nan_to_num(features_all, nan=0.0)

        seq = features_all[idx - seq_len:idx]
        feat = np.concatenate([seq.mean(axis=0), seq.std(axis=0), seq[-1],
                               np.polyfit(range(seq_len), seq[:, 0], 1)])

        try:
            confidence = self._gbm_model.predict_proba(feat.reshape(1, -1))[0][1]
            return float(confidence)
        except Exception:
            return None

    def _score_xgb(self, close, osc_values, df, idx) -> Optional[float]:
        """Score a single bar with SMOTE+XGBoost (#12)."""
        if self._xgb_model is None:
            return None

        n = len(close)
        if idx < 10:
            return None

        returns_1 = np.zeros(n)
        returns_1[1:] = np.diff(close) / close[:-1]
        vol_20 = pd.Series(returns_1).rolling(20, min_periods=1).std().values

        high = df['High'].values.astype(float) if 'High' in df.columns else close
        low = df['Low'].values.astype(float) if 'Low' in df.columns else close

        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

        volume = df['Volume'].values.astype(float) if 'Volume' in df.columns else np.ones(n)
        vol_sma = pd.Series(volume.astype(float)).rolling(20, min_periods=1).mean().values
        vol_ratio = volume / (vol_sma + 1e-10)

        price_series = pd.Series(close)
        rolling_min = price_series.rolling(20, min_periods=1).min().values
        rolling_max = price_series.rolling(20, min_periods=1).max().values
        price_range = rolling_max - rolling_min
        range_pos = np.where(price_range > 0, (close - rolling_min) / price_range, 0.5)

        vel = osc_values[idx] - osc_values[idx - 1] if idx > 0 else 0
        acc = vel - (osc_values[idx - 1] - osc_values[idx - 2]) if idx > 1 else 0
        feat = np.array([[osc_values[idx], vel, acc, vol_20[idx], atr[idx],
                          vol_ratio[idx], range_pos[idx], returns_1[idx]]])

        try:
            confidence = self._xgb_model.predict_proba(feat)[0][1]
            return float(confidence)
        except Exception:
            return None

    def needs_retrain(self, hours=24) -> bool:
        """Check if models should be retrained (default: daily)."""
        if self._last_train_time is None:
            return True
        elapsed = (pd.Timestamp.now() - self._last_train_time).total_seconds() / 3600
        return elapsed >= hours


class NovelExitModel:
    """
    ML-based exit model for v7+ strategies.

    Uses a GradientBoostingClassifier to predict optimal exit timing.
    Trained on historical trade data — exits when the model predicts that
    exiting now gives better risk-adjusted return than holding.

    Features: bars_held, pnl, velocity, acceleration, jerk, osc, vol_20,
              atr_14, distance_from_high, price_ratio

    This model was discovered via exit_strategy_comparison.py as Strategy D,
    which achieved +11.59% return (2x baseline), 90% WR, 0.16% max DD.
    """

    def __init__(self):
        self._model = None
        self.trained = False
        self._last_train_time = None

    def train(self, df: pd.DataFrame, entry_bars: np.ndarray,
              stop_loss_pct: float = 2.5, forward_bars: int = 10,
              oscillator_col: str = 'JD_Osc'):
        """
        Train the exit model on historical entry signals.

        Args:
            df: DataFrame with OHLCV + indicators
            entry_bars: Array of bar indices where filtered entries occurred
            stop_loss_pct: Stop loss percentage for training labels
            forward_bars: How many bars ahead to evaluate hold vs exit
            oscillator_col: Oscillator column name
        """
        from sklearn.ensemble import GradientBoostingClassifier

        if len(entry_bars) < 5:
            print("   [ExitModel] Insufficient entries for training")
            return

        close = df['Close'].values.astype(float)
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        n = len(close)

        # Get indicator arrays
        osc = self._get_col(df, [oscillator_col, 'JD_Osc', 'osc_smooth', 'composite_smooth'])
        vel = self._get_col(df, ['velocity'])
        accel_arr = self._get_col(df, ['acceleration'])
        jerk_arr = self._get_col(df, ['jerk'])

        returns_1 = np.zeros(n)
        returns_1[1:] = np.diff(close) / close[:-1]
        vol_20 = pd.Series(returns_1).rolling(20, min_periods=1).std().values

        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().values

        # Train on first 70% of entries, but model is evaluated on ALL
        split = max(1, int(len(entry_bars) * 0.7))
        train_entries = entry_bars[:split]

        X_train, y_train = [], []
        for eb in train_entries:
            ep = close[eb]
            hwm = ep
            max_bar = min(n - forward_bars - 1, eb + 200)
            for j in range(eb + 1, max_bar):
                if close[j] > hwm:
                    hwm = close[j]
                pnl_now = ((close[j] - ep) / ep) * 100
                if pnl_now <= -stop_loss_pct:
                    break

                feat = self._build_features(
                    close, osc, vel, accel_arr, jerk_arr,
                    vol_20, atr_14, ep, hwm, j, eb
                )

                future_end = min(j + forward_bars, n - 1)
                future_prices = close[j:future_end + 1]
                max_future_pnl = ((future_prices.max() - ep) / ep) * 100
                future_dd = ((future_prices.max() - future_prices.min()) / future_prices.max()) * 100
                risk_adj_hold = max_future_pnl - future_dd * 0.5
                label = 1 if pnl_now >= risk_adj_hold else 0

                X_train.append(feat)
                y_train.append(label)

        if len(X_train) < 50:
            print("   [ExitModel] Insufficient training samples")
            return

        X_train, y_train = np.array(X_train), np.array(y_train)
        self._model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42
        )
        self._model.fit(X_train, y_train)
        self.trained = True
        self._last_train_time = pd.Timestamp.now()
        print(f"   [ExitModel] Trained on {len(train_entries)} entries, "
              f"{len(X_train)} samples, exit rate={y_train.mean():.2f}")

    def should_exit(self, df: pd.DataFrame, entry_price: float,
                    entry_bar_idx: int, current_bar_idx: int,
                    high_watermark: float = None,
                    oscillator_col: str = 'JD_Osc') -> Tuple[bool, float, str]:
        """
        Check if the model recommends exiting at the current bar.

        Args:
            df: DataFrame with OHLCV + indicators
            entry_price: Position entry price
            entry_bar_idx: Bar index of entry
            current_bar_idx: Current bar index to evaluate
            high_watermark: Highest price since entry (computed if None)
            oscillator_col: Oscillator column name

        Returns:
            Tuple of (should_exit, probability, reason_string)
        """
        if not self.trained or self._model is None:
            return False, 0.0, "model not trained"

        close = df['Close'].values.astype(float)
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        n = len(close)

        idx = current_bar_idx if current_bar_idx >= 0 else n + current_bar_idx

        if idx < 1 or idx >= n:
            return False, 0.0, "invalid index"

        osc = self._get_col(df, [oscillator_col, 'JD_Osc', 'osc_smooth', 'composite_smooth'])
        vel = self._get_col(df, ['velocity'])
        accel_arr = self._get_col(df, ['acceleration'])
        jerk_arr = self._get_col(df, ['jerk'])

        returns_1 = np.zeros(n)
        returns_1[1:] = np.diff(close) / close[:-1]
        vol_20 = pd.Series(returns_1).rolling(20, min_periods=1).std().values

        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().values

        if high_watermark is None:
            high_watermark = close[entry_bar_idx:idx + 1].max()

        feat = self._build_features(
            close, osc, vel, accel_arr, jerk_arr,
            vol_20, atr_14, entry_price, high_watermark, idx, entry_bar_idx
        )

        try:
            prob = self._model.predict_proba(feat.reshape(1, -1))[0][1]
            if prob > 0.5:
                return True, prob, f"ML Exit (p={prob:.2f})"
            return False, prob, f"ML hold (p={prob:.2f})"
        except Exception:
            return False, 0.0, "prediction error"

    def _build_features(self, close, osc, vel, accel_arr, jerk_arr,
                        vol_20, atr_14, entry_price, hwm, i, entry_bar):
        """Build feature vector for a single bar."""
        bars_held = i - entry_bar
        pnl = ((close[i] - entry_price) / entry_price) * 100
        dist_high = ((hwm - close[i]) / hwm) * 100 if hwm > 0 else 0
        return np.array([bars_held, pnl, vel[i], accel_arr[i], jerk_arr[i],
                         osc[i], vol_20[i], atr_14[i], dist_high,
                         close[i] / entry_price - 1])

    @staticmethod
    def _get_col(df, col_names):
        """Get first matching column as numpy array, or zeros."""
        for col in col_names:
            if col in df.columns:
                return np.nan_to_num(df[col].values.astype(float), nan=0.0)
        return np.zeros(len(df))

    def needs_retrain(self, hours=24) -> bool:
        """Check if model should be retrained."""
        if self._last_train_time is None:
            return True
        elapsed = (pd.Timestamp.now() - self._last_train_time).total_seconds() / 3600
        return elapsed >= hours
