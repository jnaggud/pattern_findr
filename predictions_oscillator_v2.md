# Price Prediction Feature - Implementation Plan

## Overview
Add a **Price Prediction Suite** to `oscillator_predictor_page.py` as new tabs, providing:
1. Daily Range Prediction (High/Low)
2. Price Targets (optimal take-profit levels)
3. Exit Timing predictions

Then port to `velocity_live_trader_2.py` for production use.

---

## Architecture

```
[Price Data + Options Data]
        |
        v
[Feature Engineering] --> ATR, IV, PCR, Technical Indicators, LSTM Features
        |
        v
[ML Models] --> XGBRegressor (Range), XGBRegressor (Targets), LSTM (Timing)
        |
        v
[Conformal Prediction] --> Prediction Intervals with confidence
        |
        v
[Streamlit UI] --> 3 Tabs with Charts, Metrics, Alerts
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `price_prediction.py` | CREATE | Core prediction classes (~400 lines) |
| `oscillator_predictor_page.py` | MODIFY | Add 3 new tabs for predictions |
| `polygon_manager.py` | MODIFY | Add IV calculation, options chain methods |
| `velocity_live_trader_2.py` | CREATE | Production version with predictions (Phase 2) |

---

## Phase 1: Create price_prediction.py

### Class 1: PriceRangePredictor
```python
class PriceRangePredictor:
    """Predicts daily high/low range using options IV + technical indicators"""

    def get_options_features(self, ticker: str) -> dict
    def calculate_implied_volatility(self, options_data: dict) -> float
    def create_range_features(self, df: pd.DataFrame, options_features: dict) -> pd.DataFrame
    def train_range_model(self, features: pd.DataFrame, target: pd.Series) -> dict
    def predict_daily_range(self, features: pd.DataFrame) -> Tuple[float, float, float]
```

**Features for Range Prediction:**
- ATR (7, 14, 21 periods)
- Historical range percentiles (5d, 10d, 20d)
- Implied Volatility (from options or VIX proxy)
- Put/Call Ratio (volume and OI)
- Day of week / month effects
- Market regime (bull/bear/sideways)
- Oscillator volatility

**Target Variable:** `next_day_range = (next_high - next_low) / close`

### Class 2: PriceTargetCalculator
```python
class PriceTargetCalculator:
    """Calculates optimal take-profit levels"""

    def calculate_support_resistance(self, df: pd.DataFrame) -> dict
    def calculate_atr_targets(self, price: float, atr: float, direction: str) -> dict
    def calculate_fibonacci_targets(self, swing_high: float, swing_low: float) -> dict
    def calculate_options_targets(self, options_data: dict) -> dict
    def get_optimal_targets(self, df, price, direction, options_data) -> dict
```

**Target Methods:**
1. ATR-based (1x, 1.5x, 2x ATR)
2. Fibonacci extensions (1.272, 1.618, 2.0)
3. Support/Resistance levels
4. Options max pain and high OI strikes

### Class 3: ExitTimingPredictor
```python
class ExitTimingPredictor:
    """Predicts optimal exit timing using oscillator momentum"""

    def create_timing_features(self, df: pd.DataFrame) -> pd.DataFrame
    def train_timing_model(self, df: pd.DataFrame) -> dict
    def predict_exit_timing(self, df: pd.DataFrame, position: str) -> dict
```

**Features for Exit Timing:**
- Oscillator velocity and acceleration
- Distance from overbought/oversold
- Momentum exhaustion indicators
- Historical holding period analysis
- LSTM sequence features (optional)

**Output:** `{bars_until_exit, confidence, exit_urgency}`

---

## Phase 2: Enhance polygon_manager.py

Add these methods:

```python
def get_options_chain(self, ticker: str, expiration: str = None) -> list:
    """Get full options chain with strikes, IV, OI"""
    # Endpoint: /v3/snapshot/options/{underlying}

def calculate_aggregate_iv(self, options_chain: list, current_price: float) -> float:
    """Weighted average IV (ATM-weighted, OI-weighted)"""

def get_max_pain(self, ticker: str, expiration: str = None) -> float:
    """Calculate max pain price from options OI"""

def get_high_oi_strikes(self, ticker: str) -> dict:
    """Get strikes with highest OI (calls and puts)"""
```

---

## Phase 3: Add Streamlit UI (oscillator_predictor_page.py)

### Tab Structure

Add after existing "Model Training" section:

```python
st.markdown("---")
st.header("Price Prediction Suite")

pred_tab1, pred_tab2, pred_tab3 = st.tabs([
    "Daily Range",
    "Price Targets",
    "Exit Timing"
])
```

### Tab 1: Daily Range Prediction

```
+-----------------------------------------------------------+
|  DAILY RANGE PREDICTION                                    |
+-----------------------------------------------------------+
|  [Current Stats Row]                                       |
|  Today's Range: $X.XX (Y%) | ATR(14): $X.XX | IV: XX%     |
+-----------------------------------------------------------+
|  [PREDICTION CARD]                                         |
|  +-----------------------------------------------------+  |
|  | TOMORROW'S PREDICTED RANGE                          |  |
|  | High: $XXX.XX (+/-$X.XX)                            |  |
|  | Low:  $XXX.XX (+/-$X.XX)                            |  |
|  | Range: $X.XX (Y.X ATRs)                             |  |
|  | Confidence: 85%  [========  ]                       |  |
|  +-----------------------------------------------------+  |
+-----------------------------------------------------------+
|  [Candlestick Chart with Predicted Range Bands]           |
|  - Horizontal bands showing predicted high/low            |
|  - Confidence interval shading                            |
+-----------------------------------------------------------+
|  [Model Performance]                                       |
|  RMSE: $X.XX | R2: 0.XX | Direction Accuracy: XX%         |
+-----------------------------------------------------------+
```

### Tab 2: Price Targets

```
+-----------------------------------------------------------+
|  PRICE TARGETS                                             |
+-----------------------------------------------------------+
|  Position: LONG | Entry: $XXX.XX | Current: $XXX.XX       |
+-----------------------------------------------------------+
|  [TARGET LEVELS]                                           |
|  +-----------------------------------------------------+  |
|  | Target 1 (Conservative): $XXX.XX  (+X.X%)           |  |
|  |   Method: 1x ATR                                    |  |
|  | Target 2 (Moderate):     $XXX.XX  (+X.X%)           |  |
|  |   Method: Fib 1.272 + Resistance                    |  |
|  | Target 3 (Aggressive):   $XXX.XX  (+X.X%)           |  |
|  |   Method: 2x ATR + Options OI                       |  |
|  +-----------------------------------------------------+  |
+-----------------------------------------------------------+
|  [OPTIONS-BASED LEVELS] (if available)                    |
|  Max Pain: $XXX | High Call OI: $XXX | High Put OI: $XXX  |
+-----------------------------------------------------------+
|  [Chart with Target Lines]                                |
|  - Entry, targets, S/R levels plotted                     |
+-----------------------------------------------------------+
|  [Risk/Reward Calculator]                                 |
|  Stop Loss: $XXX.XX  -->  R:R Ratios: 1:1.5, 1:2.3, 1:3.8 |
+-----------------------------------------------------------+
```

### Tab 3: Exit Timing

```
+-----------------------------------------------------------+
|  EXIT TIMING PREDICTION                                    |
+-----------------------------------------------------------+
|  [Current Position]                                        |
|  Position: LONG | Entry: Dec 23 | Days Held: 14           |
|  Current P/L: +X.X% ($XXX)                                |
+-----------------------------------------------------------+
|  [EXIT TIMING PREDICTION]                                  |
|  +-----------------------------------------------------+  |
|  | Predicted Exit Window: 3-7 bars                     |  |
|  | Exit Urgency: MEDIUM  [========    ]                |  |
|  | Confidence: 72%                                     |  |
|  |                                                     |  |
|  | RECOMMENDATION: HOLD - Momentum still positive      |  |
|  +-----------------------------------------------------+  |
+-----------------------------------------------------------+
|  [Momentum Indicators]                                     |
|  Oscillator Velocity: +0.02 (bullish)                     |
|  Momentum Exhaustion: 35% (low)                           |
|  Distance from Peak: 0.15 (room to run)                   |
+-----------------------------------------------------------+
|  [Oscillator Chart with Exit Zones]                       |
|  - Velocity + Acceleration plotted                        |
|  - Exit probability zones highlighted                     |
+-----------------------------------------------------------+
```

---

## Phase 4: Port to velocity_live_trader_2.py

After Streamlit prototype works:

1. Copy velocity_live_trader.py to velocity_live_trader_2.py
2. Import from price_prediction.py
3. Add prediction generation to main loop
4. Enhance Discord alerts with predictions:
   - Daily range forecast in morning alert
   - Price targets in entry alerts
   - Exit timing in status updates

---

## Existing Code to Leverage

| Function | File | Usage |
|----------|------|-------|
| `train_regressor()` | ml_utils.py:922 | Train XGBRegressor for range model |
| `ConformalPredictionWrapper` | conformal_utils.py | Prediction intervals |
| `MarketRegimeDetector` | regime_utils.py | Regime context for predictions |
| `get_options_sentiment()` | polygon_manager.py | PCR data |
| `create_ml_features()` | oscillator_predictor_page.py | Base features |
| ATR, RSI, etc. | indicators.py | Technical features |

---

## Implementation Sequence

### Step 1: price_prediction.py Foundation [COMPLETED]
- Create PriceRangePredictor class
- Create PriceTargetCalculator class
- Create ExitTimingPredictor class
- Add feature engineering functions

### Step 2: Enhance polygon_manager.py [COMPLETED]
- Add get_options_chain() method
- Add calculate_aggregate_iv() method
- Add get_max_pain() method
- Add get_high_oi_strikes() method
- Add get_options_summary() method

### Step 3: Daily Range Tab [IN PROGRESS]
- Add tab to oscillator_predictor_page.py
- Build range feature pipeline
- Train range model with existing train_regressor()
- Add conformal prediction intervals
- Build Streamlit UI with charts

### Step 4: Price Targets Tab [PENDING]
- Implement target calculation methods
- Build risk/reward calculator
- Build Streamlit UI

### Step 5: Exit Timing Tab [PENDING]
- Implement timing prediction
- Build momentum indicators
- Build Streamlit UI

### Step 6: Production Port [PENDING]
- Create velocity_live_trader_2.py
- Integrate predictions into alerts
- Test with live data

---

## Key Design Decisions

1. **Range Prediction:** Use XGBRegressor (not classification) with conformal intervals
2. **IV Calculation:** Weighted average from options chain (ATM + OI weighted)
3. **Target Methods:** Combine ATR, Fibonacci, S/R, and Options data
4. **Exit Timing:** Use oscillator momentum features (velocity, acceleration, exhaustion)
5. **Confidence:** Use ConformalPredictionWrapper for all prediction intervals

---

## Implementation Progress

| Task | Status | Notes |
|------|--------|-------|
| price_prediction.py | COMPLETE | 3 classes with ~1100 lines |
| polygon_manager.py enhancements | COMPLETE | 5 new methods added |
| Daily Range Tab | COMPLETE | XGBoost model with Optuna tuning, prediction charts |
| Price Targets Tab | COMPLETE | ATR, Fibonacci, S/R, and Options targets with R:R calculator |
| Exit Timing Tab | COMPLETE | Momentum analysis, urgency scoring, timing prediction |
| **Full Polygon Integration** | COMPLETE | IV, Max Pain, High OI strikes used in model training |
| velocity_live_trader_2.py | PENDING | Phase 2 - Production port |

## Polygon Data Integration Details (2025-01-06)

### Features Used in Range Model:
- **PCR Volume/OI**: Put/Call ratio (sentiment indicator)
- **IV Weighted**: ATM-weighted implied volatility from options chain
- **IV Call/Put**: Separate call and put IV for skew analysis
- **IV Skew**: Put IV - Call IV (positive = more fear)
- **Max Pain Distance**: Distance from max pain price (attracts price)
- **High OI Strikes Distance**: Distance from highest call/put OI strikes

### UI Enhancements:
- Daily Range tab shows: Price, Range, ATR, PCR, and IV with skew
- Price Targets tab shows: PCR, Max Pain, High Call OI, High Put OI
- All options metrics show % distance from current price

---

## Files Modified/Created (2025-01-06)

### price_prediction.py (NEW - ~975 lines)
- `PriceRangePredictor`: XGBoost model for daily range prediction
- `PriceTargetCalculator`: ATR, Fibonacci, S/R target calculation
- `ExitTimingPredictor`: Momentum-based exit timing analysis

### polygon_manager.py (MODIFIED - +200 lines)
- `get_options_chain()`: Full options chain with Greeks
- `calculate_aggregate_iv()`: Weighted IV calculation
- `get_max_pain()`: Max pain calculation
- `get_high_oi_strikes()`: High OI strike identification
- `get_options_summary()`: Comprehensive options data

### oscillator_predictor_page.py (MODIFIED - +630 lines)
- Added imports for price_prediction module
- Added "Price Prediction Suite" section with 3 tabs:
  1. **Daily Range**: Model training, prediction display, price chart with bands
  2. **Price Targets**: Entry/stop input, target levels, S/R display, chart with lines
  3. **Exit Timing**: Position input, urgency analysis, momentum indicators, multi-panel chart
