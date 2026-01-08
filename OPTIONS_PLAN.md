# Options Trading Builder - Implementation Plan

## Overview
Create a comprehensive **Options Trading Builder** section in `oscillator_predictor_page.py` that:
1. Loads saved strategies from velocity_live_trader.py
2. Saves and loads range predictions from walk-forward analysis
3. Generates options trade recommendations for any ticker
4. Supports multiple strategy types: Single Leg, Vertical Spreads, Straddles, Strangles, Calendar Spreads
5. Full trade tracking with historical P&L

---

## User Requirements (Confirmed)

| Setting | Choice |
|---------|--------|
| Risk Calculation | Max Loss (Options-based) |
| Position Sizing | Fixed contracts (user specifies) |
| Expiration Selection | Multiple DTEs with system suggestions + manual override |
| Trade Tracking | Full tracking with saved recommendations and historical P&L |

---

## Architecture

```
+---------------------------------------------------------------------+
|                     OPTIONS TRADING BUILDER                          |
+---------------------------------------------------------------------+
|                                                                       |
|  [Saved Strategy]          [Range Predictions]     [Live Options]    |
|  velocity_live_trader      walk-forward saves      polygon_manager   |
|         |                         |                       |          |
|  +-------------------------------------------------------------+    |
|  |              OPTIONS STRATEGY ENGINE                         |    |
|  |  - Direction from velocity signals (LONG/SHORT)              |    |
|  |  - Magnitude from range prediction (expected move)           |    |
|  |  - Timing from exit prediction (optimal DTE)                 |    |
|  |  - Volatility from IV analysis (strategy selection)          |    |
|  +-------------------------------------------------------------+    |
|                              |                                       |
|  +-------------------------------------------------------------+    |
|  |              STRATEGY RECOMMENDATIONS                        |    |
|  |  - Single Leg (calls/puts)                                   |    |
|  |  - Vertical Spreads (debit/credit)                          |    |
|  |  - Straddles/Strangles (volatility plays)                   |    |
|  |  - Calendar Spreads (time decay plays)                      |    |
|  +-------------------------------------------------------------+    |
|                              |                                       |
|  +-------------------------------------------------------------+    |
|  |              TRADE TRACKER                                   |    |
|  |  - Saved recommendations with entry criteria                 |    |
|  |  - Position management (open/closed)                        |    |
|  |  - Historical P&L with analytics                            |    |
|  +-------------------------------------------------------------+    |
|                                                                       |
+---------------------------------------------------------------------+
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `options_builder.py` | CREATE | Core options strategy engine (~600 lines) |
| `oscillator_predictor_page.py` | MODIFY | Add Options Builder section with tabs |
| `price_prediction.py` | MODIFY | Add save/load for walk-forward predictions |
| `options_trades.json` | CREATE (runtime) | Trade tracking database |

---

## Phase 1: Create options_builder.py [COMPLETE]

### Classes Implemented:
- **OptionsStrategyEngine** - Core engine for generating recommendations
- **SingleLegStrategy** - Long calls/puts
- **VerticalSpreadStrategy** - Bull/bear call/put spreads (debit and credit)
- **StraddleStrangleStrategy** - Long/short straddles and strangles
- **CalendarSpreadStrategy** - Calendar (horizontal) spreads
- **TradeTracker** - Full trade tracking with P&L

---

## Phase 2: Modify price_prediction.py - Add Save/Load

Add to `PriceRangePredictor`:

```python
def save_predictions(self, predictions: dict, path: str = None):
    """Save walk-forward predictions to JSON"""
    if path is None:
        path = f"predictions/{predictions['ticker']}_range_predictions.json"

    save_data = {
        'ticker': predictions['ticker'],
        'timestamp': datetime.now().isoformat(),
        'model_r2': predictions['r2'],
        'predictions': {
            'predicted_high': predictions['pred_high'],
            'predicted_low': predictions['pred_low'],
            'confidence_interval': predictions['ci_level'],
            'ci_high_upper': predictions['ci_high_upper'],
            'ci_high_lower': predictions['ci_high_lower'],
            'ci_low_upper': predictions['ci_low_upper'],
            'ci_low_lower': predictions['ci_low_lower'],
        },
        'model_params': predictions.get('model_params', {}),
        'feature_importance': predictions.get('feature_importance', {})
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(save_data, f, indent=2)
    return path

def load_predictions(self, path: str) -> dict:
    """Load saved predictions"""
    with open(path, 'r') as f:
        return json.load(f)
```

---

## Phase 3: Add Streamlit UI (oscillator_predictor_page.py)

### Section Location
Add after "Price Prediction Suite" section (around line 6900):

```python
st.markdown("---")
st.header("Options Trading Builder")
```

### Tab Structure

```python
opt_tab1, opt_tab2, opt_tab3, opt_tab4 = st.tabs([
    "Strategy Builder",
    "Trade Recommendations",
    "Position Tracker",
    "Performance"
])
```

### Tab 1: Strategy Builder

```
+---------------------------------------------------------------------+
|  STRATEGY BUILDER                                                    |
+---------------------------------------------------------------------+
|  [Input Row]                                                         |
|  Ticker: [____]  |  Contracts: [__]  |  [Load Strategy] [Load Pred] |
+---------------------------------------------------------------------+
|  [Loaded Data Cards]                                                 |
|  +-------------------------+  +-------------------------------+     |
|  | VELOCITY STRATEGY       |  | RANGE PREDICTION              |     |
|  | Signal: BULLISH         |  | Tomorrow High: $XXX.XX        |     |
|  | Entry: $XXX.XX          |  | Tomorrow Low:  $XXX.XX        |     |
|  | Target: $XXX.XX (+X.X%) |  | Expected Move: X.X%           |     |
|  | Stop: $XXX.XX (-X.X%)   |  | R2: 0.XX                      |     |
|  | Hold Time: ~X days      |  |                               |     |
|  +-------------------------+  +-------------------------------+     |
+---------------------------------------------------------------------+
|  [Market Context]                                                    |
|  Current Price: $XXX.XX | IV Rank: XX% | IV %ile: XX% | Regime: XX  |
+---------------------------------------------------------------------+
|  [DTE Selection]                                                     |
|  System Suggested: * 14 DTE, 21 DTE, 30 DTE                         |
|  [o 7] [* 14] [o 21] [o 30] [o 45] [o 60] [Custom: ____]           |
+---------------------------------------------------------------------+
|  [Strategy Type Selection]                                           |
|  Recommended: * Bull Call Spread (IV Low + Bullish)                 |
|  [o Single Leg] [* Vertical] [o Straddle] [o Strangle] [o Calendar] |
+---------------------------------------------------------------------+
|                        [Generate Recommendations]                    |
+---------------------------------------------------------------------+
```

### Tab 2: Trade Recommendations

Displays generated trade recommendations with:
- Strategy name and type
- All legs with strikes, expirations, premiums
- Entry cost, max profit, max loss
- Break-even price(s)
- Probability of profit
- Risk/reward ratio
- Payoff diagram
- Save/Copy buttons

### Tab 3: Position Tracker

- Open positions with current P&L
- Saved recommendations not yet executed
- Manual trade entry
- Mark closed functionality

### Tab 4: Performance

- Total trades, win rate, avg P&L
- Performance by strategy type
- Performance by ticker
- Equity curve chart
- Trade history table

---

## Implementation Sequence

### Step 1: Create options_builder.py (~600 lines) [COMPLETE]
- OptionsStrategyEngine class
- SingleLegStrategy class
- VerticalSpreadStrategy class
- StraddleStrangleStrategy class
- CalendarSpreadStrategy class
- TradeTracker class

### Step 2: Add prediction save/load to price_prediction.py [COMPLETE]
- save_predictions() method
- load_predictions() method
- list_saved_predictions() method
- Predictions saved to `predictions/{ticker}_range_predictions.json`

### Step 3: Add Options Builder UI to oscillator_predictor_page.py [COMPLETE]
- Added section after Price Prediction Suite (lines 8052-8713)
- Tab 1: Strategy Builder with inputs
- Tab 2: Trade Recommendations display with payoff diagrams
- Tab 3: Position Tracker with execute/close functionality
- Tab 4: Performance Analytics with equity curve

### Step 4: Integration Testing [COMPLETE]
- options_builder.py imports verified
- save/load predictions tested and working
- TradeTracker persistence verified

---

## Key Dependencies (Already Exist)

| Resource | File | Usage |
|----------|------|-------|
| `get_options_chain()` | polygon_manager.py | Live options data |
| `get_max_pain()` | polygon_manager.py | Price magnets |
| `get_atm_iv()` | polygon_manager.py | IV analysis |
| `get_term_structure()` | polygon_manager.py | DTE IV comparison |
| Strategy configs | velocity_live_trader.py | Direction, targets |
| Range predictions | price_prediction.py | Expected moves |
| Exit timing | price_prediction.py | Hold duration |

---

## Trade JSON Schema

```json
{
  "trade_id": "SPY_20260107_001",
  "ticker": "SPY",
  "strategy_type": "vertical_spread",
  "direction": "bullish",
  "created_at": "2026-01-07T08:45:00",
  "status": "open",

  "legs": [
    {"action": "buy", "type": "call", "strike": 590, "expiration": "2026-01-21", "contracts": 2},
    {"action": "sell", "type": "call", "strike": 595, "expiration": "2026-01-21", "contracts": 2}
  ],

  "entry": {
    "price": 2.35,
    "date": "2026-01-07",
    "underlying_price": 589.50
  },

  "targets": {
    "max_profit": 2.65,
    "max_loss": 2.35,
    "break_even": 592.35
  },

  "exit": null,

  "source": {
    "velocity_strategy": "strategies/SPY_Strategy-test_20260105/",
    "range_prediction": "predictions/SPY_range_predictions.json"
  },

  "notes": "User notes here"
}
```

---

## Risk Calculations (Max Loss Based)

| Strategy | Max Loss Calculation |
|----------|---------------------|
| Long Call/Put | Premium paid x contracts x 100 |
| Debit Spread | Net debit x contracts x 100 |
| Credit Spread | (Width - Credit) x contracts x 100 |
| Long Straddle | Total premium x contracts x 100 |
| Long Strangle | Total premium x contracts x 100 |
| Short Straddle | Unlimited (warn user) |
| Calendar Spread | Net debit x contracts x 100 |

---

## DTE Suggestion Logic

```python
def suggest_dte(predicted_hold_days: float, strategy_type: str) -> list[tuple[int, str]]:
    """
    Suggest DTEs based on predicted hold time and strategy type

    Returns: [(dte, reason), ...]
    """
    suggestions = []
    base_dte = int(predicted_hold_days * 1.5)  # 50% buffer

    if strategy_type == 'single_leg':
        suggestions.append((max(14, base_dte), "Minimum for theta protection"))
        suggestions.append((base_dte + 7, "Optimal based on hold time"))
        suggestions.append((45, "LEAPS-lite for reduced theta"))

    elif strategy_type == 'vertical_spread':
        suggestions.append((base_dte, "Matches predicted hold"))
        suggestions.append((base_dte + 7, "Extra buffer"))
        suggestions.append((21, "Standard monthly cycle"))

    elif strategy_type in ['straddle', 'strangle']:
        suggestions.append((14, "High gamma, lower cost"))
        suggestions.append((30, "More time for move"))
        suggestions.append((45, "Reduced theta decay"))

    elif strategy_type == 'calendar_spread':
        suggestions.append((7, "Near leg - front week"))
        suggestions.append((30, "Far leg - monthly"))

    return suggestions
```

---

## Future Enhancement: Overnight Scanner

(Not in current scope - document for later)
- Scan multiple tickers overnight
- Apply velocity signals + range predictions
- Generate batch recommendations
- Morning alert with top picks
