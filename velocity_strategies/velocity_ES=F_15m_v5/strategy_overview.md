# ES Futures 15-Minute Strategy Performance Summary

## Overview

This document summarizes the backtested performance of a proprietary mean-reversion strategy for E-mini S&P 500 futures (ES) operating on 15-minute timeframes.

---

## Risk Profile

| Metric | Value | Notes |
|--------|-------|-------|
| **Stop Loss** | Wide | Designed as tail-risk protection |
| **Take Profit** | Defined | Allows winners to run |
| **Avg Intra-Trade Risk (MAE)** | 0.08% | ~5 points on ES |
| **Max Intra-Trade Risk (MAE)** | 0.60% | ~36 points on ES |
| **Max Portfolio Drawdown** | 0.31% | Peak-to-trough |

### Risk Management Notes

The strategy employs multiple exit mechanisms that typically close positions well before the stop loss level. Across 500+ backtested trades, the hard stop was never triggered—all exits occurred through the strategy's dynamic exit rules.

MAE (Maximum Adverse Excursion) analysis shows:
- 80% of trades experience less than 0.1% adverse movement
- Average trade moves against position by only ~5 points before closing
- Position sizing should be based on observed MAE range, not theoretical stop distance

---

## Backtested Performance

Testing methodology: Walk-forward validation with out-of-sample verification.

| Metric | Training Period | Test Period (Out-of-Sample) |
|--------|-----------------|----------------------------|
| Duration | 61 days | 13 days |
| Total Return | 73.7% | 14.3% |
| Daily Return | 1.21% | 1.10% |
| Win Rate | 83.2% | 79.8% |
| Profit Factor | 15.77 | 14.40 |
| Trade Count | 441 | 104 |

---

## Trade Efficiency Analysis

| Metric | Value |
|--------|-------|
| Avg MAE | 0.08% (~5 pts) |
| Max MAE | 0.60% (~36 pts) |
| Avg MFE | 0.18% (~11 pts) |
| Edge Ratio (MFE/MAE) | 2.2x |
| Avg Trade Efficiency | 62% |

The 2.2x edge ratio indicates favorable price movement is consistently larger than adverse movement, confirming precise entry timing.

---

## Trading Characteristics

- **Frequency**: Approximately 8 trades per day during regular trading hours
- **Typical Hold Time**: Short duration, most positions closed within 1-3 bars
- **Market Conditions**: Designed for mean-reverting price action

---

## Cost Considerations

Based on typical retail commission ($4.50 RT) and realistic slippage assumptions:

| Scenario | Costs as % of Gross | Net Retained |
|----------|---------------------|--------------|
| 1-Tick Slippage | ~8% | ~92% |
| 2-Tick Slippage | ~14% | ~86% |

---

*Performance based on backtested results. Past performance does not guarantee future results. Futures trading involves substantial risk.*
