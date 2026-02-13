# Stop Loss Analysis - ES=F

## Strategy: velocity_ES=F_15m_v5
**Analysis Date:** 2026-02-03
**Data Period:** 2025-11-23 to 2026-02-03
**Current Price:** 7005.25

---

## Performance Summary

| Stop (points) | Stop (%) | Trades | Win Rate | Total Return | P&L (points) | P&L ($) | Max DD (points) | Profit Factor |
|---------|----------|--------|----------|--------------|---------|---------|----------|---------------|
| 5.0 | 0.07% | 529 | 73.3% | 57.97% | 4060.8 | $203,040 | 39.9 | 7.93 |
| 10.0 | 0.14% | 528 | 78.4% | 64.98% | 4551.8 | $227,588 | 39.9 | 8.30 |
| 20.0 | 0.29% | 527 | 80.5% | 70.50% | 4938.8 | $246,939 | 59.8 | 8.58 |
| 36.0 | 0.51% | 524 | 80.9% | 66.75% | 4675.7 | $233,786 | 107.4 | 6.68 |

---

## Key Findings

### Best Total Return: 20.0pt stop
- Return: 70.50%
- Win Rate: 80.5%
- Profit Factor: 8.58

### Best Win Rate: 36.0pt stop
- Win Rate: 80.9%
- Return: 66.75%

### Best Risk-Adjusted: 10.0pt stop
- Return/Drawdown Ratio: 114.03

---

## Comparison vs Widest Stop (36.0pt)

### 5.0pt vs 36.0pt
- Return difference: -8.78%
- Win rate difference: -7.6%
- Performance retention: 87%

### 10.0pt vs 36.0pt
- Return difference: -1.77%
- Win rate difference: -2.5%
- Performance retention: 97%

### 20.0pt vs 36.0pt
- Return difference: +3.76%
- Win rate difference: -0.5%
- Performance retention: 106%

---

## MAE Analysis

| Stop (points) | Avg MAE (%) | Max MAE (%) | Stop Triggered % |
|---------|-------------|-------------|------------------|
| 5.0 | 0.03% | 0.07% | - |
| 10.0 | 0.03% | 0.14% | - |
| 20.0 | 0.04% | 0.29% | - |
| 36.0 | 0.05% | 0.51% | - |

---

## Recommendations

Based on this analysis:

1. **Tightest Viable Stop**: The tightest stop that maintains positive returns
2. **Optimal Balance**: Best return-to-drawdown ratio
3. **Conservative**: Wider stop for maximum trade preservation

---

*Analysis based on backtested results. Past performance does not guarantee future results.*
