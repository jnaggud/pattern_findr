"""
Options Trading Builder
Generates options trade recommendations based on velocity strategies and range predictions.
Supports: Single Leg, Vertical Spreads, Straddles, Strangles, Calendar Spreads
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class OptionsStrategyEngine:
    """Core engine for generating options trade recommendations"""

    def __init__(self, polygon_manager=None):
        self.polygon = polygon_manager
        self.strategy_types = ['single_leg', 'vertical_spread', 'straddle',
                               'strangle', 'calendar_spread']

        # Strategy generators
        self.single_leg = SingleLegStrategy(polygon_manager)
        self.vertical = VerticalSpreadStrategy(polygon_manager)
        self.straddle_strangle = StraddleStrangleStrategy(polygon_manager)
        self.calendar = CalendarSpreadStrategy(polygon_manager)

    def load_velocity_strategy(self, strategy_path: str) -> Optional[dict]:
        """Load saved strategy config from velocity_live_trader"""
        # Try multiple config file names (velocity_live_trader uses velocity_config.json)
        possible_configs = [
            os.path.join(strategy_path, 'velocity_config.json'),  # velocity_live_trader format
            os.path.join(strategy_path, 'strategy_config.json'),  # old format
        ]

        config_path = None
        for path in possible_configs:
            if os.path.exists(path):
                config_path = path
                break

        if config_path is None:
            # Try just the path directly if it's a JSON file
            if os.path.exists(strategy_path) and strategy_path.endswith('.json'):
                config_path = strategy_path
            else:
                return None

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Extract key trading parameters
            # velocity_live_trader uses bundle_name or strategy_name instead of strategy_id
            strategy_id = config.get('strategy_id') or config.get('bundle_name') or config.get('strategy_name') or 'unknown'
            return {
                'strategy_id': strategy_id,
                'ticker': config.get('ticker', ''),
                'signal_mode': config.get('signal_mode', config.get('signal_type', '')),
                'direction': self._infer_direction(config),
                'parameters': {
                    'oversold_threshold': config.get('oversold_threshold', -0.1),
                    'overbought_threshold': config.get('overbought_threshold', 0.1),
                    'take_profit_pct': config.get('take_profit_pct', 2.0),
                    'stop_loss_pct': config.get('stop_loss_pct', 5.0),
                },
                'backtest_metrics': config.get('backtest_metrics', {}),
                'avg_hold_days': config.get('avg_hold_days',
                                           config.get('backtest_metrics', {}).get('avg_hold_days', 10))
            }
        except Exception as e:
            print(f"Error loading strategy config: {e}")
            return None

    def _infer_direction(self, config: dict) -> str:
        """Infer trading direction from strategy config"""
        # Check explicit direction
        if 'direction' in config:
            return config['direction']

        # Infer from thresholds (more aggressive on one side)
        oversold = abs(config.get('oversold_threshold', -0.1))
        overbought = abs(config.get('overbought_threshold', 0.1))

        if oversold < overbought:
            return 'long'  # More sensitive to oversold = bullish bias
        elif overbought < oversold:
            return 'short'  # More sensitive to overbought = bearish bias
        return 'neutral'

    def load_range_predictions(self, predictions_path: str) -> Optional[dict]:
        """Load saved walk-forward predictions"""
        if not os.path.exists(predictions_path):
            return None

        try:
            with open(predictions_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading predictions: {e}")
            return None

    def get_market_context(self, ticker: str, df: pd.DataFrame = None,
                          current_price: float = None) -> dict:
        """Get current market context for strategy selection"""
        context = {
            'ticker': ticker,
            'current_price': current_price,
            'iv_rank': None,
            'iv_percentile': None,
            'regime': 'unknown',
            'direction': 'neutral',
            'magnitude': 0.0
        }

        if self.polygon is None:
            return context

        try:
            # Get IV data
            iv_data = self.polygon.get_atm_iv(ticker)
            if iv_data:
                context['current_iv'] = iv_data.get('iv', 0)
                context['iv_rank'] = iv_data.get('iv_rank', 50)
                context['iv_percentile'] = iv_data.get('iv_percentile', 50)

            # Get options sentiment
            sentiment = self.polygon.get_options_sentiment(ticker)
            if sentiment:
                context['pcr'] = sentiment.get('pcr', 1.0)
                context['call_volume'] = sentiment.get('call_volume', 0)
                context['put_volume'] = sentiment.get('put_volume', 0)

            # Infer direction from PCR
            if context.get('pcr'):
                if context['pcr'] > 1.2:
                    context['direction'] = 'bearish'  # High put activity
                elif context['pcr'] < 0.8:
                    context['direction'] = 'bullish'  # High call activity

        except Exception as e:
            print(f"Error getting market context: {e}")

        return context

    def select_optimal_strategy(self, context: dict, velocity_strategy: dict = None,
                                range_prediction: dict = None) -> Tuple[str, str]:
        """
        Select best strategy type based on context

        Returns: (strategy_type, reasoning)
        """
        iv_rank = context.get('iv_rank', 50)
        direction = velocity_strategy.get('direction', 'neutral') if velocity_strategy else context.get('direction', 'neutral')

        # Determine IV environment
        iv_high = iv_rank > 60
        iv_low = iv_rank < 40

        # Determine expected move magnitude
        expected_move = 0
        if range_prediction:
            preds = range_prediction.get('predictions', {})
            pred_high = preds.get('predicted_high', 0)
            pred_low = preds.get('predicted_low', 0)
            current = context.get('current_price', 0)
            if current > 0:
                expected_move = ((pred_high - pred_low) / current) * 100

        big_move_expected = expected_move > 2.0  # More than 2% range

        # Strategy selection logic
        if direction in ['long', 'bullish']:
            if iv_high:
                return 'vertical_spread', f"IV Rank {iv_rank}% is elevated - use Bull Put Spread (credit) to benefit from IV crush"
            elif iv_low and big_move_expected:
                return 'single_leg', f"IV Rank {iv_rank}% is low with {expected_move:.1f}% expected move - Long Call for maximum upside"
            else:
                return 'vertical_spread', f"Bull Call Spread balances cost and profit potential"

        elif direction in ['short', 'bearish']:
            if iv_high:
                return 'vertical_spread', f"IV Rank {iv_rank}% is elevated - use Bear Call Spread (credit) to benefit from IV crush"
            elif iv_low and big_move_expected:
                return 'single_leg', f"IV Rank {iv_rank}% is low with {expected_move:.1f}% expected move - Long Put for maximum downside capture"
            else:
                return 'vertical_spread', f"Bear Put Spread balances cost and profit potential"

        else:  # Neutral
            if iv_high:
                return 'straddle', f"IV Rank {iv_rank}% is elevated - Short Straddle to capture IV crush (high risk)"
            elif big_move_expected:
                return 'strangle', f"Expecting {expected_move:.1f}% move - Long Strangle for breakout play"
            else:
                return 'calendar_spread', f"Neutral outlook with moderate IV - Calendar Spread for time decay"

    def calculate_optimal_dte(self, predicted_hold_days: float,
                              strategy_type: str) -> List[Tuple[int, str]]:
        """Calculate optimal DTE based on predicted hold time and strategy"""
        suggestions = []

        # Base DTE = predicted hold + 50% buffer
        base_dte = max(7, int(predicted_hold_days * 1.5))

        if strategy_type == 'single_leg':
            suggestions.append((max(14, base_dte), "Minimum for theta protection"))
            suggestions.append((base_dte + 7, "Optimal based on hold time"))
            suggestions.append((45, "Reduced theta decay"))

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

        # Sort by DTE
        suggestions.sort(key=lambda x: x[0])
        return suggestions

    def generate_recommendations(self, ticker: str, strategy_type: str,
                                 direction: str, current_price: float,
                                 target_price: float, stop_price: float,
                                 dte: int, num_contracts: int,
                                 expected_hold_days: float = 10) -> List[dict]:
        """Generate trade recommendations based on parameters"""
        recommendations = []

        if strategy_type == 'single_leg':
            recs = self.single_leg.generate(
                ticker, direction, current_price, target_price,
                [dte], num_contracts
            )
            recommendations.extend(recs)

        elif strategy_type == 'vertical_spread':
            if direction in ['long', 'bullish']:
                # Bull call spread
                rec = self.vertical.generate_bull_call_spread(
                    ticker, current_price, target_price, dte, num_contracts
                )
                if rec:
                    recommendations.append(rec)

                # Also suggest bull put spread if IV is high
                rec2 = self.vertical.generate_bull_put_spread(
                    ticker, current_price, stop_price, dte, num_contracts
                )
                if rec2:
                    rec2['alternative'] = True
                    recommendations.append(rec2)
            else:
                # Bear put spread
                rec = self.vertical.generate_bear_put_spread(
                    ticker, current_price, target_price, dte, num_contracts
                )
                if rec:
                    recommendations.append(rec)

                # Also suggest bear call spread
                rec2 = self.vertical.generate_bear_call_spread(
                    ticker, current_price, stop_price, dte, num_contracts
                )
                if rec2:
                    rec2['alternative'] = True
                    recommendations.append(rec2)

        elif strategy_type == 'straddle':
            expected_move = abs(target_price - current_price) / current_price
            rec = self.straddle_strangle.generate_long_straddle(
                ticker, current_price, expected_move, dte, num_contracts
            )
            if rec:
                recommendations.append(rec)

        elif strategy_type == 'strangle':
            expected_move = abs(target_price - current_price) / current_price
            rec = self.straddle_strangle.generate_long_strangle(
                ticker, current_price, expected_move, dte, num_contracts
            )
            if rec:
                recommendations.append(rec)

        elif strategy_type == 'calendar_spread':
            far_dte = dte + 21  # Far leg is ~3 weeks further out
            rec = self.calendar.generate_calendar_spread(
                ticker, current_price, dte, far_dte, num_contracts
            )
            if rec:
                recommendations.append(rec)

        return recommendations


class SingleLegStrategy:
    """Single call or put strategy"""

    def __init__(self, polygon_manager=None):
        self.polygon = polygon_manager

    def generate(self, ticker: str, direction: str, current_price: float,
                 target_price: float, dte_options: List[int],
                 num_contracts: int) -> List[dict]:
        """Generate single leg recommendations"""
        recommendations = []

        option_type = 'call' if direction in ['long', 'bullish'] else 'put'

        for dte in dte_options:
            # Calculate strike selection
            # ATM strike
            atm_strike = round(current_price)

            # Slightly OTM strike (1-2% OTM)
            if option_type == 'call':
                otm_strike = round(current_price * 1.01)
                itm_strike = round(current_price * 0.99)
            else:
                otm_strike = round(current_price * 0.99)
                itm_strike = round(current_price * 1.01)

            # Get option prices from Polygon if available
            premium_atm = self._estimate_premium(current_price, atm_strike, dte, option_type)
            premium_otm = self._estimate_premium(current_price, otm_strike, dte, option_type)

            # ATM recommendation
            rec_atm = self._create_recommendation(
                ticker, option_type, atm_strike, dte, num_contracts,
                current_price, target_price, premium_atm, "ATM"
            )
            recommendations.append(rec_atm)

            # OTM recommendation
            rec_otm = self._create_recommendation(
                ticker, option_type, otm_strike, dte, num_contracts,
                current_price, target_price, premium_otm, "Slightly OTM"
            )
            rec_otm['alternative'] = True
            recommendations.append(rec_otm)

        return recommendations

    def _estimate_premium(self, spot: float, strike: float, dte: int,
                          option_type: str) -> float:
        """Estimate option premium (simplified Black-Scholes approximation)"""
        # Simplified premium estimation
        # In production, this would use actual market data from Polygon
        iv = 0.20  # Assume 20% IV as default
        t = dte / 365.0

        # Very simplified ATM approximation
        atm_premium = spot * iv * np.sqrt(t) * 0.4

        # Adjust for moneyness
        moneyness = (spot - strike) / spot if option_type == 'call' else (strike - spot) / spot

        if moneyness > 0:  # ITM
            intrinsic = abs(spot - strike)
            premium = atm_premium + intrinsic * 0.8
        else:  # OTM
            premium = atm_premium * (1 - abs(moneyness) * 2)

        return max(0.05, round(premium, 2))

    def _create_recommendation(self, ticker: str, option_type: str, strike: float,
                               dte: int, num_contracts: int, current_price: float,
                               target_price: float, premium: float,
                               strike_type: str) -> dict:
        """Create a single leg recommendation"""
        expiration = (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d')

        # Calculate metrics
        max_loss = premium * num_contracts * 100

        if option_type == 'call':
            break_even = strike + premium
            profit_at_target = max(0, target_price - strike - premium) * num_contracts * 100
        else:
            break_even = strike - premium
            profit_at_target = max(0, strike - target_price - premium) * num_contracts * 100

        # Estimate probability of profit from delta (simplified)
        delta = self._estimate_delta(current_price, strike, dte, option_type)
        prob_profit = delta * 100 if option_type == 'call' else (1 - delta) * 100

        return {
            'strategy_type': 'single_leg',
            'strategy_name': f"Long {option_type.title()} ({strike_type})",
            'ticker': ticker,
            'direction': 'bullish' if option_type == 'call' else 'bearish',
            'legs': [
                {
                    'action': 'buy',
                    'type': option_type,
                    'strike': strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': premium
                }
            ],
            'entry_cost': premium,
            'max_profit': 'unlimited' if option_type == 'call' else strike - premium,
            'max_loss': premium,
            'max_loss_dollars': max_loss,
            'break_even': break_even,
            'prob_profit': round(prob_profit, 1),
            'profit_at_target': profit_at_target,
            'dte': dte,
            'risk_reward': round(profit_at_target / max_loss, 2) if max_loss > 0 else 0,
            'notes': f"{strike_type} {option_type} targeting ${target_price:.2f}"
        }

    def _estimate_delta(self, spot: float, strike: float, dte: int,
                        option_type: str) -> float:
        """Estimate option delta (simplified)"""
        moneyness = (spot - strike) / spot
        t = dte / 365.0

        # Simplified delta approximation
        if option_type == 'call':
            if moneyness > 0.05:  # Deep ITM
                return 0.85
            elif moneyness > 0:  # Slightly ITM
                return 0.6
            elif moneyness > -0.02:  # ATM
                return 0.5
            elif moneyness > -0.05:  # Slightly OTM
                return 0.35
            else:  # Deep OTM
                return 0.15
        else:  # Put
            call_delta = self._estimate_delta(spot, strike, dte, 'call')
            return call_delta - 1


class VerticalSpreadStrategy:
    """Bull/Bear call/put spreads"""

    def __init__(self, polygon_manager=None):
        self.polygon = polygon_manager

    def generate_bull_call_spread(self, ticker: str, current_price: float,
                                   target_price: float, dte: int,
                                   num_contracts: int) -> dict:
        """Generate bull call spread (debit)"""
        # Buy lower strike call, sell higher strike call
        buy_strike = round(current_price)  # ATM
        sell_strike = round(min(target_price, current_price * 1.05))  # Target or 5% OTM

        # Ensure minimum spread width
        if sell_strike <= buy_strike:
            sell_strike = buy_strike + 5

        spread_width = sell_strike - buy_strike

        # Estimate premiums
        buy_premium = self._estimate_premium(current_price, buy_strike, dte, 'call')
        sell_premium = self._estimate_premium(current_price, sell_strike, dte, 'call')

        net_debit = buy_premium - sell_premium
        max_profit = spread_width - net_debit
        max_loss = net_debit
        break_even = buy_strike + net_debit

        expiration = (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d')

        return {
            'strategy_type': 'vertical_spread',
            'strategy_name': f"Bull Call Spread ${buy_strike}/${sell_strike}",
            'ticker': ticker,
            'direction': 'bullish',
            'spread_type': 'debit',
            'legs': [
                {
                    'action': 'buy',
                    'type': 'call',
                    'strike': buy_strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': buy_premium
                },
                {
                    'action': 'sell',
                    'type': 'call',
                    'strike': sell_strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': sell_premium
                }
            ],
            'entry_cost': net_debit,
            'max_profit': max_profit,
            'max_profit_dollars': max_profit * num_contracts * 100,
            'max_loss': max_loss,
            'max_loss_dollars': max_loss * num_contracts * 100,
            'break_even': break_even,
            'spread_width': spread_width,
            'dte': dte,
            'risk_reward': round(max_profit / max_loss, 2) if max_loss > 0 else 0,
            'prob_profit': self._estimate_pop(current_price, break_even, dte, 'above'),
            'notes': f"Max profit if {ticker} closes above ${sell_strike} at expiration"
        }

    def generate_bear_put_spread(self, ticker: str, current_price: float,
                                  target_price: float, dte: int,
                                  num_contracts: int) -> dict:
        """Generate bear put spread (debit)"""
        buy_strike = round(current_price)  # ATM
        sell_strike = round(max(target_price, current_price * 0.95))  # Target or 5% below

        if sell_strike >= buy_strike:
            sell_strike = buy_strike - 5

        spread_width = buy_strike - sell_strike

        buy_premium = self._estimate_premium(current_price, buy_strike, dte, 'put')
        sell_premium = self._estimate_premium(current_price, sell_strike, dte, 'put')

        net_debit = buy_premium - sell_premium
        max_profit = spread_width - net_debit
        max_loss = net_debit
        break_even = buy_strike - net_debit

        expiration = (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d')

        return {
            'strategy_type': 'vertical_spread',
            'strategy_name': f"Bear Put Spread ${buy_strike}/${sell_strike}",
            'ticker': ticker,
            'direction': 'bearish',
            'spread_type': 'debit',
            'legs': [
                {
                    'action': 'buy',
                    'type': 'put',
                    'strike': buy_strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': buy_premium
                },
                {
                    'action': 'sell',
                    'type': 'put',
                    'strike': sell_strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': sell_premium
                }
            ],
            'entry_cost': net_debit,
            'max_profit': max_profit,
            'max_profit_dollars': max_profit * num_contracts * 100,
            'max_loss': max_loss,
            'max_loss_dollars': max_loss * num_contracts * 100,
            'break_even': break_even,
            'spread_width': spread_width,
            'dte': dte,
            'risk_reward': round(max_profit / max_loss, 2) if max_loss > 0 else 0,
            'prob_profit': self._estimate_pop(current_price, break_even, dte, 'below'),
            'notes': f"Max profit if {ticker} closes below ${sell_strike} at expiration"
        }

    def generate_bull_put_spread(self, ticker: str, current_price: float,
                                  support_price: float, dte: int,
                                  num_contracts: int) -> dict:
        """Generate bull put spread (credit)"""
        sell_strike = round(min(support_price, current_price * 0.97))  # At support or 3% below
        buy_strike = sell_strike - 5  # $5 wide spread

        sell_premium = self._estimate_premium(current_price, sell_strike, dte, 'put')
        buy_premium = self._estimate_premium(current_price, buy_strike, dte, 'put')

        net_credit = sell_premium - buy_premium
        spread_width = sell_strike - buy_strike
        max_profit = net_credit
        max_loss = spread_width - net_credit
        break_even = sell_strike - net_credit

        expiration = (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d')

        return {
            'strategy_type': 'vertical_spread',
            'strategy_name': f"Bull Put Spread ${sell_strike}/${buy_strike}",
            'ticker': ticker,
            'direction': 'bullish',
            'spread_type': 'credit',
            'legs': [
                {
                    'action': 'sell',
                    'type': 'put',
                    'strike': sell_strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': sell_premium
                },
                {
                    'action': 'buy',
                    'type': 'put',
                    'strike': buy_strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': buy_premium
                }
            ],
            'entry_cost': -net_credit,  # Negative = credit received
            'max_profit': max_profit,
            'max_profit_dollars': max_profit * num_contracts * 100,
            'max_loss': max_loss,
            'max_loss_dollars': max_loss * num_contracts * 100,
            'break_even': break_even,
            'spread_width': spread_width,
            'dte': dte,
            'risk_reward': round(max_profit / max_loss, 2) if max_loss > 0 else 0,
            'prob_profit': self._estimate_pop(current_price, break_even, dte, 'above'),
            'notes': f"Keep credit if {ticker} stays above ${sell_strike} at expiration"
        }

    def generate_bear_call_spread(self, ticker: str, current_price: float,
                                   resistance_price: float, dte: int,
                                   num_contracts: int) -> dict:
        """Generate bear call spread (credit)"""
        sell_strike = round(max(resistance_price, current_price * 1.03))  # At resistance or 3% above
        buy_strike = sell_strike + 5  # $5 wide spread

        sell_premium = self._estimate_premium(current_price, sell_strike, dte, 'call')
        buy_premium = self._estimate_premium(current_price, buy_strike, dte, 'call')

        net_credit = sell_premium - buy_premium
        spread_width = buy_strike - sell_strike
        max_profit = net_credit
        max_loss = spread_width - net_credit
        break_even = sell_strike + net_credit

        expiration = (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d')

        return {
            'strategy_type': 'vertical_spread',
            'strategy_name': f"Bear Call Spread ${sell_strike}/${buy_strike}",
            'ticker': ticker,
            'direction': 'bearish',
            'spread_type': 'credit',
            'legs': [
                {
                    'action': 'sell',
                    'type': 'call',
                    'strike': sell_strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': sell_premium
                },
                {
                    'action': 'buy',
                    'type': 'call',
                    'strike': buy_strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': buy_premium
                }
            ],
            'entry_cost': -net_credit,
            'max_profit': max_profit,
            'max_profit_dollars': max_profit * num_contracts * 100,
            'max_loss': max_loss,
            'max_loss_dollars': max_loss * num_contracts * 100,
            'break_even': break_even,
            'spread_width': spread_width,
            'dte': dte,
            'risk_reward': round(max_profit / max_loss, 2) if max_loss > 0 else 0,
            'prob_profit': self._estimate_pop(current_price, break_even, dte, 'below'),
            'notes': f"Keep credit if {ticker} stays below ${sell_strike} at expiration"
        }

    def _estimate_premium(self, spot: float, strike: float, dte: int,
                          option_type: str) -> float:
        """Estimate option premium"""
        iv = 0.20
        t = dte / 365.0
        atm_premium = spot * iv * np.sqrt(t) * 0.4

        moneyness = (spot - strike) / spot if option_type == 'call' else (strike - spot) / spot

        if moneyness > 0:
            intrinsic = abs(spot - strike)
            premium = atm_premium + intrinsic * 0.8
        else:
            premium = atm_premium * max(0.1, 1 - abs(moneyness) * 3)

        return max(0.05, round(premium, 2))

    def _estimate_pop(self, spot: float, break_even: float, dte: int,
                      direction: str) -> float:
        """Estimate probability of profit"""
        # Simplified - assumes normal distribution
        distance = (break_even - spot) / spot

        # Rough probability based on distance
        if direction == 'above':
            if distance > 0.05:
                return 35.0
            elif distance > 0.02:
                return 45.0
            elif distance > 0:
                return 48.0
            elif distance > -0.02:
                return 55.0
            else:
                return 65.0
        else:  # below
            if distance < -0.05:
                return 35.0
            elif distance < -0.02:
                return 45.0
            elif distance < 0:
                return 48.0
            elif distance < 0.02:
                return 55.0
            else:
                return 65.0


class StraddleStrangleStrategy:
    """Straddle and strangle strategies"""

    def __init__(self, polygon_manager=None):
        self.polygon = polygon_manager

    def generate_long_straddle(self, ticker: str, current_price: float,
                                expected_move: float, dte: int,
                                num_contracts: int) -> dict:
        """Long straddle for expected big move"""
        strike = round(current_price)

        call_premium = self._estimate_premium(current_price, strike, dte, 'call')
        put_premium = self._estimate_premium(current_price, strike, dte, 'put')

        total_premium = call_premium + put_premium
        break_even_up = strike + total_premium
        break_even_down = strike - total_premium

        expiration = (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d')

        return {
            'strategy_type': 'straddle',
            'strategy_name': f"Long Straddle ${strike}",
            'ticker': ticker,
            'direction': 'neutral',
            'legs': [
                {
                    'action': 'buy',
                    'type': 'call',
                    'strike': strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': call_premium
                },
                {
                    'action': 'buy',
                    'type': 'put',
                    'strike': strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': put_premium
                }
            ],
            'entry_cost': total_premium,
            'max_profit': 'unlimited',
            'max_loss': total_premium,
            'max_loss_dollars': total_premium * num_contracts * 100,
            'break_even_up': break_even_up,
            'break_even_down': break_even_down,
            'break_even_move_pct': (total_premium / current_price) * 100,
            'dte': dte,
            'notes': f"Profit if {ticker} moves more than {(total_premium/current_price)*100:.1f}% in either direction"
        }

    def generate_long_strangle(self, ticker: str, current_price: float,
                                expected_move: float, dte: int,
                                num_contracts: int) -> dict:
        """Long strangle - cheaper than straddle, needs bigger move"""
        call_strike = round(current_price * 1.02)  # 2% OTM
        put_strike = round(current_price * 0.98)   # 2% OTM

        call_premium = self._estimate_premium(current_price, call_strike, dte, 'call')
        put_premium = self._estimate_premium(current_price, put_strike, dte, 'put')

        total_premium = call_premium + put_premium
        break_even_up = call_strike + total_premium
        break_even_down = put_strike - total_premium

        expiration = (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d')

        return {
            'strategy_type': 'strangle',
            'strategy_name': f"Long Strangle ${put_strike}/${call_strike}",
            'ticker': ticker,
            'direction': 'neutral',
            'legs': [
                {
                    'action': 'buy',
                    'type': 'call',
                    'strike': call_strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': call_premium
                },
                {
                    'action': 'buy',
                    'type': 'put',
                    'strike': put_strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': put_premium
                }
            ],
            'entry_cost': total_premium,
            'max_profit': 'unlimited',
            'max_loss': total_premium,
            'max_loss_dollars': total_premium * num_contracts * 100,
            'break_even_up': break_even_up,
            'break_even_down': break_even_down,
            'break_even_move_pct': ((break_even_up - current_price) / current_price) * 100,
            'dte': dte,
            'notes': f"Cheaper than straddle but needs bigger move to profit"
        }

    def generate_short_straddle(self, ticker: str, current_price: float,
                                 expected_range: float, dte: int,
                                 num_contracts: int) -> dict:
        """Short straddle for expected low volatility (HIGH RISK)"""
        strike = round(current_price)

        call_premium = self._estimate_premium(current_price, strike, dte, 'call')
        put_premium = self._estimate_premium(current_price, strike, dte, 'put')

        total_credit = call_premium + put_premium
        break_even_up = strike + total_credit
        break_even_down = strike - total_credit

        expiration = (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d')

        return {
            'strategy_type': 'straddle',
            'strategy_name': f"Short Straddle ${strike} (HIGH RISK)",
            'ticker': ticker,
            'direction': 'neutral',
            'is_short': True,
            'legs': [
                {
                    'action': 'sell',
                    'type': 'call',
                    'strike': strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': call_premium
                },
                {
                    'action': 'sell',
                    'type': 'put',
                    'strike': strike,
                    'expiration': expiration,
                    'contracts': num_contracts,
                    'premium': put_premium
                }
            ],
            'entry_cost': -total_credit,  # Credit received
            'max_profit': total_credit,
            'max_profit_dollars': total_credit * num_contracts * 100,
            'max_loss': 'unlimited',
            'max_loss_dollars': 'UNLIMITED - MARGIN REQUIRED',
            'break_even_up': break_even_up,
            'break_even_down': break_even_down,
            'dte': dte,
            'risk_warning': 'UNLIMITED RISK - Only for experienced traders with margin approval',
            'notes': f"Max profit if {ticker} stays at ${strike} through expiration"
        }

    def _estimate_premium(self, spot: float, strike: float, dte: int,
                          option_type: str) -> float:
        """Estimate option premium"""
        iv = 0.20
        t = dte / 365.0
        atm_premium = spot * iv * np.sqrt(t) * 0.4

        moneyness = (spot - strike) / spot if option_type == 'call' else (strike - spot) / spot

        if moneyness > 0:
            intrinsic = abs(spot - strike)
            premium = atm_premium + intrinsic * 0.8
        else:
            premium = atm_premium * max(0.1, 1 - abs(moneyness) * 3)

        return max(0.05, round(premium, 2))


class CalendarSpreadStrategy:
    """Calendar (horizontal) spreads"""

    def __init__(self, polygon_manager=None):
        self.polygon = polygon_manager

    def generate_calendar_spread(self, ticker: str, current_price: float,
                                  near_dte: int, far_dte: int,
                                  num_contracts: int) -> dict:
        """Calendar spread for time decay capture"""
        strike = round(current_price)  # ATM

        # Sell near-dated, buy far-dated
        near_premium = self._estimate_premium(current_price, strike, near_dte, 'call')
        far_premium = self._estimate_premium(current_price, strike, far_dte, 'call')

        net_debit = far_premium - near_premium

        near_exp = (datetime.now() + timedelta(days=near_dte)).strftime('%Y-%m-%d')
        far_exp = (datetime.now() + timedelta(days=far_dte)).strftime('%Y-%m-%d')

        return {
            'strategy_type': 'calendar_spread',
            'strategy_name': f"Calendar Spread ${strike} ({near_dte}/{far_dte} DTE)",
            'ticker': ticker,
            'direction': 'neutral',
            'legs': [
                {
                    'action': 'sell',
                    'type': 'call',
                    'strike': strike,
                    'expiration': near_exp,
                    'contracts': num_contracts,
                    'premium': near_premium
                },
                {
                    'action': 'buy',
                    'type': 'call',
                    'strike': strike,
                    'expiration': far_exp,
                    'contracts': num_contracts,
                    'premium': far_premium
                }
            ],
            'entry_cost': net_debit,
            'max_profit': 'Varies (max when price at strike at near expiration)',
            'max_loss': net_debit,
            'max_loss_dollars': net_debit * num_contracts * 100,
            'near_dte': near_dte,
            'far_dte': far_dte,
            'notes': f"Profits from time decay differential. Best if {ticker} stays near ${strike}"
        }

    def _estimate_premium(self, spot: float, strike: float, dte: int,
                          option_type: str) -> float:
        """Estimate option premium"""
        iv = 0.20
        t = dte / 365.0
        atm_premium = spot * iv * np.sqrt(t) * 0.4
        return max(0.05, round(atm_premium, 2))


class TradeTracker:
    """Full trade tracking with P&L"""

    def __init__(self, db_path: str = "options_trades.json"):
        self.db_path = db_path
        self.trades = self._load_trades()

    def _load_trades(self) -> dict:
        """Load trades from JSON file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except:
                return {'trades': [], 'next_id': 1}
        return {'trades': [], 'next_id': 1}

    def _save_trades(self):
        """Save trades to JSON file"""
        with open(self.db_path, 'w') as f:
            json.dump(self.trades, f, indent=2, default=str)

    def save_recommendation(self, trade: dict) -> str:
        """Save a trade recommendation, returns trade_id"""
        trade_id = f"{trade.get('ticker', 'UNK')}_{datetime.now().strftime('%Y%m%d')}_{self.trades['next_id']:03d}"

        trade_record = {
            'trade_id': trade_id,
            'created_at': datetime.now().isoformat(),
            'status': 'saved',  # saved, executed, closed
            **trade,
            'execution': None,
            'close': None,
            'pnl': None
        }

        self.trades['trades'].append(trade_record)
        self.trades['next_id'] += 1
        self._save_trades()

        return trade_id

    def mark_executed(self, trade_id: str, execution_details: dict) -> bool:
        """Mark trade as executed with fill price"""
        for trade in self.trades['trades']:
            if trade['trade_id'] == trade_id:
                trade['status'] = 'executed'
                trade['execution'] = {
                    'date': datetime.now().isoformat(),
                    'fill_price': execution_details.get('fill_price'),
                    'underlying_price': execution_details.get('underlying_price'),
                    'notes': execution_details.get('notes', '')
                }
                self._save_trades()
                return True
        return False

    def mark_closed(self, trade_id: str, close_details: dict) -> bool:
        """Mark trade as closed, calculate P&L"""
        for trade in self.trades['trades']:
            if trade['trade_id'] == trade_id:
                trade['status'] = 'closed'
                trade['close'] = {
                    'date': datetime.now().isoformat(),
                    'close_price': close_details.get('close_price'),
                    'underlying_price': close_details.get('underlying_price'),
                    'notes': close_details.get('notes', '')
                }

                # Calculate P&L
                entry_price = trade.get('execution', {}).get('fill_price') or trade.get('entry_cost', 0)
                close_price = close_details.get('close_price', 0)
                contracts = trade.get('legs', [{}])[0].get('contracts', 1)

                # For credits (negative entry), profit = entry - close
                # For debits (positive entry), profit = close - entry
                if entry_price < 0:  # Credit spread
                    pnl_per_contract = abs(entry_price) - close_price
                else:  # Debit spread
                    pnl_per_contract = close_price - entry_price

                trade['pnl'] = {
                    'per_contract': round(pnl_per_contract, 2),
                    'total': round(pnl_per_contract * contracts * 100, 2),
                    'pct_return': round((pnl_per_contract / abs(entry_price)) * 100, 2) if entry_price != 0 else 0
                }

                self._save_trades()
                return True
        return False

    def delete_trade(self, trade_id: str) -> bool:
        """Delete a saved trade"""
        for i, trade in enumerate(self.trades['trades']):
            if trade['trade_id'] == trade_id:
                del self.trades['trades'][i]
                self._save_trades()
                return True
        return False

    def get_trade(self, trade_id: str) -> Optional[dict]:
        """Get a specific trade by ID"""
        for trade in self.trades['trades']:
            if trade['trade_id'] == trade_id:
                return trade
        return None

    def get_open_positions(self) -> List[dict]:
        """Get all open positions"""
        return [t for t in self.trades['trades'] if t['status'] == 'executed']

    def get_saved_recommendations(self) -> List[dict]:
        """Get saved but not executed recommendations"""
        return [t for t in self.trades['trades'] if t['status'] == 'saved']

    def get_closed_trades(self) -> List[dict]:
        """Get all closed trades"""
        return [t for t in self.trades['trades'] if t['status'] == 'closed']

    def get_historical_trades(self, ticker: str = None) -> List[dict]:
        """Get historical trades with P&L"""
        trades = [t for t in self.trades['trades'] if t['status'] == 'closed']
        if ticker:
            trades = [t for t in trades if t.get('ticker') == ticker]
        return sorted(trades, key=lambda x: x.get('close', {}).get('date', ''), reverse=True)

    def get_performance_stats(self, ticker: str = None) -> dict:
        """Calculate win rate, avg P&L, total P&L, etc."""
        closed = self.get_historical_trades(ticker)

        if not closed:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'by_strategy': {},
                'by_ticker': {}
            }

        pnls = [t.get('pnl', {}).get('total', 0) for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # By strategy type
        by_strategy = {}
        for trade in closed:
            st = trade.get('strategy_type', 'unknown')
            if st not in by_strategy:
                by_strategy[st] = {'trades': 0, 'wins': 0, 'pnl': 0}
            by_strategy[st]['trades'] += 1
            by_strategy[st]['pnl'] += trade.get('pnl', {}).get('total', 0)
            if trade.get('pnl', {}).get('total', 0) > 0:
                by_strategy[st]['wins'] += 1

        # By ticker
        by_ticker = {}
        for trade in closed:
            tk = trade.get('ticker', 'unknown')
            if tk not in by_ticker:
                by_ticker[tk] = {'trades': 0, 'wins': 0, 'pnl': 0}
            by_ticker[tk]['trades'] += 1
            by_ticker[tk]['pnl'] += trade.get('pnl', {}).get('total', 0)
            if trade.get('pnl', {}).get('total', 0) > 0:
                by_ticker[tk]['wins'] += 1

        return {
            'total_trades': len(closed),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round(len(wins) / len(closed) * 100, 1) if closed else 0,
            'avg_pnl': round(sum(pnls) / len(pnls), 2) if pnls else 0,
            'total_pnl': round(sum(pnls), 2),
            'best_trade': round(max(pnls), 2) if pnls else 0,
            'worst_trade': round(min(pnls), 2) if pnls else 0,
            'by_strategy': by_strategy,
            'by_ticker': by_ticker
        }


# Utility functions
def format_recommendation_for_display(rec: dict) -> str:
    """Format a recommendation for display"""
    lines = []
    lines.append(f"**{rec.get('strategy_name', 'Unknown Strategy')}**")
    lines.append(f"Ticker: {rec.get('ticker', 'N/A')} | Direction: {rec.get('direction', 'N/A')}")
    lines.append("")

    # Legs
    for leg in rec.get('legs', []):
        action = leg.get('action', '').upper()
        opt_type = leg.get('type', '').upper()
        strike = leg.get('strike', 0)
        exp = leg.get('expiration', 'N/A')
        contracts = leg.get('contracts', 1)
        premium = leg.get('premium', 0)
        lines.append(f"  {action} {contracts}x ${strike} {opt_type} @ ${premium:.2f} (Exp: {exp})")

    lines.append("")
    lines.append(f"Entry Cost: ${rec.get('entry_cost', 0):.2f}")
    lines.append(f"Max Profit: ${rec.get('max_profit', 'N/A')}")
    lines.append(f"Max Loss: ${rec.get('max_loss_dollars', 'N/A')}")

    if 'break_even' in rec:
        lines.append(f"Break-even: ${rec.get('break_even', 0):.2f}")
    elif 'break_even_up' in rec:
        lines.append(f"Break-even Up: ${rec.get('break_even_up', 0):.2f}")
        lines.append(f"Break-even Down: ${rec.get('break_even_down', 0):.2f}")

    if 'prob_profit' in rec:
        lines.append(f"Prob of Profit: {rec.get('prob_profit', 0):.1f}%")

    if 'risk_reward' in rec:
        lines.append(f"Risk/Reward: 1:{rec.get('risk_reward', 0):.2f}")

    if 'notes' in rec:
        lines.append("")
        lines.append(f"Notes: {rec.get('notes', '')}")

    return "\n".join(lines)


def create_payoff_data(rec: dict, price_range: Tuple[float, float] = None,
                       num_points: int = 100) -> pd.DataFrame:
    """Create payoff diagram data for a recommendation"""
    legs = rec.get('legs', [])
    if not legs:
        return pd.DataFrame()

    # Determine price range
    strikes = [leg.get('strike', 0) for leg in legs]
    if price_range is None:
        min_strike = min(strikes)
        max_strike = max(strikes)
        spread = max_strike - min_strike if max_strike != min_strike else max_strike * 0.1
        price_range = (min_strike - spread * 2, max_strike + spread * 2)

    prices = np.linspace(price_range[0], price_range[1], num_points)
    payoffs = []

    for price in prices:
        total_payoff = 0
        for leg in legs:
            strike = leg.get('strike', 0)
            premium = leg.get('premium', 0)
            contracts = leg.get('contracts', 1)
            opt_type = leg.get('type', 'call')
            action = leg.get('action', 'buy')

            # Calculate intrinsic value at expiration
            if opt_type == 'call':
                intrinsic = max(0, price - strike)
            else:
                intrinsic = max(0, strike - price)

            # Calculate P&L for this leg
            if action == 'buy':
                leg_pnl = (intrinsic - premium) * contracts * 100
            else:  # sell
                leg_pnl = (premium - intrinsic) * contracts * 100

            total_payoff += leg_pnl

        payoffs.append(total_payoff)

    return pd.DataFrame({
        'price': prices,
        'payoff': payoffs
    })
