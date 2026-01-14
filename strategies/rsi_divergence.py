"""
RSI Divergence Strategy - Mean Reversion with Divergence Detection

Identifies bullish and bearish divergences between price and RSI.
Excellent for catching reversals at key levels.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategyConfig, SignalType, TPLevel


class RSIDivergenceStrategy(BaseStrategy):
    """
    RSI Divergence Trading Strategy

    Entry Logic:
    - BUY: Bullish divergence (price makes lower low, RSI makes higher low)
    - SELL: Bearish divergence (price makes higher high, RSI makes lower high)

    Exit Logic:
    - ATR-based stop loss
    - RSI-based take profits
    """

    @property
    def name(self) -> str:
        return "rsi_divergence"

    @property
    def description(self) -> str:
        return "RSI divergence strategy for catching reversals at key levels"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            # RSI parameters
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,

            # Divergence detection
            'lookback_period': 20,  # Bars to look back for pivots
            'pivot_strength': 3,    # Bars on each side for pivot

            # ATR for SL/TP
            'atr_period': 14,
            'atr_mult_sl': 2.0,
            'atr_mult_tp': [1.5, 2.5, 3.5, 4.5, 5.5, 7.0],

            # Filters
            'require_extreme_rsi': True,  # RSI must be in OB/OS zone
            'use_volume_confirm': True,
            'volume_mult': 1.2,
        }

    @property
    def params_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rsi_period": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 30,
                    "description": "RSI calculation period"
                },
                "rsi_overbought": {
                    "type": "integer",
                    "minimum": 60,
                    "maximum": 90,
                    "description": "RSI overbought level"
                },
                "rsi_oversold": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 40,
                    "description": "RSI oversold level"
                },
                "lookback_period": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 50,
                    "description": "Lookback for divergence detection"
                },
                "pivot_strength": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 10,
                    "description": "Bars for pivot detection"
                },
                "atr_mult_sl": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 5.0,
                    "description": "ATR multiplier for stop loss"
                },
                "require_extreme_rsi": {
                    "type": "boolean",
                    "description": "Require RSI in OB/OS zone"
                }
            }
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and detect pivots"""

        # RSI
        df['rsi'] = self.rsi(df['close'], self.params['rsi_period'])

        # ATR
        df['atr'] = self.atr(df, self.params['atr_period'])

        # Volume
        if self.params['use_volume_confirm']:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Detect price pivots (local highs and lows)
        strength = self.params['pivot_strength']

        # Price pivots
        df['price_pivot_high'] = False
        df['price_pivot_low'] = False

        # RSI pivots
        df['rsi_pivot_high'] = False
        df['rsi_pivot_low'] = False

        # Find local extrema using rolling window
        for i in range(strength, len(df) - strength):
            # Price pivot high
            if df.iloc[i]['high'] == df.iloc[i-strength:i+strength+1]['high'].max():
                df.iloc[i, df.columns.get_loc('price_pivot_high')] = True

            # Price pivot low
            if df.iloc[i]['low'] == df.iloc[i-strength:i+strength+1]['low'].min():
                df.iloc[i, df.columns.get_loc('price_pivot_low')] = True

            # RSI pivot high
            if df.iloc[i]['rsi'] == df.iloc[i-strength:i+strength+1]['rsi'].max():
                df.iloc[i, df.columns.get_loc('rsi_pivot_high')] = True

            # RSI pivot low
            if df.iloc[i]['rsi'] == df.iloc[i-strength:i+strength+1]['rsi'].min():
                df.iloc[i, df.columns.get_loc('rsi_pivot_low')] = True

        # Detect divergences
        df['bullish_div'] = False
        df['bearish_div'] = False

        lookback = self.params['lookback_period']

        for i in range(lookback, len(df)):
            # Check for bullish divergence at current low
            if df.iloc[i]['price_pivot_low']:
                bullish = self._check_bullish_divergence(df, i, lookback)
                if bullish:
                    df.iloc[i, df.columns.get_loc('bullish_div')] = True

            # Check for bearish divergence at current high
            if df.iloc[i]['price_pivot_high']:
                bearish = self._check_bearish_divergence(df, i, lookback)
                if bearish:
                    df.iloc[i, df.columns.get_loc('bearish_div')] = True

        # EMA for trend context
        df['ema_50'] = self.ema(df['close'], 50)

        return df

    def _check_bullish_divergence(self, df: pd.DataFrame, idx: int, lookback: int) -> bool:
        """
        Check for bullish divergence:
        Price makes lower low, RSI makes higher low
        """
        current_low = df.iloc[idx]['low']
        current_rsi = df.iloc[idx]['rsi']

        # Find previous pivot low
        for j in range(idx - 3, max(idx - lookback, 0), -1):
            if df.iloc[j]['price_pivot_low']:
                prev_low = df.iloc[j]['low']
                prev_rsi = df.iloc[j]['rsi']

                # Bullish divergence: price lower, RSI higher
                if current_low < prev_low and current_rsi > prev_rsi:
                    return True
                break

        return False

    def _check_bearish_divergence(self, df: pd.DataFrame, idx: int, lookback: int) -> bool:
        """
        Check for bearish divergence:
        Price makes higher high, RSI makes lower high
        """
        current_high = df.iloc[idx]['high']
        current_rsi = df.iloc[idx]['rsi']

        # Find previous pivot high
        for j in range(idx - 3, max(idx - lookback, 0), -1):
            if df.iloc[j]['price_pivot_high']:
                prev_high = df.iloc[j]['high']
                prev_rsi = df.iloc[j]['rsi']

                # Bearish divergence: price higher, RSI lower
                if current_high > prev_high and current_rsi < prev_rsi:
                    return True
                break

        return False

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals on divergences"""
        df['signal'] = 0

        for i in range(len(df)):
            row = df.iloc[i]

            # Bullish divergence -> BUY
            if row['bullish_div']:
                # Check RSI extreme if required
                if self.params['require_extreme_rsi']:
                    if row['rsi'] <= self.params['rsi_oversold'] + 10:
                        df.iloc[i, df.columns.get_loc('signal')] = 1
                else:
                    df.iloc[i, df.columns.get_loc('signal')] = 1

            # Bearish divergence -> SELL
            elif row['bearish_div']:
                if self.params['require_extreme_rsi']:
                    if row['rsi'] >= self.params['rsi_overbought'] - 10:
                        df.iloc[i, df.columns.get_loc('signal')] = -1
                else:
                    df.iloc[i, df.columns.get_loc('signal')] = -1

        return df

    def calculate_tp_sl(
        self,
        df: pd.DataFrame,
        signal: SignalType,
        entry_price: float
    ) -> Tuple[float, List[TPLevel]]:
        """Calculate TP and SL based on ATR and recent swing"""
        last_row = df.iloc[-1]
        atr = last_row['atr']
        atr_mults = self.params['atr_mult_tp']
        tp_percentages = self.config.tp_percentages

        take_profits = []

        if signal == SignalType.BUY:
            # Stop below recent low
            recent_low = df.tail(10)['low'].min()
            atr_stop = entry_price - (atr * self.params['atr_mult_sl'])
            stop_loss = min(recent_low - atr * 0.5, atr_stop)

            for i, (mult, pct) in enumerate(zip(atr_mults, tp_percentages)):
                tp_price = entry_price + (atr * mult)
                take_profits.append(TPLevel(
                    level=i + 1,
                    price=tp_price,
                    percentage=pct
                ))
        else:
            # Stop above recent high
            recent_high = df.tail(10)['high'].max()
            atr_stop = entry_price + (atr * self.params['atr_mult_sl'])
            stop_loss = max(recent_high + atr * 0.5, atr_stop)

            for i, (mult, pct) in enumerate(zip(atr_mults, tp_percentages)):
                tp_price = entry_price - (atr * mult)
                take_profits.append(TPLevel(
                    level=i + 1,
                    price=tp_price,
                    percentage=pct
                ))

        return stop_loss, take_profits

    def calculate_confidence(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate signal confidence"""
        row = df.iloc[idx]
        confidence = 0.5

        # RSI extremity
        rsi = row['rsi']
        if rsi <= self.params['rsi_oversold'] or rsi >= self.params['rsi_overbought']:
            confidence += 0.2
        elif rsi <= 35 or rsi >= 65:
            confidence += 0.1

        # Volume confirmation
        if self.params['use_volume_confirm'] and 'volume_ratio' in df.columns:
            if row['volume_ratio'] > self.params['volume_mult']:
                confidence += 0.15

        # Price at support/resistance (near EMA)
        ema_dist = abs(row['close'] - row['ema_50']) / row['close']
        if ema_dist < 0.02:  # Within 2% of EMA
            confidence += 0.1

        return min(confidence, 1.0)

    def apply_filters(self, df: pd.DataFrame, idx: int) -> Dict[str, bool]:
        """Apply entry filters"""
        row = df.iloc[idx]
        filters = {}

        # Volume filter
        if self.params['use_volume_confirm']:
            filters['volume'] = row.get('volume_ratio', 1.0) >= self.params['volume_mult']
        else:
            filters['volume'] = True

        # RSI extreme filter
        if self.params['require_extreme_rsi']:
            if row['signal'] > 0:
                filters['rsi_extreme'] = row['rsi'] <= self.params['rsi_oversold'] + 10
            else:
                filters['rsi_extreme'] = row['rsi'] >= self.params['rsi_overbought'] - 10
        else:
            filters['rsi_extreme'] = True

        return filters
