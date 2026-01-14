"""
Bollinger Bands Strategy - Volatility-based Mean Reversion and Breakout

Combines mean reversion at bands with breakout detection.
Includes squeeze detection for explosive moves.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategyConfig, SignalType, TPLevel


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Trading Strategy

    Two modes:
    1. Mean Reversion: Trade bounces off bands
    2. Breakout: Trade breakouts after squeeze

    Uses Keltner Channels for squeeze detection.
    """

    @property
    def name(self) -> str:
        return "bollinger"

    @property
    def description(self) -> str:
        return "Bollinger Bands with squeeze detection for mean reversion and breakouts"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            # Bollinger parameters
            'bb_period': 20,
            'bb_std': 2.0,

            # Keltner for squeeze
            'kc_period': 20,
            'kc_mult': 1.5,

            # Mode: 'reversion' or 'breakout'
            'mode': 'reversion',

            # ATR for SL/TP
            'atr_period': 14,
            'atr_mult_sl': 1.5,

            # Filters
            'use_rsi_confirm': True,
            'rsi_period': 14,
            'rsi_ob': 70,
            'rsi_os': 30,

            # Squeeze settings
            'squeeze_lookback': 6,  # Min bars in squeeze for breakout
        }

    @property
    def params_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bb_period": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 50,
                    "description": "Bollinger Bands period"
                },
                "bb_std": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 4.0,
                    "description": "Standard deviation multiplier"
                },
                "mode": {
                    "type": "string",
                    "enum": ["reversion", "breakout"],
                    "description": "Trading mode"
                },
                "kc_period": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 50,
                    "description": "Keltner Channel period"
                },
                "kc_mult": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 3.0,
                    "description": "Keltner multiplier"
                },
                "use_rsi_confirm": {
                    "type": "boolean",
                    "description": "Use RSI for confirmation"
                }
            }
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands, Keltner Channels, and squeeze"""

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(
            df['close'],
            self.params['bb_period'],
            self.params['bb_std']
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower

        # Bollinger Band Width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # %B - Position within bands
        df['percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Keltner Channels
        kc_period = self.params['kc_period']
        kc_mult = self.params['kc_mult']

        df['kc_middle'] = self.ema(df['close'], kc_period)
        df['atr'] = self.atr(df, self.params['atr_period'])
        df['kc_upper'] = df['kc_middle'] + (df['atr'] * kc_mult)
        df['kc_lower'] = df['kc_middle'] - (df['atr'] * kc_mult)

        # Squeeze detection: BB inside KC
        df['squeeze'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])

        # Squeeze release (was in squeeze, now breaking out)
        df['squeeze_release'] = df['squeeze'].shift(1) & ~df['squeeze']

        # Momentum (for squeeze breakout direction)
        df['momentum'] = df['close'] - df['close'].shift(self.params['bb_period'])

        # RSI
        if self.params['use_rsi_confirm']:
            df['rsi'] = self.rsi(df['close'], self.params['rsi_period'])

        # Band touch detection
        df['touch_upper'] = df['high'] >= df['bb_upper']
        df['touch_lower'] = df['low'] <= df['bb_lower']

        # Close outside bands
        df['close_above'] = df['close'] > df['bb_upper']
        df['close_below'] = df['close'] < df['bb_lower']

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on mode"""
        df['signal'] = 0
        mode = self.params['mode']

        for i in range(2, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            if mode == 'reversion':
                # Mean Reversion: Buy at lower band, sell at upper band
                # with reversal confirmation

                # BUY: Price touches lower band and starts reverting
                if prev['touch_lower'] and row['close'] > prev['close']:
                    if not self.params['use_rsi_confirm'] or row['rsi'] < self.params['rsi_os'] + 10:
                        df.iloc[i, df.columns.get_loc('signal')] = 1

                # SELL: Price touches upper band and starts reverting
                elif prev['touch_upper'] and row['close'] < prev['close']:
                    if not self.params['use_rsi_confirm'] or row['rsi'] > self.params['rsi_ob'] - 10:
                        df.iloc[i, df.columns.get_loc('signal')] = -1

            else:  # breakout mode
                # Breakout: Trade squeeze release in momentum direction

                if row['squeeze_release']:
                    # Count bars in squeeze
                    squeeze_bars = 0
                    for j in range(i-1, max(0, i-20), -1):
                        if df.iloc[j]['squeeze']:
                            squeeze_bars += 1
                        else:
                            break

                    # Only trade if was in squeeze long enough
                    if squeeze_bars >= self.params['squeeze_lookback']:
                        if row['momentum'] > 0:
                            df.iloc[i, df.columns.get_loc('signal')] = 1
                        else:
                            df.iloc[i, df.columns.get_loc('signal')] = -1

        return df

    def calculate_tp_sl(
        self,
        df: pd.DataFrame,
        signal: SignalType,
        entry_price: float
    ) -> Tuple[float, List[TPLevel]]:
        """Calculate TP/SL based on band width and ATR"""
        last_row = df.iloc[-1]
        atr = last_row['atr']
        bb_width = last_row['bb_upper'] - last_row['bb_lower']

        tp_percentages = self.config.tp_percentages
        take_profits = []

        mode = self.params['mode']

        if signal == SignalType.BUY:
            stop_loss = entry_price - (atr * self.params['atr_mult_sl'])

            if mode == 'reversion':
                # TPs toward middle and upper band
                targets = [
                    last_row['bb_middle'],
                    last_row['bb_middle'] + bb_width * 0.25,
                    last_row['bb_upper'],
                    last_row['bb_upper'] + atr,
                    last_row['bb_upper'] + atr * 2,
                    last_row['bb_upper'] + atr * 3,
                ]
            else:
                # Breakout: ATR-based targets
                targets = [entry_price + atr * mult for mult in [1, 2, 3, 4, 5, 6]]

            for i, (target, pct) in enumerate(zip(targets, tp_percentages)):
                take_profits.append(TPLevel(
                    level=i + 1,
                    price=target,
                    percentage=pct
                ))
        else:
            stop_loss = entry_price + (atr * self.params['atr_mult_sl'])

            if mode == 'reversion':
                targets = [
                    last_row['bb_middle'],
                    last_row['bb_middle'] - bb_width * 0.25,
                    last_row['bb_lower'],
                    last_row['bb_lower'] - atr,
                    last_row['bb_lower'] - atr * 2,
                    last_row['bb_lower'] - atr * 3,
                ]
            else:
                targets = [entry_price - atr * mult for mult in [1, 2, 3, 4, 5, 6]]

            for i, (target, pct) in enumerate(zip(targets, tp_percentages)):
                take_profits.append(TPLevel(
                    level=i + 1,
                    price=target,
                    percentage=pct
                ))

        return stop_loss, take_profits

    def calculate_confidence(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate signal confidence"""
        row = df.iloc[idx]
        confidence = 0.5
        mode = self.params['mode']

        if mode == 'reversion':
            # %B extremity
            if row['percent_b'] < 0.1 or row['percent_b'] > 0.9:
                confidence += 0.2

            # RSI confirmation
            if self.params['use_rsi_confirm']:
                if row['signal'] > 0 and row['rsi'] < 35:
                    confidence += 0.15
                elif row['signal'] < 0 and row['rsi'] > 65:
                    confidence += 0.15

        else:  # breakout
            # Squeeze duration (longer = stronger)
            squeeze_bars = 0
            for j in range(idx-1, max(0, idx-20), -1):
                if df.iloc[j]['squeeze']:
                    squeeze_bars += 1
                else:
                    break

            if squeeze_bars >= 10:
                confidence += 0.2
            elif squeeze_bars >= 6:
                confidence += 0.1

            # Momentum strength
            if abs(row['momentum']) > row['atr'] * 2:
                confidence += 0.15

        # Band width (narrower = more explosive)
        if row['bb_width'] < df['bb_width'].rolling(50).mean().iloc[-1]:
            confidence += 0.1

        return min(confidence, 1.0)

    def apply_filters(self, df: pd.DataFrame, idx: int) -> Dict[str, bool]:
        """Apply filters"""
        row = df.iloc[idx]
        filters = {}

        # RSI filter
        if self.params['use_rsi_confirm']:
            if row['signal'] > 0:
                filters['rsi'] = row['rsi'] < self.params['rsi_os'] + 15
            else:
                filters['rsi'] = row['rsi'] > self.params['rsi_ob'] - 15
        else:
            filters['rsi'] = True

        # Volume spike (optional)
        filters['default'] = True

        return filters
