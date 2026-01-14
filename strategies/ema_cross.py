"""
EMA Crossover Strategy - Classic Moving Average Strategy

Simple yet effective trend-following strategy based on EMA crossovers.
Includes MACD confirmation and dynamic TP/SL based on ATR.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategyConfig, SignalType, TPLevel


class EMACrossStrategy(BaseStrategy):
    """
    EMA Crossover Trading Strategy

    Entry Logic:
    - BUY: Fast EMA crosses above Slow EMA with MACD confirmation
    - SELL: Fast EMA crosses below Slow EMA with MACD confirmation

    Exit Logic:
    - ATR-based TP levels
    - ATR-based stop loss
    - Optional trailing stop
    """

    @property
    def name(self) -> str:
        return "ema_cross"

    @property
    def description(self) -> str:
        return "EMA crossover strategy with MACD confirmation and ATR-based risk management"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            # EMA parameters
            'fast_ema': 9,
            'slow_ema': 21,
            'signal_ema': 50,  # Overall trend filter

            # MACD for confirmation
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'use_macd_confirm': True,

            # ATR for TP/SL
            'atr_period': 14,
            'atr_mult_sl': 2.0,
            'atr_mult_tp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],

            # Filters
            'use_trend_filter': True,
            'min_ema_separation': 0.001,  # 0.1% minimum separation
        }

    @property
    def params_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fast_ema": {
                    "type": "integer",
                    "minimum": 3,
                    "maximum": 50,
                    "description": "Fast EMA period"
                },
                "slow_ema": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 100,
                    "description": "Slow EMA period"
                },
                "signal_ema": {
                    "type": "integer",
                    "minimum": 20,
                    "maximum": 200,
                    "description": "Signal/trend EMA period"
                },
                "atr_period": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 50,
                    "description": "ATR period"
                },
                "atr_mult_sl": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 5.0,
                    "description": "ATR multiplier for stop loss"
                },
                "use_macd_confirm": {
                    "type": "boolean",
                    "description": "Require MACD confirmation"
                },
                "use_trend_filter": {
                    "type": "boolean",
                    "description": "Use trend EMA filter"
                }
            }
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs, MACD, and ATR"""

        # EMAs
        df['ema_fast'] = self.ema(df['close'], self.params['fast_ema'])
        df['ema_slow'] = self.ema(df['close'], self.params['slow_ema'])
        df['ema_signal'] = self.ema(df['close'], self.params['signal_ema'])

        # EMA separation (percentage)
        df['ema_separation'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow']

        # MACD
        macd_line, signal_line, histogram = self.macd(
            df['close'],
            self.params['macd_fast'],
            self.params['macd_slow'],
            self.params['macd_signal']
        )
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram

        # ATR
        df['atr'] = self.atr(df, self.params['atr_period'])

        # RSI for additional info
        df['rsi'] = self.rsi(df['close'], 14)

        # Crossover detection
        df['ema_cross_up'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['ema_cross_down'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))

        # MACD crossover
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals on EMA crossovers"""
        df['signal'] = 0

        for i in range(1, len(df)):
            row = df.iloc[i]

            # Check for EMA crossover
            if row['ema_cross_up']:
                # Check MACD confirmation if enabled
                if self.params['use_macd_confirm']:
                    # MACD should be bullish or crossing up
                    if row['macd_hist'] > 0 or row['macd_cross_up']:
                        df.iloc[i, df.columns.get_loc('signal')] = 1
                else:
                    df.iloc[i, df.columns.get_loc('signal')] = 1

            elif row['ema_cross_down']:
                # Check MACD confirmation if enabled
                if self.params['use_macd_confirm']:
                    if row['macd_hist'] < 0 or row['macd_cross_down']:
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
        """Calculate ATR-based TP and SL levels"""
        atr = df.iloc[-1]['atr']
        atr_mults = self.params['atr_mult_tp']
        tp_percentages = self.config.tp_percentages

        take_profits = []

        if signal == SignalType.BUY:
            stop_loss = entry_price - (atr * self.params['atr_mult_sl'])

            for i, (mult, pct) in enumerate(zip(atr_mults, tp_percentages)):
                tp_price = entry_price + (atr * mult)
                take_profits.append(TPLevel(
                    level=i + 1,
                    price=tp_price,
                    percentage=pct
                ))
        else:
            stop_loss = entry_price + (atr * self.params['atr_mult_sl'])

            for i, (mult, pct) in enumerate(zip(atr_mults, tp_percentages)):
                tp_price = entry_price - (atr * mult)
                take_profits.append(TPLevel(
                    level=i + 1,
                    price=tp_price,
                    percentage=pct
                ))

        return stop_loss, take_profits

    def calculate_confidence(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate confidence based on signal strength"""
        row = df.iloc[idx]
        confidence = 0.5

        # EMA separation strength
        sep = abs(row['ema_separation'])
        if sep > 0.01:  # > 1%
            confidence += 0.15
        elif sep > 0.005:
            confidence += 0.1

        # MACD histogram strength
        if abs(row['macd_hist']) > row['atr'] * 0.1:
            confidence += 0.1

        # Trend alignment
        if self.params['use_trend_filter']:
            if row['signal'] > 0 and row['close'] > row['ema_signal']:
                confidence += 0.15
            elif row['signal'] < 0 and row['close'] < row['ema_signal']:
                confidence += 0.15

        # RSI not extreme
        if 30 < row['rsi'] < 70:
            confidence += 0.1

        return min(confidence, 1.0)

    def apply_filters(self, df: pd.DataFrame, idx: int) -> Dict[str, bool]:
        """Apply entry filters"""
        row = df.iloc[idx]
        filters = {}

        # Trend filter
        if self.params['use_trend_filter']:
            if row['signal'] > 0:
                filters['trend'] = row['close'] > row['ema_signal']
            else:
                filters['trend'] = row['close'] < row['ema_signal']
        else:
            filters['trend'] = True

        # Minimum separation
        filters['separation'] = abs(row['ema_separation']) >= self.params['min_ema_separation']

        # MACD alignment
        if self.params['use_macd_confirm']:
            if row['signal'] > 0:
                filters['macd'] = row['macd_hist'] > 0
            else:
                filters['macd'] = row['macd_hist'] < 0
        else:
            filters['macd'] = True

        return filters
