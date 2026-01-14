"""
Velas Channel Strategy - Plugin Implementation

Channel-based strategy with Fibonacci TP levels and ATR-based stops.
Uses price channels with breakout confirmation.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategyConfig, StrategyResult, SignalType, TPLevel


class VelasStrategy(BaseStrategy):
    """
    Velas Channel Trading Strategy

    Entry Logic:
    - BUY: Price breaks above upper channel with confirmation
    - SELL: Price breaks below lower channel with confirmation

    Exit Logic:
    - 6 TP levels based on Fibonacci extensions
    - Stop loss based on ATR or channel opposite
    - Trailing stop after TP1 hit
    """

    @property
    def name(self) -> str:
        return "velas"

    @property
    def description(self) -> str:
        return "Channel breakout strategy with Fibonacci TP levels and ATR-based risk management"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            # Channel parameters
            'channel_period': 20,
            'channel_mult': 2.0,

            # ATR for volatility
            'atr_period': 14,
            'atr_mult_sl': 1.5,

            # Confirmation
            'confirmation_candles': 2,
            'min_channel_width': 0.005,  # 0.5% minimum

            # Fibonacci levels for TP
            'fib_levels': [0.236, 0.382, 0.5, 0.618, 0.786, 1.0],

            # Filters
            'use_volume_filter': True,
            'volume_mult': 1.5,
            'use_trend_filter': True,
            'trend_ema_period': 50,
        }

    @property
    def params_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel_period": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 100,
                    "description": "Period for channel calculation"
                },
                "channel_mult": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 5.0,
                    "description": "Channel width multiplier"
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
                "confirmation_candles": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Candles needed for confirmation"
                },
                "fib_levels": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Fibonacci levels for take profits"
                },
                "use_volume_filter": {
                    "type": "boolean",
                    "description": "Enable volume confirmation"
                },
                "volume_mult": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 5.0,
                    "description": "Volume multiplier threshold"
                },
                "use_trend_filter": {
                    "type": "boolean",
                    "description": "Enable trend filter"
                },
                "trend_ema_period": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 200,
                    "description": "EMA period for trend filter"
                }
            }
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Velas channel and supporting indicators"""
        period = self.params['channel_period']
        mult = self.params['channel_mult']
        atr_period = self.params['atr_period']

        # Price channel (Donchian-style with EMA smoothing)
        df['channel_high'] = df['high'].rolling(window=period).max()
        df['channel_low'] = df['low'].rolling(window=period).min()
        df['channel_mid'] = (df['channel_high'] + df['channel_low']) / 2

        # Standard deviation for dynamic width
        df['std'] = df['close'].rolling(window=period).std()

        # Adjusted channel with volatility
        df['upper_channel'] = df['channel_mid'] + (df['std'] * mult)
        df['lower_channel'] = df['channel_mid'] - (df['std'] * mult)

        # Channel width percentage
        df['channel_width'] = (df['upper_channel'] - df['lower_channel']) / df['channel_mid']

        # ATR
        df['atr'] = self.atr(df, atr_period)

        # Trend EMA
        if self.params['use_trend_filter']:
            df['trend_ema'] = self.ema(df['close'], self.params['trend_ema_period'])

        # Volume SMA
        if self.params['use_volume_filter']:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

        # RSI for additional confirmation
        df['rsi'] = self.rsi(df['close'], 14)

        # Price position in channel (0 = lower, 1 = upper)
        df['channel_position'] = (df['close'] - df['lower_channel']) / (df['upper_channel'] - df['lower_channel'])

        # Breakout detection
        df['break_upper'] = (df['close'] > df['upper_channel']) & (df['close'].shift(1) <= df['upper_channel'].shift(1))
        df['break_lower'] = (df['close'] < df['lower_channel']) & (df['close'].shift(1) >= df['lower_channel'].shift(1))

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on channel breakouts"""
        df['signal'] = 0
        min_width = self.params['min_channel_width']
        confirm = self.params['confirmation_candles']

        for i in range(confirm, len(df)):
            # Skip if channel too narrow
            if df.iloc[i]['channel_width'] < min_width:
                continue

            # Count confirmation candles above/below channel
            closes_above = sum(
                df.iloc[i-j]['close'] > df.iloc[i-j]['upper_channel']
                for j in range(confirm)
            )
            closes_below = sum(
                df.iloc[i-j]['close'] < df.iloc[i-j]['lower_channel']
                for j in range(confirm)
            )

            # BUY signal: breakout above with confirmation
            if closes_above >= confirm:
                df.iloc[i, df.columns.get_loc('signal')] = 1

            # SELL signal: breakout below with confirmation
            elif closes_below >= confirm:
                df.iloc[i, df.columns.get_loc('signal')] = -1

        return df

    def calculate_tp_sl(
        self,
        df: pd.DataFrame,
        signal: SignalType,
        entry_price: float
    ) -> Tuple[float, List[TPLevel]]:
        """Calculate TP and SL based on channel and Fibonacci levels"""
        last_row = df.iloc[-1]
        atr = last_row['atr']
        channel_height = last_row['upper_channel'] - last_row['lower_channel']

        fib_levels = self.params['fib_levels']
        tp_percentages = self.config.tp_percentages

        take_profits = []

        if signal == SignalType.BUY:
            # Stop loss below lower channel or ATR-based
            atr_stop = entry_price - (atr * self.params['atr_mult_sl'])
            channel_stop = last_row['lower_channel']
            stop_loss = max(atr_stop, channel_stop)  # Use the closer one

            # TP levels based on Fibonacci extensions
            for i, (fib, pct) in enumerate(zip(fib_levels, tp_percentages)):
                tp_price = entry_price + (channel_height * fib)
                take_profits.append(TPLevel(
                    level=i + 1,
                    price=tp_price,
                    percentage=pct
                ))

        else:  # SELL
            # Stop loss above upper channel or ATR-based
            atr_stop = entry_price + (atr * self.params['atr_mult_sl'])
            channel_stop = last_row['upper_channel']
            stop_loss = min(atr_stop, channel_stop)

            # TP levels based on Fibonacci extensions (downward)
            for i, (fib, pct) in enumerate(zip(fib_levels, tp_percentages)):
                tp_price = entry_price - (channel_height * fib)
                take_profits.append(TPLevel(
                    level=i + 1,
                    price=tp_price,
                    percentage=pct
                ))

        return stop_loss, take_profits

    def calculate_confidence(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate signal confidence based on multiple factors"""
        row = df.iloc[idx]
        confidence = 0.5  # Base confidence

        # Volume confirmation
        if self.params['use_volume_filter'] and 'volume_ratio' in df.columns:
            if row['volume_ratio'] > self.params['volume_mult']:
                confidence += 0.15

        # Trend alignment
        if self.params['use_trend_filter'] and 'trend_ema' in df.columns:
            signal = row['signal']
            if signal > 0 and row['close'] > row['trend_ema']:
                confidence += 0.15
            elif signal < 0 and row['close'] < row['trend_ema']:
                confidence += 0.15

        # RSI confirmation
        if 'rsi' in df.columns:
            rsi = row['rsi']
            signal = row['signal']
            if signal > 0 and 40 < rsi < 70:  # Not overbought for longs
                confidence += 0.1
            elif signal < 0 and 30 < rsi < 60:  # Not oversold for shorts
                confidence += 0.1

        # Channel width (wider = stronger breakout)
        if row['channel_width'] > 0.02:  # > 2%
            confidence += 0.1

        return min(confidence, 1.0)

    def apply_filters(self, df: pd.DataFrame, idx: int) -> Dict[str, bool]:
        """Apply entry filters"""
        row = df.iloc[idx]
        filters = {}

        # Volume filter
        if self.params['use_volume_filter']:
            filters['volume'] = row.get('volume_ratio', 0) >= self.params['volume_mult']
        else:
            filters['volume'] = True

        # Trend filter
        if self.params['use_trend_filter']:
            signal = row['signal']
            if signal > 0:
                filters['trend'] = row['close'] > row.get('trend_ema', row['close'])
            elif signal < 0:
                filters['trend'] = row['close'] < row.get('trend_ema', row['close'])
            else:
                filters['trend'] = True
        else:
            filters['trend'] = True

        # Channel width filter
        filters['channel_width'] = row['channel_width'] >= self.params['min_channel_width']

        return filters
