"""
VELAS Trading System - Core Strategy Logic

Implements the Velas channel-based trading strategy.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np


@dataclass
class VelasParams:
    """Parameters for Velas strategy."""

    i1: int = 60       # Channel period (highest/lowest)
    i2: int = 14       # StdDev/ATR period
    i3: float = 1.2    # StdDev multiplier
    i4: float = 1.5    # ATR multiplier
    i5: float = 1.5    # MA distance percent

    def to_dict(self) -> dict:
        return {
            "i1": self.i1,
            "i2": self.i2,
            "i3": self.i3,
            "i4": self.i4,
            "i5": self.i5,
        }


@dataclass
class TPSLLevels:
    """Take-Profit and Stop-Loss levels."""

    entry_price: float
    side: str  # "LONG" or "SHORT"
    tp_levels: List[float] = field(default_factory=list)
    tp_distribution: List[int] = field(default_factory=list)
    sl_price: float = 0.0

    def to_dict(self) -> dict:
        return {
            "entry_price": self.entry_price,
            "side": self.side,
            "tp_levels": self.tp_levels,
            "tp_distribution": self.tp_distribution,
            "sl_price": self.sl_price,
        }


class VelasStrategy:
    """
    Velas Trading Strategy Implementation.

    Based on channel breakouts with Fibonacci levels,
    ATR-based stops, and standard deviation filters.
    """

    def __init__(
        self,
        params: Optional[VelasParams] = None,
        tp_percents: Optional[List[float]] = None,
        tp_distribution: Optional[List[int]] = None,
        sl_percent: float = 8.5,
    ) -> None:
        """
        Initialize Velas strategy.

        Args:
            params: Strategy parameters (i1-i5)
            tp_percents: Take-profit levels in percent [1.0, 2.0, 3.0, ...]
            tp_distribution: Position distribution per TP [17, 17, 17, ...]
            sl_percent: Stop-loss percent from entry
        """
        self.params = params or VelasParams()
        self.tp_percents = tp_percents or [1.0, 2.0, 3.0, 4.0, 7.5, 14.0]
        self.tp_distribution = tp_distribution or [17, 17, 17, 17, 16, 16]
        self.sl_percent = sl_percent

        # Calculated indicators storage
        self._indicators: Optional[pd.DataFrame] = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all strategy indicators.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]

        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()

        # === Channel calculation (highest/lowest) ===
        df["channel_high"] = df["high"].rolling(window=self.params.i1).max()
        df["channel_low"] = df["low"].rolling(window=self.params.i1).min()
        df["channel_mid"] = (df["channel_high"] + df["channel_low"]) / 2

        # === Fibonacci levels ===
        channel_range = df["channel_high"] - df["channel_low"]
        df["fib_0236"] = df["channel_low"] + channel_range * 0.236
        df["fib_0382"] = df["channel_low"] + channel_range * 0.382
        df["fib_0500"] = df["channel_low"] + channel_range * 0.500
        df["fib_0618"] = df["channel_low"] + channel_range * 0.618
        df["fib_0786"] = df["channel_low"] + channel_range * 0.786

        # === ATR (Average True Range) ===
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=self.params.i2).mean()

        # === Standard Deviation ===
        df["stddev"] = df["close"].rolling(window=self.params.i2).std()

        # === Moving Average ===
        df["ma"] = df["close"].rolling(window=self.params.i2).mean()

        # === Volatility bands ===
        df["upper_band"] = df["ma"] + df["stddev"] * self.params.i3
        df["lower_band"] = df["ma"] - df["stddev"] * self.params.i3

        # === ATR bands for stops ===
        df["atr_upper"] = df["close"] + df["atr"] * self.params.i4
        df["atr_lower"] = df["close"] - df["atr"] * self.params.i4

        # === Trend detection ===
        df["trend"] = np.where(
            df["close"] > df["channel_mid"],
            1,  # Bullish
            np.where(df["close"] < df["channel_mid"], -1, 0)  # Bearish or Neutral
        )

        # === Price momentum ===
        df["momentum"] = df["close"].pct_change(periods=self.params.i2) * 100

        # === MA distance (for filter) ===
        df["ma_distance_pct"] = ((df["close"] - df["ma"]) / df["ma"]) * 100

        self._indicators = df
        return df

    def generate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry signals based on strategy logic.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            DataFrame with signal columns added
        """
        df = df.copy()

        # Ensure indicators are calculated
        if "channel_high" not in df.columns:
            df = self.calculate_indicators(df)

        # === LONG signal conditions ===
        # 1. Price breaks above channel high
        # 2. Close above MA
        # 3. MA distance within threshold
        # 4. Trend is bullish

        long_breakout = (df["close"] > df["channel_high"].shift(1))
        long_above_ma = (df["close"] > df["ma"])
        long_ma_distance = (df["ma_distance_pct"].abs() <= self.params.i5)
        long_trend = (df["trend"] == 1)

        df["long_signal"] = (
            long_breakout &
            long_above_ma &
            long_ma_distance &
            long_trend
        ).astype(int)

        # === SHORT signal conditions ===
        # 1. Price breaks below channel low
        # 2. Close below MA
        # 3. MA distance within threshold
        # 4. Trend is bearish

        short_breakout = (df["close"] < df["channel_low"].shift(1))
        short_below_ma = (df["close"] < df["ma"])
        short_ma_distance = (df["ma_distance_pct"].abs() <= self.params.i5)
        short_trend = (df["trend"] == -1)

        df["short_signal"] = (
            short_breakout &
            short_below_ma &
            short_ma_distance &
            short_trend
        ).astype(int)

        # === Combined signal ===
        # 1 = LONG, -1 = SHORT, 0 = No signal
        df["signal"] = df["long_signal"] - df["short_signal"]

        return df

    def calculate_tp_sl(
        self,
        entry_price: float,
        side: str,
        atr: Optional[float] = None,
    ) -> TPSLLevels:
        """
        Calculate Take-Profit and Stop-Loss levels.

        Args:
            entry_price: Entry price
            side: "LONG" or "SHORT"
            atr: ATR value for dynamic SL (optional)

        Returns:
            TPSLLevels with calculated levels
        """
        tp_levels = []

        if side == "LONG":
            # TP levels above entry
            for tp_pct in self.tp_percents:
                tp_price = entry_price * (1 + tp_pct / 100)
                tp_levels.append(round(tp_price, 8))

            # SL below entry
            if atr:
                sl_price = entry_price - atr * self.params.i4
            else:
                sl_price = entry_price * (1 - self.sl_percent / 100)

        else:  # SHORT
            # TP levels below entry
            for tp_pct in self.tp_percents:
                tp_price = entry_price * (1 - tp_pct / 100)
                tp_levels.append(round(tp_price, 8))

            # SL above entry
            if atr:
                sl_price = entry_price + atr * self.params.i4
            else:
                sl_price = entry_price * (1 + self.sl_percent / 100)

        return TPSLLevels(
            entry_price=entry_price,
            side=side,
            tp_levels=tp_levels,
            tp_distribution=self.tp_distribution.copy(),
            sl_price=round(sl_price, 8),
        )

    def get_signal_at_index(
        self,
        df: pd.DataFrame,
        idx: int,
    ) -> Optional[Tuple[str, TPSLLevels]]:
        """
        Get signal and levels at specific index.

        Args:
            df: DataFrame with signals
            idx: Index to check

        Returns:
            Tuple of (side, levels) or None if no signal
        """
        if idx < 0 or idx >= len(df):
            return None

        row = df.iloc[idx]

        if row.get("signal", 0) == 0:
            return None

        side = "LONG" if row["signal"] == 1 else "SHORT"
        entry_price = row["close"]
        atr = row.get("atr")

        levels = self.calculate_tp_sl(entry_price, side, atr)

        return side, levels

    def backtest_signals(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run basic backtest on signals.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with backtest results
        """
        # Calculate indicators and signals
        df = self.calculate_indicators(df)
        df = self.generate_entry_signals(df)

        # Track trades
        trades = []
        in_position = False
        position_side = None
        entry_price = 0.0
        entry_idx = 0
        tp_levels = []
        sl_price = 0.0
        remaining_position = 100.0

        for idx in range(len(df)):
            row = df.iloc[idx]

            if not in_position:
                # Check for entry signal
                if row["signal"] != 0:
                    in_position = True
                    position_side = "LONG" if row["signal"] == 1 else "SHORT"
                    entry_price = row["close"]
                    entry_idx = idx

                    levels = self.calculate_tp_sl(entry_price, position_side, row["atr"])
                    tp_levels = levels.tp_levels.copy()
                    sl_price = levels.sl_price
                    remaining_position = 100.0

            else:
                # Check for exit conditions
                high = row["high"]
                low = row["low"]

                # Check TPs
                tp_hit = None
                for i, tp in enumerate(tp_levels):
                    if position_side == "LONG" and high >= tp:
                        tp_hit = i
                        break
                    elif position_side == "SHORT" and low <= tp:
                        tp_hit = i
                        break

                # Check SL
                sl_hit = False
                if position_side == "LONG" and low <= sl_price:
                    sl_hit = True
                elif position_side == "SHORT" and high >= sl_price:
                    sl_hit = True

                if sl_hit:
                    # Full stop out
                    pnl_pct = ((sl_price - entry_price) / entry_price) * 100
                    if position_side == "SHORT":
                        pnl_pct = -pnl_pct

                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": idx,
                        "side": position_side,
                        "entry_price": entry_price,
                        "exit_price": sl_price,
                        "exit_type": "SL",
                        "pnl_pct": pnl_pct * (remaining_position / 100),
                        "remaining": 0,
                    })

                    in_position = False

                elif tp_hit is not None:
                    # Partial close at TP
                    tp_price = tp_levels[tp_hit]
                    tp_dist = self.tp_distribution[tp_hit]

                    pnl_pct = ((tp_price - entry_price) / entry_price) * 100
                    if position_side == "SHORT":
                        pnl_pct = -pnl_pct

                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": idx,
                        "side": position_side,
                        "entry_price": entry_price,
                        "exit_price": tp_price,
                        "exit_type": f"TP{tp_hit + 1}",
                        "pnl_pct": pnl_pct * (tp_dist / 100),
                        "remaining": remaining_position - tp_dist,
                    })

                    remaining_position -= tp_dist
                    tp_levels = tp_levels[tp_hit + 1:]

                    # Move SL to breakeven after first TP
                    if tp_hit == 0:
                        sl_price = entry_price

                    if remaining_position <= 0 or not tp_levels:
                        in_position = False

        return pd.DataFrame(trades) if trades else pd.DataFrame()
