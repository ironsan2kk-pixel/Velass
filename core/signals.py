"""
VELAS Trading System - Signal Generation

Generates trading signals with full metadata for execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import pandas as pd
import uuid

from core.strategy import VelasStrategy, VelasParams, TPSLLevels


class SignalSide(Enum):
    """Signal direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class SignalStatus(Enum):
    """Signal lifecycle status."""
    PENDING = "PENDING"      # Generated, not yet sent
    ACTIVE = "ACTIVE"        # Sent to Telegram/Cornix
    PARTIAL = "PARTIAL"      # Some TPs hit
    CLOSED = "CLOSED"        # All TPs hit or SL hit
    CANCELLED = "CANCELLED"  # Manually cancelled


@dataclass
class Signal:
    """
    Trading signal with full execution details.
    """

    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    timeframe: str = ""

    # Direction and prices
    side: SignalSide = SignalSide.LONG
    entry_price: float = 0.0
    entry_zone: tuple = (0.0, 0.0)  # (min, max) entry zone

    # TP/SL
    tp_levels: List[float] = field(default_factory=list)
    tp_distribution: List[int] = field(default_factory=list)
    sl_price: float = 0.0

    # Status tracking
    status: SignalStatus = SignalStatus.PENDING
    tps_hit: List[int] = field(default_factory=list)
    remaining_position: float = 100.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    activated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # Metadata
    indicators: Dict[str, float] = field(default_factory=dict)
    filters_passed: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "entry_zone": self.entry_zone,
            "tp_levels": self.tp_levels,
            "tp_distribution": self.tp_distribution,
            "sl_price": self.sl_price,
            "status": self.status.value,
            "tps_hit": self.tps_hit,
            "remaining_position": self.remaining_position,
            "created_at": self.created_at.isoformat(),
            "indicators": self.indicators,
            "filters_passed": self.filters_passed,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Create signal from dictionary."""
        signal = cls(
            id=data.get("id", ""),
            symbol=data.get("symbol", ""),
            timeframe=data.get("timeframe", ""),
            side=SignalSide(data.get("side", "LONG")),
            entry_price=data.get("entry_price", 0.0),
            entry_zone=tuple(data.get("entry_zone", (0.0, 0.0))),
            tp_levels=data.get("tp_levels", []),
            tp_distribution=data.get("tp_distribution", []),
            sl_price=data.get("sl_price", 0.0),
            status=SignalStatus(data.get("status", "PENDING")),
            tps_hit=data.get("tps_hit", []),
            remaining_position=data.get("remaining_position", 100.0),
            indicators=data.get("indicators", {}),
            filters_passed=data.get("filters_passed", []),
            score=data.get("score", 0.0),
        )
        if data.get("created_at"):
            signal.created_at = datetime.fromisoformat(data["created_at"])
        return signal

    def get_risk_reward(self) -> float:
        """Calculate risk/reward ratio to first TP."""
        if not self.tp_levels or self.sl_price == 0:
            return 0.0

        if self.side == SignalSide.LONG:
            risk = self.entry_price - self.sl_price
            reward = self.tp_levels[0] - self.entry_price
        else:
            risk = self.sl_price - self.entry_price
            reward = self.entry_price - self.tp_levels[0]

        if risk <= 0:
            return 0.0

        return reward / risk

    def mark_tp_hit(self, tp_index: int) -> None:
        """Mark a TP as hit and update position."""
        if tp_index not in self.tps_hit:
            self.tps_hit.append(tp_index)
            if tp_index < len(self.tp_distribution):
                self.remaining_position -= self.tp_distribution[tp_index]

            if self.remaining_position <= 0:
                self.status = SignalStatus.CLOSED
                self.closed_at = datetime.now()
            else:
                self.status = SignalStatus.PARTIAL

    def mark_sl_hit(self) -> None:
        """Mark signal as stopped out."""
        self.status = SignalStatus.CLOSED
        self.closed_at = datetime.now()
        self.remaining_position = 0.0


class SignalGenerator:
    """
    Generates signals from strategy for multiple symbols/timeframes.
    """

    def __init__(
        self,
        strategy: Optional[VelasStrategy] = None,
        params: Optional[VelasParams] = None,
    ) -> None:
        """
        Initialize signal generator.

        Args:
            strategy: Velas strategy instance
            params: Strategy parameters (creates new strategy if provided)
        """
        if strategy:
            self.strategy = strategy
        elif params:
            self.strategy = VelasStrategy(params=params)
        else:
            self.strategy = VelasStrategy()

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        entry_zone_pct: float = 0.5,
    ) -> Optional[Signal]:
        """
        Generate signal for a single symbol/timeframe.

        Args:
            df: OHLCV DataFrame
            symbol: Trading pair symbol
            timeframe: Timeframe string
            entry_zone_pct: Entry zone width in percent

        Returns:
            Signal if generated, None otherwise
        """
        # Calculate indicators and signals
        df = self.strategy.calculate_indicators(df)
        df = self.strategy.generate_entry_signals(df)

        # Check last bar for signal
        if len(df) < 2:
            return None

        last_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]

        # Only generate signal on fresh breakout (not held from previous bar)
        if last_bar["signal"] == 0:
            return None

        if prev_bar["signal"] == last_bar["signal"]:
            return None  # Signal was already active

        side = SignalSide.LONG if last_bar["signal"] == 1 else SignalSide.SHORT
        entry_price = last_bar["close"]

        # Calculate entry zone
        zone_offset = entry_price * (entry_zone_pct / 100)
        if side == SignalSide.LONG:
            entry_zone = (entry_price, entry_price + zone_offset)
        else:
            entry_zone = (entry_price - zone_offset, entry_price)

        # Get TP/SL levels
        levels = self.strategy.calculate_tp_sl(
            entry_price=entry_price,
            side=side.value,
            atr=last_bar.get("atr"),
        )

        # Collect indicator values
        indicators = {
            "atr": float(last_bar.get("atr", 0)),
            "stddev": float(last_bar.get("stddev", 0)),
            "ma": float(last_bar.get("ma", 0)),
            "ma_distance_pct": float(last_bar.get("ma_distance_pct", 0)),
            "channel_high": float(last_bar.get("channel_high", 0)),
            "channel_low": float(last_bar.get("channel_low", 0)),
            "momentum": float(last_bar.get("momentum", 0)),
        }

        signal = Signal(
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            entry_price=entry_price,
            entry_zone=entry_zone,
            tp_levels=levels.tp_levels,
            tp_distribution=levels.tp_distribution,
            sl_price=levels.sl_price,
            indicators=indicators,
        )

        return signal

    def generate_signals_batch(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]],
    ) -> List[Signal]:
        """
        Generate signals for multiple symbols and timeframes.

        Args:
            data: Nested dict {symbol: {timeframe: DataFrame}}

        Returns:
            List of generated signals
        """
        signals = []

        for symbol, timeframes in data.items():
            for timeframe, df in timeframes.items():
                signal = self.generate_signal(df, symbol, timeframe)
                if signal:
                    signals.append(signal)

        return signals

    def update_signal_status(
        self,
        signal: Signal,
        current_price: float,
        high: float,
        low: float,
    ) -> Signal:
        """
        Update signal status based on current price.

        Args:
            signal: Signal to update
            current_price: Current market price
            high: High of current bar
            low: Low of current bar

        Returns:
            Updated signal
        """
        if signal.status == SignalStatus.CLOSED:
            return signal

        # Check SL
        if signal.side == SignalSide.LONG:
            if low <= signal.sl_price:
                signal.mark_sl_hit()
                return signal
        else:
            if high >= signal.sl_price:
                signal.mark_sl_hit()
                return signal

        # Check TPs
        for i, tp in enumerate(signal.tp_levels):
            if i in signal.tps_hit:
                continue

            if signal.side == SignalSide.LONG:
                if high >= tp:
                    signal.mark_tp_hit(i)
            else:
                if low <= tp:
                    signal.mark_tp_hit(i)

        return signal


def calculate_signal_score(
    signal: Signal,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate quality score for signal prioritization.

    Args:
        signal: Signal to score
        weights: Custom weights for scoring factors

    Returns:
        Score between 0 and 100
    """
    default_weights = {
        "risk_reward": 30.0,
        "momentum": 25.0,
        "atr_normalized": 20.0,
        "ma_distance": 25.0,
    }
    weights = weights or default_weights

    score = 0.0

    # Risk/Reward factor (higher is better)
    rr = signal.get_risk_reward()
    rr_score = min(rr / 3.0, 1.0) * weights["risk_reward"]  # Cap at RR=3
    score += rr_score

    # Momentum factor (stronger momentum = better)
    momentum = abs(signal.indicators.get("momentum", 0))
    momentum_score = min(momentum / 5.0, 1.0) * weights["momentum"]  # Cap at 5%
    score += momentum_score

    # ATR factor (reasonable volatility is good)
    atr = signal.indicators.get("atr", 0)
    entry = signal.entry_price
    if entry > 0:
        atr_pct = (atr / entry) * 100
        # Sweet spot around 1-2% ATR
        if 1.0 <= atr_pct <= 2.0:
            atr_score = weights["atr_normalized"]
        else:
            atr_score = max(0, weights["atr_normalized"] - abs(atr_pct - 1.5) * 10)
        score += atr_score

    # MA distance factor (closer to MA = better entry)
    ma_dist = abs(signal.indicators.get("ma_distance_pct", 0))
    ma_score = max(0, (2.0 - ma_dist) / 2.0) * weights["ma_distance"]
    score += ma_score

    signal.score = round(score, 2)
    return signal.score
