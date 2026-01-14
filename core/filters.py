"""
VELAS Trading System - Entry Filters

Implements various filters to improve signal quality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from core.signals import Signal, SignalSide


@dataclass
class FilterResult:
    """Result of filter evaluation."""
    passed: bool
    name: str
    reason: str = ""
    value: float = 0.0
    threshold: float = 0.0


class BaseFilter(ABC):
    """Base class for all filters."""

    name: str = "base"
    enabled: bool = True

    @abstractmethod
    def evaluate(
        self,
        signal: Signal,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> FilterResult:
        """
        Evaluate filter for given signal.

        Args:
            signal: Trading signal
            df: OHLCV DataFrame
            **kwargs: Additional data

        Returns:
            FilterResult with pass/fail status
        """
        pass


class MTFFilter(BaseFilter):
    """
    Multi-Timeframe Filter.

    Requires higher timeframe trend to align with signal direction.
    """

    name = "mtf"

    def __init__(
        self,
        htf_timeframe: str = "4h",
        ma_period: int = 50,
    ) -> None:
        self.htf_timeframe = htf_timeframe
        self.ma_period = ma_period

    def evaluate(
        self,
        signal: Signal,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> FilterResult:
        """Check if higher TF trend aligns with signal."""
        if htf_df is None or len(htf_df) < self.ma_period:
            return FilterResult(
                passed=False,
                name=self.name,
                reason="No HTF data available",
            )

        # Calculate HTF MA
        htf_ma = htf_df["close"].rolling(window=self.ma_period).mean().iloc[-1]
        htf_close = htf_df["close"].iloc[-1]

        # Determine HTF trend
        htf_trend = 1 if htf_close > htf_ma else -1

        # Check alignment
        signal_direction = 1 if signal.side == SignalSide.LONG else -1
        aligned = htf_trend == signal_direction

        return FilterResult(
            passed=aligned,
            name=self.name,
            reason=f"HTF trend {'aligned' if aligned else 'not aligned'}",
            value=htf_close,
            threshold=htf_ma,
        )


class VolumeFilter(BaseFilter):
    """
    Volume Filter.

    Requires volume to be above average.
    """

    name = "volume"

    def __init__(
        self,
        period: int = 20,
        multiplier: float = 1.2,
    ) -> None:
        self.period = period
        self.multiplier = multiplier

    def evaluate(
        self,
        signal: Signal,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> FilterResult:
        """Check if current volume is above average."""
        if len(df) < self.period or "volume" not in df.columns:
            return FilterResult(
                passed=False,
                name=self.name,
                reason="Insufficient data",
            )

        current_volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(window=self.period).mean().iloc[-1]
        threshold = avg_volume * self.multiplier

        passed = current_volume >= threshold

        return FilterResult(
            passed=passed,
            name=self.name,
            reason=f"Volume {'above' if passed else 'below'} threshold",
            value=current_volume,
            threshold=threshold,
        )


class RSIFilter(BaseFilter):
    """
    RSI Filter.

    For LONG: RSI should be above threshold (momentum confirmation)
    For SHORT: RSI should be below threshold
    """

    name = "rsi"

    def __init__(
        self,
        period: int = 14,
        long_level: float = 50.0,
        short_level: float = 50.0,
    ) -> None:
        self.period = period
        self.long_level = long_level
        self.short_level = short_level

    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI indicator."""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def evaluate(
        self,
        signal: Signal,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> FilterResult:
        """Check RSI conditions."""
        if len(df) < self.period + 1:
            return FilterResult(
                passed=False,
                name=self.name,
                reason="Insufficient data",
            )

        rsi = self.calculate_rsi(df).iloc[-1]

        if signal.side == SignalSide.LONG:
            passed = rsi >= self.long_level
            threshold = self.long_level
        else:
            passed = rsi <= self.short_level
            threshold = self.short_level

        return FilterResult(
            passed=passed,
            name=self.name,
            reason=f"RSI={rsi:.1f} {'passed' if passed else 'failed'}",
            value=rsi,
            threshold=threshold,
        )


class ADXFilter(BaseFilter):
    """
    ADX Filter.

    Requires ADX to be above threshold (trending market).
    """

    name = "adx"

    def __init__(
        self,
        period: int = 14,
        level: float = 25.0,
    ) -> None:
        self.period = period
        self.level = level

    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADX indicator."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed values
        atr = tr.rolling(window=self.period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=self.period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=self.period).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.period).mean()

        return adx

    def evaluate(
        self,
        signal: Signal,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> FilterResult:
        """Check ADX conditions."""
        if len(df) < self.period * 2:
            return FilterResult(
                passed=False,
                name=self.name,
                reason="Insufficient data",
            )

        adx = self.calculate_adx(df).iloc[-1]
        passed = adx >= self.level

        return FilterResult(
            passed=passed,
            name=self.name,
            reason=f"ADX={adx:.1f} {'above' if passed else 'below'} {self.level}",
            value=adx,
            threshold=self.level,
        )


class VolatilityFilter(BaseFilter):
    """
    Volatility Filter (Bollinger Band Width).

    Filters out low volatility periods.
    """

    name = "volatility"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        min_width_percentile: float = 20.0,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.min_width_percentile = min_width_percentile

    def calculate_bb_width(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Band width as percentage."""
        ma = df["close"].rolling(window=self.bb_period).mean()
        std = df["close"].rolling(window=self.bb_period).std()

        upper = ma + std * self.bb_std
        lower = ma - std * self.bb_std

        width = ((upper - lower) / ma) * 100
        return width

    def evaluate(
        self,
        signal: Signal,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> FilterResult:
        """Check volatility conditions."""
        if len(df) < self.bb_period * 2:
            return FilterResult(
                passed=False,
                name=self.name,
                reason="Insufficient data",
            )

        bb_width = self.calculate_bb_width(df)
        current_width = bb_width.iloc[-1]

        # Calculate percentile threshold
        threshold = np.percentile(bb_width.dropna(), self.min_width_percentile)
        passed = current_width >= threshold

        return FilterResult(
            passed=passed,
            name=self.name,
            reason=f"BB Width={current_width:.2f}% {'above' if passed else 'below'} threshold",
            value=current_width,
            threshold=threshold,
        )


class BTCCorrelationFilter(BaseFilter):
    """
    BTC Correlation Filter.

    For highly correlated pairs, requires BTC trend alignment.
    """

    name = "btc_correlation"

    def __init__(
        self,
        correlation_threshold: float = 0.7,
        lookback_period: int = 50,
    ) -> None:
        self.correlation_threshold = correlation_threshold
        self.lookback_period = lookback_period

    def evaluate(
        self,
        signal: Signal,
        df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> FilterResult:
        """Check BTC correlation and trend alignment."""
        if btc_df is None or len(btc_df) < self.lookback_period:
            return FilterResult(
                passed=True,  # Pass if no BTC data (can't verify)
                name=self.name,
                reason="No BTC data, skipping filter",
            )

        if signal.symbol == "BTCUSDT":
            return FilterResult(
                passed=True,
                name=self.name,
                reason="Signal is BTC, skipping filter",
            )

        # Calculate correlation
        if len(df) < self.lookback_period:
            return FilterResult(
                passed=True,
                name=self.name,
                reason="Insufficient data for correlation",
            )

        symbol_returns = df["close"].pct_change().tail(self.lookback_period)
        btc_returns = btc_df["close"].pct_change().tail(self.lookback_period)

        correlation = symbol_returns.corr(btc_returns)

        # If highly correlated, check BTC trend
        if abs(correlation) >= self.correlation_threshold:
            btc_ma = btc_df["close"].rolling(window=20).mean().iloc[-1]
            btc_close = btc_df["close"].iloc[-1]
            btc_trend = 1 if btc_close > btc_ma else -1

            signal_direction = 1 if signal.side == SignalSide.LONG else -1
            aligned = btc_trend == signal_direction

            return FilterResult(
                passed=aligned,
                name=self.name,
                reason=f"Corr={correlation:.2f}, BTC trend {'aligned' if aligned else 'not aligned'}",
                value=correlation,
                threshold=self.correlation_threshold,
            )

        return FilterResult(
            passed=True,
            name=self.name,
            reason=f"Low correlation ({correlation:.2f}), filter not applied",
            value=correlation,
            threshold=self.correlation_threshold,
        )


class FilterManager:
    """
    Manages all filters and evaluates signals.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize filter manager.

        Args:
            config: Filter configuration dictionary
        """
        self.config = config or {}
        self.filters: List[BaseFilter] = []
        self._init_filters()

    def _init_filters(self) -> None:
        """Initialize filters from config."""
        # MTF Filter
        mtf_config = self.config.get("mtf", {})
        if mtf_config.get("enabled", True):
            self.filters.append(MTFFilter(
                htf_timeframe=mtf_config.get("timeframe", "4h"),
                ma_period=mtf_config.get("ma_period", 50),
            ))

        # Volume Filter
        vol_config = self.config.get("volume", {})
        if vol_config.get("enabled", True):
            self.filters.append(VolumeFilter(
                period=vol_config.get("period", 20),
                multiplier=vol_config.get("multiplier", 1.2),
            ))

        # RSI Filter
        rsi_config = self.config.get("rsi", {})
        if rsi_config.get("enabled", False):
            self.filters.append(RSIFilter(
                period=rsi_config.get("period", 14),
                long_level=rsi_config.get("long_level", 50),
                short_level=rsi_config.get("short_level", 50),
            ))

        # ADX Filter
        adx_config = self.config.get("adx", {})
        if adx_config.get("enabled", True):
            self.filters.append(ADXFilter(
                period=adx_config.get("period", 14),
                level=adx_config.get("level", 25),
            ))

        # Volatility Filter
        vol_filt_config = self.config.get("volatility", {})
        if vol_filt_config.get("enabled", True):
            self.filters.append(VolatilityFilter(
                bb_period=vol_filt_config.get("bb_period", 20),
                bb_std=vol_filt_config.get("bb_std", 2.0),
                min_width_percentile=vol_filt_config.get("min_width_percentile", 20),
            ))

        # BTC Correlation Filter
        btc_config = self.config.get("btc_correlation", {})
        if btc_config.get("enabled", False):
            self.filters.append(BTCCorrelationFilter(
                correlation_threshold=btc_config.get("threshold", 0.7),
            ))

    def add_filter(self, filter_instance: BaseFilter) -> None:
        """Add a custom filter."""
        self.filters.append(filter_instance)

    def remove_filter(self, filter_name: str) -> None:
        """Remove filter by name."""
        self.filters = [f for f in self.filters if f.name != filter_name]

    def evaluate_signal(
        self,
        signal: Signal,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        btc_df: Optional[pd.DataFrame] = None,
        require_all: bool = True,
    ) -> Tuple[bool, List[FilterResult]]:
        """
        Evaluate all filters for a signal.

        Args:
            signal: Signal to evaluate
            df: OHLCV DataFrame
            htf_df: Higher timeframe DataFrame (for MTF filter)
            btc_df: BTC DataFrame (for correlation filter)
            require_all: If True, all filters must pass

        Returns:
            Tuple of (passed, list of FilterResults)
        """
        results = []

        for filter_instance in self.filters:
            result = filter_instance.evaluate(
                signal=signal,
                df=df,
                htf_df=htf_df,
                btc_df=btc_df,
            )
            results.append(result)

            if result.passed:
                signal.filters_passed.append(filter_instance.name)

        if require_all:
            passed = all(r.passed for r in results)
        else:
            # Majority vote
            passed = sum(r.passed for r in results) > len(results) / 2

        return passed, results

    def get_filter_summary(self, results: List[FilterResult]) -> Dict[str, Any]:
        """Get summary of filter results."""
        return {
            "total_filters": len(results),
            "passed": sum(r.passed for r in results),
            "failed": sum(not r.passed for r in results),
            "details": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "reason": r.reason,
                    "value": r.value,
                    "threshold": r.threshold,
                }
                for r in results
            ],
        }
