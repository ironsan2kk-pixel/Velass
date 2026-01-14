"""
Tests for VELAS Trading System - Core Strategy
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.strategy import VelasStrategy, VelasParams, TPSLLevels
from core.signals import Signal, SignalSide, SignalGenerator, calculate_signal_score
from core.filters import FilterManager, VolumeFilter, ADXFilter, RSIFilter
from core.portfolio import PortfolioManager, Position


# === Fixtures ===

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n = 200

    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")

    # Generate realistic price movement
    returns = np.random.normal(0.0001, 0.01, n)
    close = 100 * np.cumprod(1 + returns)

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]

    volume = np.random.uniform(1000, 5000, n)

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    return df


@pytest.fixture
def strategy() -> VelasStrategy:
    """Create strategy instance."""
    return VelasStrategy()


@pytest.fixture
def portfolio_manager() -> PortfolioManager:
    """Create portfolio manager."""
    return PortfolioManager(
        max_positions=5,
        max_per_group=2,
        position_size_pct=2.0,
        groups={
            "majors": ["BTCUSDT", "ETHUSDT"],
            "alts": ["SOLUSDT", "AVAXUSDT"],
        }
    )


# === Strategy Tests ===

class TestVelasStrategy:
    """Tests for VelasStrategy class."""

    def test_init_default_params(self):
        """Test default parameter initialization."""
        strategy = VelasStrategy()

        assert strategy.params.i1 == 60
        assert strategy.params.i2 == 14
        assert strategy.params.i3 == 1.2
        assert strategy.params.i4 == 1.5
        assert strategy.params.i5 == 1.5

    def test_init_custom_params(self):
        """Test custom parameter initialization."""
        params = VelasParams(i1=80, i2=20, i3=1.5, i4=2.0, i5=2.0)
        strategy = VelasStrategy(params=params)

        assert strategy.params.i1 == 80
        assert strategy.params.i2 == 20

    def test_calculate_indicators(self, strategy: VelasStrategy, sample_ohlcv_data: pd.DataFrame):
        """Test indicator calculation."""
        df = strategy.calculate_indicators(sample_ohlcv_data)

        assert "channel_high" in df.columns
        assert "channel_low" in df.columns
        assert "channel_mid" in df.columns
        assert "atr" in df.columns
        assert "stddev" in df.columns
        assert "ma" in df.columns
        assert "trend" in df.columns

        # Check no NaN after warmup period
        assert not df["channel_high"].iloc[70:].isna().any()
        assert not df["atr"].iloc[20:].isna().any()

    def test_generate_entry_signals(self, strategy: VelasStrategy, sample_ohlcv_data: pd.DataFrame):
        """Test signal generation."""
        df = strategy.generate_entry_signals(sample_ohlcv_data)

        assert "long_signal" in df.columns
        assert "short_signal" in df.columns
        assert "signal" in df.columns

        # Signal values should be -1, 0, or 1
        assert df["signal"].isin([-1, 0, 1]).all()

    def test_calculate_tp_sl_long(self, strategy: VelasStrategy):
        """Test TP/SL calculation for LONG."""
        entry = 100.0
        levels = strategy.calculate_tp_sl(entry, "LONG")

        assert levels.side == "LONG"
        assert levels.entry_price == entry
        assert len(levels.tp_levels) == 6

        # TPs should be above entry for LONG
        for tp in levels.tp_levels:
            assert tp > entry

        # SL should be below entry for LONG
        assert levels.sl_price < entry

    def test_calculate_tp_sl_short(self, strategy: VelasStrategy):
        """Test TP/SL calculation for SHORT."""
        entry = 100.0
        levels = strategy.calculate_tp_sl(entry, "SHORT")

        assert levels.side == "SHORT"

        # TPs should be below entry for SHORT
        for tp in levels.tp_levels:
            assert tp < entry

        # SL should be above entry for SHORT
        assert levels.sl_price > entry


# === Signal Tests ===

class TestSignalGenerator:
    """Tests for SignalGenerator class."""

    def test_generate_signal(self, sample_ohlcv_data: pd.DataFrame):
        """Test signal generation."""
        generator = SignalGenerator()
        signal = generator.generate_signal(
            sample_ohlcv_data,
            symbol="BTCUSDT",
            timeframe="1h"
        )

        # Signal may or may not be generated depending on data
        if signal:
            assert signal.symbol == "BTCUSDT"
            assert signal.timeframe == "1h"
            assert signal.side in [SignalSide.LONG, SignalSide.SHORT]
            assert signal.entry_price > 0
            assert len(signal.tp_levels) > 0

    def test_signal_score(self):
        """Test signal scoring."""
        signal = Signal(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            entry_price=100.0,
            tp_levels=[101.0, 102.0, 103.0],
            sl_price=95.0,
            indicators={
                "atr": 1.5,
                "momentum": 2.0,
                "ma_distance_pct": 0.5,
            }
        )

        score = calculate_signal_score(signal)

        assert signal.score == score
        assert 0 <= score <= 100


# === Filter Tests ===

class TestFilters:
    """Tests for filter classes."""

    def test_volume_filter(self, sample_ohlcv_data: pd.DataFrame):
        """Test volume filter."""
        filter_instance = VolumeFilter(period=20, multiplier=1.2)

        signal = Signal(symbol="TEST", side=SignalSide.LONG)
        result = filter_instance.evaluate(signal, sample_ohlcv_data)

        assert result.name == "volume"
        assert isinstance(result.passed, bool)

    def test_rsi_filter(self, sample_ohlcv_data: pd.DataFrame):
        """Test RSI filter."""
        filter_instance = RSIFilter(period=14, long_level=50, short_level=50)

        signal = Signal(symbol="TEST", side=SignalSide.LONG)
        result = filter_instance.evaluate(signal, sample_ohlcv_data)

        assert result.name == "rsi"
        assert 0 <= result.value <= 100

    def test_filter_manager(self, sample_ohlcv_data: pd.DataFrame):
        """Test filter manager."""
        config = {
            "volume": {"enabled": True, "period": 20, "multiplier": 1.0},
            "adx": {"enabled": True, "period": 14, "level": 20},
            "rsi": {"enabled": False},
        }

        manager = FilterManager(config)
        signal = Signal(symbol="TEST", side=SignalSide.LONG)

        passed, results = manager.evaluate_signal(signal, sample_ohlcv_data)

        assert isinstance(passed, bool)
        assert len(results) > 0


# === Portfolio Tests ===

class TestPortfolioManager:
    """Tests for PortfolioManager class."""

    def test_init(self, portfolio_manager: PortfolioManager):
        """Test initialization."""
        assert portfolio_manager.max_positions == 5
        assert portfolio_manager.max_per_group == 2
        assert len(portfolio_manager.groups) == 2

    def test_get_symbol_group(self, portfolio_manager: PortfolioManager):
        """Test symbol group mapping."""
        assert portfolio_manager.get_symbol_group("BTCUSDT") == "majors"
        assert portfolio_manager.get_symbol_group("SOLUSDT") == "alts"
        assert portfolio_manager.get_symbol_group("UNKNOWN") == "unknown"

    def test_can_open_position(self, portfolio_manager: PortfolioManager):
        """Test position opening validation."""
        signal = Signal(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            entry_price=50000,
            sl_price=47500,
            tp_levels=[51000, 52000],
        )

        can_open, reason = portfolio_manager.can_open_position(signal)
        assert can_open is True

    def test_open_position(self, portfolio_manager: PortfolioManager):
        """Test opening position."""
        signal = Signal(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            entry_price=50000,
            sl_price=47500,
            tp_levels=[51000, 52000],
        )

        position = portfolio_manager.open_position(signal)

        assert position is not None
        assert position.symbol == "BTCUSDT"
        assert position.entry_price == 50000

    def test_max_positions_limit(self, portfolio_manager: PortfolioManager):
        """Test max positions limit."""
        symbols = ["BTC", "ETH", "SOL", "AVAX", "DOT", "LINK"]

        for i, sym in enumerate(symbols[:5]):
            signal = Signal(
                symbol=f"{sym}USDT",
                side=SignalSide.LONG,
                entry_price=100,
                sl_price=95,
                tp_levels=[105],
            )
            portfolio_manager.open_position(signal)

        # 6th position should be rejected
        signal = Signal(
            symbol="LINKUSDT",
            side=SignalSide.LONG,
            entry_price=100,
            sl_price=95,
            tp_levels=[105],
        )

        can_open, reason = portfolio_manager.can_open_position(signal)
        assert can_open is False
        assert "Max positions" in reason

    def test_portfolio_summary(self, portfolio_manager: PortfolioManager):
        """Test portfolio summary."""
        signal = Signal(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            entry_price=50000,
            sl_price=47500,
            tp_levels=[51000],
        )
        portfolio_manager.open_position(signal)

        summary = portfolio_manager.get_portfolio_summary()

        assert "open_positions" in summary
        assert "total_exposure" in summary
        assert "portfolio_heat" in summary
        assert summary["open_positions"] == 1


# === Run tests ===

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
