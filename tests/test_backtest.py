"""
Tests for VELAS Trading System - Backtest Module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from backtest.engine import Backtester, BacktestConfig, BacktestResult, Trade, TradeResult
from backtest.metrics import MetricsCalculator, TradeMetrics, PortfolioMetrics
from backtest.reports import ReportGenerator


# === Fixtures ===

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data with trends."""
    np.random.seed(42)
    n = 500

    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")

    # Create trending data
    trend = np.sin(np.linspace(0, 4 * np.pi, n)) * 10
    noise = np.random.normal(0, 2, n)
    close = 100 + trend + np.cumsum(noise * 0.1)

    high = close + np.abs(np.random.normal(0, 1, n))
    low = close - np.abs(np.random.normal(0, 1, n))
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
def backtest_config() -> BacktestConfig:
    """Create backtest configuration."""
    return BacktestConfig(
        tp_percents=[1.0, 2.0, 3.0],
        tp_distribution=[34, 33, 33],
        sl_percent=5.0,
        position_size_pct=2.0,
    )


@pytest.fixture
def sample_trades() -> list:
    """Create sample trades for metrics testing."""
    return [
        Trade(id="1", side="LONG", entry_price=100, exit_price=102,
              pnl_pct=0.5, pnl_r=1.0, result=TradeResult.WIN,
              tps_hit=[0], duration_bars=10),
        Trade(id="2", side="LONG", entry_price=100, exit_price=98,
              pnl_pct=-0.3, pnl_r=-0.6, result=TradeResult.LOSS,
              tps_hit=[], duration_bars=15),
        Trade(id="3", side="SHORT", entry_price=100, exit_price=97,
              pnl_pct=0.8, pnl_r=1.6, result=TradeResult.WIN,
              tps_hit=[0, 1], duration_bars=20),
        Trade(id="4", side="LONG", entry_price=100, exit_price=101,
              pnl_pct=0.2, pnl_r=0.4, result=TradeResult.WIN,
              tps_hit=[0], duration_bars=8),
        Trade(id="5", side="SHORT", entry_price=100, exit_price=102,
              pnl_pct=-0.4, pnl_r=-0.8, result=TradeResult.LOSS,
              tps_hit=[], duration_bars=12),
    ]


# === Backtester Tests ===

class TestBacktester:
    """Tests for Backtester class."""

    def test_init_default_config(self):
        """Test default initialization."""
        backtester = Backtester()

        assert backtester.config is not None
        assert backtester.strategy is not None

    def test_init_custom_config(self, backtest_config: BacktestConfig):
        """Test custom configuration."""
        backtester = Backtester(config=backtest_config)

        assert backtester.config.sl_percent == 5.0
        assert len(backtester.config.tp_percents) == 3

    def test_run_backtest(
        self,
        backtest_config: BacktestConfig,
        sample_ohlcv_data: pd.DataFrame
    ):
        """Test running backtest."""
        backtester = Backtester(config=backtest_config)
        result = backtester.run(
            df=sample_ohlcv_data,
            symbol="BTCUSDT",
            timeframe="1h"
        )

        assert isinstance(result, BacktestResult)
        assert result.symbol == "BTCUSDT"
        assert result.timeframe == "1h"
        assert len(result.equity_curve) > 0

    def test_backtest_result_structure(
        self,
        backtest_config: BacktestConfig,
        sample_ohlcv_data: pd.DataFrame
    ):
        """Test backtest result structure."""
        backtester = Backtester(config=backtest_config)
        result = backtester.run(sample_ohlcv_data)

        # Check equity curve
        assert len(result.equity_curve) == len(sample_ohlcv_data)
        assert result.equity_curve[0] == 100.0  # Starting equity

        # Check drawdown curve
        assert len(result.drawdown_curve) == len(result.equity_curve)
        assert all(dd >= 0 for dd in result.drawdown_curve)


# === Metrics Tests ===

class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def test_calculate_trade_metrics(self, sample_trades: list):
        """Test trade metrics calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate_trade_metrics(sample_trades)

        assert metrics.total_trades == 5
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 2
        assert metrics.win_rate == 60.0

    def test_calculate_win_rate_by_tp(self, sample_trades: list):
        """Test TP-level win rates."""
        calc = MetricsCalculator()
        metrics = calc.calculate_trade_metrics(sample_trades)

        # TP1 hit rate
        assert metrics.win_rate_tp1 == 60.0  # 3 out of 5 hit TP1

    def test_calculate_portfolio_metrics(self, sample_trades: list):
        """Test portfolio metrics calculation."""
        calc = MetricsCalculator()

        # Create equity curve
        equity = [100.0]
        for trade in sample_trades:
            equity.append(equity[-1] * (1 + trade.pnl_pct / 100))

        # Create drawdown curve
        drawdown = []
        peak = equity[0]
        for e in equity:
            if e > peak:
                peak = e
            dd = ((peak - e) / peak) * 100 if peak > 0 else 0
            drawdown.append(dd)

        metrics = calc.calculate_portfolio_metrics(equity, drawdown, sample_trades)

        assert metrics.total_return_pct != 0
        assert metrics.max_drawdown_pct >= 0

    def test_sharpe_ratio(self, sample_trades: list):
        """Test Sharpe ratio calculation."""
        calc = MetricsCalculator()

        equity = [100.0]
        for trade in sample_trades:
            equity.append(equity[-1] * (1 + trade.pnl_pct / 100))

        drawdown = [0.0] * len(equity)

        metrics = calc.calculate_portfolio_metrics(equity, drawdown, sample_trades)

        # Sharpe should be a real number
        assert not np.isnan(metrics.sharpe_ratio)

    def test_profit_factor(self, sample_trades: list):
        """Test profit factor calculation."""
        calc = MetricsCalculator()

        equity = [100.0]
        for trade in sample_trades:
            equity.append(equity[-1] * (1 + trade.pnl_pct / 100))

        drawdown = [0.0] * len(equity)

        metrics = calc.calculate_portfolio_metrics(equity, drawdown, sample_trades)

        # Profit factor should be positive
        assert metrics.profit_factor >= 0

    def test_empty_trades(self):
        """Test metrics with no trades."""
        calc = MetricsCalculator()
        metrics = calc.calculate_trade_metrics([])

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0


# === Report Tests ===

class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_generate_summary_report(
        self,
        backtest_config: BacktestConfig,
        sample_ohlcv_data: pd.DataFrame
    ):
        """Test summary report generation."""
        backtester = Backtester(config=backtest_config)
        result = backtester.run(sample_ohlcv_data, "BTCUSDT", "1h")

        # Calculate metrics
        calc = MetricsCalculator()
        calc.calculate_all(result)

        # Generate report
        generator = ReportGenerator()
        report = generator.generate_summary_report(result)

        assert isinstance(report, str)
        assert "BTCUSDT" in report
        assert "PERFORMANCE SUMMARY" in report

    def test_trade_distribution(
        self,
        backtest_config: BacktestConfig,
        sample_ohlcv_data: pd.DataFrame
    ):
        """Test trade distribution analysis."""
        backtester = Backtester(config=backtest_config)
        result = backtester.run(sample_ohlcv_data)

        generator = ReportGenerator()
        distribution = generator.generate_trade_distribution(result)

        if result.trades:
            assert "mean" in distribution
            assert "median" in distribution
            assert "std" in distribution


# === Integration Tests ===

class TestBacktestIntegration:
    """Integration tests for backtest module."""

    def test_full_backtest_workflow(self, sample_ohlcv_data: pd.DataFrame):
        """Test complete backtest workflow."""
        # 1. Configure
        config = BacktestConfig(
            tp_percents=[1.0, 2.0, 3.0, 4.0, 7.5, 14.0],
            tp_distribution=[17, 17, 17, 17, 16, 16],
            sl_percent=8.5,
        )

        # 2. Run backtest
        backtester = Backtester(config=config)
        result = backtester.run(sample_ohlcv_data, "BTCUSDT", "1h")

        # 3. Calculate metrics
        calc = MetricsCalculator()
        metrics = calc.calculate_all(result)

        # 4. Generate report
        generator = ReportGenerator()
        report = generator.generate_summary_report(result)

        # Verify workflow completed
        assert result is not None
        assert "total_trades" in metrics
        assert len(report) > 0


# === Run tests ===

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
