"""
VELAS Trading System - Backtest Metrics

Calculates performance metrics for backtest results.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from backtest.engine import Trade, TradeResult, BacktestResult


@dataclass
class TradeMetrics:
    """Trade-level metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    win_rate: float = 0.0
    win_rate_tp1: float = 0.0
    win_rate_tp2: float = 0.0
    win_rate_tp3: float = 0.0

    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_pnl_pct: float = 0.0

    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    expectancy_r: float = 0.0

    largest_win_pct: float = 0.0
    largest_loss_pct: float = 0.0

    avg_duration_bars: float = 0.0
    avg_winner_duration: float = 0.0
    avg_loser_duration: float = 0.0

    consecutive_wins_max: int = 0
    consecutive_losses_max: int = 0

    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "win_rate_tp1": round(self.win_rate_tp1, 2),
            "win_rate_tp2": round(self.win_rate_tp2, 2),
            "win_rate_tp3": round(self.win_rate_tp3, 2),
            "avg_win_pct": round(self.avg_win_pct, 4),
            "avg_loss_pct": round(self.avg_loss_pct, 4),
            "expectancy_r": round(self.expectancy_r, 2),
            "largest_win_pct": round(self.largest_win_pct, 4),
            "largest_loss_pct": round(self.largest_loss_pct, 4),
            "avg_duration_bars": round(self.avg_duration_bars, 1),
            "consecutive_wins_max": self.consecutive_wins_max,
            "consecutive_losses_max": self.consecutive_losses_max,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "long_win_rate": round(self.long_win_rate, 2),
            "short_win_rate": round(self.short_win_rate, 2),
        }


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics."""

    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0

    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    profit_factor: float = 0.0
    recovery_factor: float = 0.0

    volatility_pct: float = 0.0
    downside_volatility_pct: float = 0.0

    risk_adjusted_return: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return_pct": round(self.total_return_pct, 2),
            "annualized_return_pct": round(self.annualized_return_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "avg_drawdown_pct": round(self.avg_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "profit_factor": round(self.profit_factor, 2),
            "recovery_factor": round(self.recovery_factor, 2),
            "volatility_pct": round(self.volatility_pct, 2),
        }


class MetricsCalculator:
    """
    Calculates comprehensive performance metrics.
    """

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 0)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_trade_metrics(self, trades: List[Trade]) -> TradeMetrics:
        """Calculate trade-level metrics."""
        metrics = TradeMetrics()

        if not trades:
            return metrics

        metrics.total_trades = len(trades)

        # Categorize trades
        winners = [t for t in trades if t.result == TradeResult.WIN]
        losers = [t for t in trades if t.result == TradeResult.LOSS]
        breakevens = [t for t in trades if t.result == TradeResult.BREAKEVEN]

        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)
        metrics.breakeven_trades = len(breakevens)

        # Win rates
        if metrics.total_trades > 0:
            metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100

        # Win rates by TP level
        tp1_hits = [t for t in trades if 0 in t.tps_hit]
        tp2_hits = [t for t in trades if 1 in t.tps_hit]
        tp3_hits = [t for t in trades if 2 in t.tps_hit]

        if metrics.total_trades > 0:
            metrics.win_rate_tp1 = (len(tp1_hits) / metrics.total_trades) * 100
            metrics.win_rate_tp2 = (len(tp2_hits) / metrics.total_trades) * 100
            metrics.win_rate_tp3 = (len(tp3_hits) / metrics.total_trades) * 100

        # P&L statistics
        all_pnl = [t.pnl_pct for t in trades]
        win_pnl = [t.pnl_pct for t in winners]
        loss_pnl = [t.pnl_pct for t in losers]

        if all_pnl:
            metrics.avg_pnl_pct = np.mean(all_pnl)
        if win_pnl:
            metrics.avg_win_pct = np.mean(win_pnl)
            metrics.largest_win_pct = max(win_pnl)
        if loss_pnl:
            metrics.avg_loss_pct = np.mean(loss_pnl)
            metrics.largest_loss_pct = min(loss_pnl)

        # R-multiple statistics
        win_r = [t.pnl_r for t in winners if t.pnl_r != 0]
        loss_r = [t.pnl_r for t in losers if t.pnl_r != 0]

        if win_r:
            metrics.avg_win_r = np.mean(win_r)
        if loss_r:
            metrics.avg_loss_r = np.mean(loss_r)

        # Expectancy
        if metrics.total_trades > 0:
            win_prob = metrics.winning_trades / metrics.total_trades
            loss_prob = metrics.losing_trades / metrics.total_trades
            metrics.expectancy_r = (
                win_prob * metrics.avg_win_r + loss_prob * metrics.avg_loss_r
            )

        # Duration statistics
        durations = [t.duration_bars for t in trades]
        winner_durations = [t.duration_bars for t in winners]
        loser_durations = [t.duration_bars for t in losers]

        if durations:
            metrics.avg_duration_bars = np.mean(durations)
        if winner_durations:
            metrics.avg_winner_duration = np.mean(winner_durations)
        if loser_durations:
            metrics.avg_loser_duration = np.mean(loser_durations)

        # Consecutive wins/losses
        metrics.consecutive_wins_max = self._max_consecutive(
            [t.result == TradeResult.WIN for t in trades]
        )
        metrics.consecutive_losses_max = self._max_consecutive(
            [t.result == TradeResult.LOSS for t in trades]
        )

        # Long/Short breakdown
        long_trades = [t for t in trades if t.side == "LONG"]
        short_trades = [t for t in trades if t.side == "SHORT"]

        metrics.long_trades = len(long_trades)
        metrics.short_trades = len(short_trades)

        long_winners = [t for t in long_trades if t.result == TradeResult.WIN]
        short_winners = [t for t in short_trades if t.result == TradeResult.WIN]

        if long_trades:
            metrics.long_win_rate = (len(long_winners) / len(long_trades)) * 100
        if short_trades:
            metrics.short_win_rate = (len(short_winners) / len(short_trades)) * 100

        return metrics

    def calculate_portfolio_metrics(
        self,
        equity_curve: List[float],
        drawdown_curve: List[float],
        trades: List[Trade],
        periods_per_year: int = 252 * 24,  # Hourly bars
    ) -> PortfolioMetrics:
        """Calculate portfolio-level metrics."""
        metrics = PortfolioMetrics()

        if not equity_curve or len(equity_curve) < 2:
            return metrics

        equity = np.array(equity_curve)
        drawdown = np.array(drawdown_curve)

        # Returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) == 0:
            return metrics

        # Total return
        metrics.total_return_pct = ((equity[-1] - equity[0]) / equity[0]) * 100

        # Annualized return
        n_periods = len(equity)
        if n_periods > 1:
            total_return = equity[-1] / equity[0]
            years = n_periods / periods_per_year
            if years > 0:
                metrics.annualized_return_pct = (
                    (total_return ** (1 / years) - 1) * 100
                )

        # Drawdown metrics
        metrics.max_drawdown_pct = np.max(drawdown)
        metrics.avg_drawdown_pct = np.mean(drawdown[drawdown > 0]) if np.any(drawdown > 0) else 0

        # Drawdown duration
        in_drawdown = False
        current_duration = 0
        max_duration = 0
        for dd in drawdown:
            if dd > 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                in_drawdown = False
                current_duration = 0
        metrics.max_drawdown_duration = max_duration

        # Volatility
        metrics.volatility_pct = np.std(returns) * np.sqrt(periods_per_year) * 100

        # Downside volatility
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics.downside_volatility_pct = (
                np.std(negative_returns) * np.sqrt(periods_per_year) * 100
            )

        # Sharpe Ratio
        if metrics.volatility_pct > 0:
            excess_return = metrics.annualized_return_pct - self.risk_free_rate
            metrics.sharpe_ratio = excess_return / metrics.volatility_pct

        # Sortino Ratio
        if metrics.downside_volatility_pct > 0:
            excess_return = metrics.annualized_return_pct - self.risk_free_rate
            metrics.sortino_ratio = excess_return / metrics.downside_volatility_pct

        # Calmar Ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return_pct / metrics.max_drawdown_pct

        # Profit Factor
        gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
        gross_loss = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss

        # Recovery Factor
        if metrics.max_drawdown_pct > 0:
            metrics.recovery_factor = metrics.total_return_pct / metrics.max_drawdown_pct

        # Risk-adjusted return
        if metrics.max_drawdown_pct > 0:
            metrics.risk_adjusted_return = (
                metrics.total_return_pct / metrics.max_drawdown_pct
            )

        return metrics

    def calculate_all(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Calculate all metrics for a backtest result.

        Returns combined metrics dictionary.
        """
        trade_metrics = self.calculate_trade_metrics(result.trades)
        portfolio_metrics = self.calculate_portfolio_metrics(
            result.equity_curve,
            result.drawdown_curve,
            result.trades,
        )

        combined = {}
        combined.update(trade_metrics.to_dict())
        combined.update(portfolio_metrics.to_dict())

        result.metrics = combined
        return combined

    def _max_consecutive(self, values: List[bool]) -> int:
        """Find maximum consecutive True values."""
        max_count = 0
        current_count = 0

        for v in values:
            if v:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def compare_results(
        self,
        results: List[BacktestResult],
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """
        Compare multiple backtest results.

        Args:
            results: List of BacktestResult
            metric: Primary comparison metric

        Returns:
            DataFrame with comparison
        """
        comparison = []

        for result in results:
            if not result.metrics:
                self.calculate_all(result)

            row = {
                "symbol": result.symbol,
                "timeframe": result.timeframe,
                "total_trades": result.metrics.get("total_trades", 0),
                "win_rate": result.metrics.get("win_rate", 0),
                "total_return": result.metrics.get("total_return_pct", 0),
                "max_drawdown": result.metrics.get("max_drawdown_pct", 0),
                "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
                "profit_factor": result.metrics.get("profit_factor", 0),
            }
            comparison.append(row)

        df = pd.DataFrame(comparison)
        if not df.empty:
            df = df.sort_values(metric, ascending=False)

        return df


def calculate_monte_carlo(
    trades: List[Trade],
    n_simulations: int = 1000,
    confidence_levels: List[float] = None,
) -> Dict[str, Any]:
    """
    Monte Carlo simulation for risk analysis.

    Args:
        trades: List of completed trades
        n_simulations: Number of simulations
        confidence_levels: Confidence levels for VaR

    Returns:
        Monte Carlo analysis results
    """
    if not trades:
        return {}

    confidence_levels = confidence_levels or [0.95, 0.99]
    pnl_values = [t.pnl_pct for t in trades]
    n_trades = len(pnl_values)

    # Run simulations
    final_returns = []

    for _ in range(n_simulations):
        # Random sampling with replacement
        sampled_pnl = np.random.choice(pnl_values, size=n_trades, replace=True)
        cumulative = np.cumsum(sampled_pnl)
        final_returns.append(cumulative[-1])

    final_returns = np.array(final_returns)

    # Calculate statistics
    results = {
        "mean_return": np.mean(final_returns),
        "median_return": np.median(final_returns),
        "std_return": np.std(final_returns),
        "min_return": np.min(final_returns),
        "max_return": np.max(final_returns),
        "positive_prob": np.mean(final_returns > 0) * 100,
    }

    # Value at Risk
    for level in confidence_levels:
        percentile = (1 - level) * 100
        var = np.percentile(final_returns, percentile)
        results[f"var_{int(level * 100)}"] = var

    return results
