"""
VELAS Trading System - Backtest Reports

Generates reports and visualizations for backtest results.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np

from backtest.engine import BacktestResult, Trade, TradeResult
from backtest.metrics import MetricsCalculator, TradeMetrics, PortfolioMetrics


class ReportGenerator:
    """
    Generates various reports from backtest results.
    """

    def __init__(self, output_dir: str = "data_store/results") -> None:
        """
        Initialize report generator.

        Args:
            output_dir: Directory for report outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary_report(
        self,
        result: BacktestResult,
    ) -> str:
        """
        Generate text summary report.

        Args:
            result: Backtest result

        Returns:
            Formatted text report
        """
        metrics = result.metrics
        if not metrics:
            calc = MetricsCalculator()
            metrics = calc.calculate_all(result)

        report = []
        report.append("=" * 60)
        report.append(f"BACKTEST REPORT: {result.symbol} {result.timeframe}")
        report.append("=" * 60)
        report.append("")

        # Performance Summary
        report.append("📊 PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Return:      {metrics.get('total_return_pct', 0):>10.2f}%")
        report.append(f"Annualized Return: {metrics.get('annualized_return_pct', 0):>10.2f}%")
        report.append(f"Max Drawdown:      {metrics.get('max_drawdown_pct', 0):>10.2f}%")
        report.append(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):>10.2f}")
        report.append(f"Sortino Ratio:     {metrics.get('sortino_ratio', 0):>10.2f}")
        report.append(f"Profit Factor:     {metrics.get('profit_factor', 0):>10.2f}")
        report.append("")

        # Trade Statistics
        report.append("📈 TRADE STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Trades:      {metrics.get('total_trades', 0):>10}")
        report.append(f"Winning Trades:    {metrics.get('winning_trades', 0):>10}")
        report.append(f"Losing Trades:     {metrics.get('losing_trades', 0):>10}")
        report.append(f"Win Rate:          {metrics.get('win_rate', 0):>10.1f}%")
        report.append("")

        # TP Win Rates
        report.append("🎯 TP HIT RATES")
        report.append("-" * 40)
        report.append(f"TP1 Hit Rate:      {metrics.get('win_rate_tp1', 0):>10.1f}%")
        report.append(f"TP2 Hit Rate:      {metrics.get('win_rate_tp2', 0):>10.1f}%")
        report.append(f"TP3 Hit Rate:      {metrics.get('win_rate_tp3', 0):>10.1f}%")
        report.append("")

        # P&L Statistics
        report.append("💰 P&L STATISTICS")
        report.append("-" * 40)
        report.append(f"Avg Win:           {metrics.get('avg_win_pct', 0):>10.4f}%")
        report.append(f"Avg Loss:          {metrics.get('avg_loss_pct', 0):>10.4f}%")
        report.append(f"Largest Win:       {metrics.get('largest_win_pct', 0):>10.4f}%")
        report.append(f"Largest Loss:      {metrics.get('largest_loss_pct', 0):>10.4f}%")
        report.append(f"Expectancy (R):    {metrics.get('expectancy_r', 0):>10.2f}")
        report.append("")

        # Long/Short Analysis
        report.append("📍 LONG/SHORT ANALYSIS")
        report.append("-" * 40)
        report.append(f"Long Trades:       {metrics.get('long_trades', 0):>10}")
        report.append(f"Short Trades:      {metrics.get('short_trades', 0):>10}")
        report.append(f"Long Win Rate:     {metrics.get('long_win_rate', 0):>10.1f}%")
        report.append(f"Short Win Rate:    {metrics.get('short_win_rate', 0):>10.1f}%")
        report.append("")

        # Duration Statistics
        report.append("⏱️ DURATION STATISTICS")
        report.append("-" * 40)
        report.append(f"Avg Duration:      {metrics.get('avg_duration_bars', 0):>10.1f} bars")
        report.append(f"Max Consec. Wins:  {metrics.get('consecutive_wins_max', 0):>10}")
        report.append(f"Max Consec. Losses:{metrics.get('consecutive_losses_max', 0):>10}")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def generate_trades_csv(
        self,
        result: BacktestResult,
        filename: Optional[str] = None,
    ) -> str:
        """
        Export trades to CSV file.

        Args:
            result: Backtest result
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trades_{result.symbol}_{result.timeframe}_{timestamp}.csv"

        filepath = self.output_dir / filename

        trades_data = [t.to_dict() for t in result.trades]
        df = pd.DataFrame(trades_data)

        df.to_csv(filepath, index=False)
        return str(filepath)

    def generate_equity_csv(
        self,
        result: BacktestResult,
        filename: Optional[str] = None,
    ) -> str:
        """
        Export equity curve to CSV.

        Args:
            result: Backtest result
            filename: Output filename

        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"equity_{result.symbol}_{result.timeframe}_{timestamp}.csv"

        filepath = self.output_dir / filename

        df = pd.DataFrame({
            "equity": result.equity_curve,
            "drawdown": result.drawdown_curve,
        })

        df.to_csv(filepath, index=False)
        return str(filepath)

    def generate_json_report(
        self,
        result: BacktestResult,
        filename: Optional[str] = None,
    ) -> str:
        """
        Export full report to JSON.

        Args:
            result: Backtest result
            filename: Output filename

        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{result.symbol}_{result.timeframe}_{timestamp}.json"

        filepath = self.output_dir / filename

        report_data = {
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "generated_at": datetime.now().isoformat(),
            "metrics": result.metrics,
            "trades_count": len(result.trades),
            "trades": [t.to_dict() for t in result.trades],
        }

        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2)

        return str(filepath)

    def generate_comparison_report(
        self,
        results: Dict[str, Dict[str, BacktestResult]],
    ) -> str:
        """
        Generate comparison report for multiple symbols/timeframes.

        Args:
            results: {symbol: {timeframe: BacktestResult}}

        Returns:
            Formatted comparison report
        """
        calc = MetricsCalculator()
        report = []

        report.append("=" * 80)
        report.append("MULTI-ASSET COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")

        # Header
        header = f"{'Symbol':<12}{'TF':<6}{'Trades':<8}{'WinRate':<10}{'Return':<12}{'MaxDD':<10}{'Sharpe':<8}{'PF':<8}"
        report.append(header)
        report.append("-" * 80)

        all_results = []

        for symbol, timeframes in results.items():
            for tf, result in timeframes.items():
                if not result.metrics:
                    calc.calculate_all(result)

                m = result.metrics
                row = (
                    f"{symbol:<12}"
                    f"{tf:<6}"
                    f"{m.get('total_trades', 0):<8}"
                    f"{m.get('win_rate', 0):<10.1f}"
                    f"{m.get('total_return_pct', 0):<12.2f}"
                    f"{m.get('max_drawdown_pct', 0):<10.2f}"
                    f"{m.get('sharpe_ratio', 0):<8.2f}"
                    f"{m.get('profit_factor', 0):<8.2f}"
                )
                report.append(row)
                all_results.append(result)

        report.append("-" * 80)

        # Portfolio summary
        if all_results:
            total_trades = sum(r.metrics.get("total_trades", 0) for r in all_results)
            avg_win_rate = np.mean([r.metrics.get("win_rate", 0) for r in all_results])
            total_return = sum(r.metrics.get("total_return_pct", 0) for r in all_results)
            max_dd = max(r.metrics.get("max_drawdown_pct", 0) for r in all_results)
            avg_sharpe = np.mean([r.metrics.get("sharpe_ratio", 0) for r in all_results])
            avg_pf = np.mean([r.metrics.get("profit_factor", 0) for r in all_results])

            report.append("")
            report.append("PORTFOLIO SUMMARY")
            report.append("-" * 40)
            report.append(f"Total Trades:      {total_trades:>10}")
            report.append(f"Avg Win Rate:      {avg_win_rate:>10.1f}%")
            report.append(f"Combined Return:   {total_return:>10.2f}%")
            report.append(f"Worst Drawdown:    {max_dd:>10.2f}%")
            report.append(f"Avg Sharpe:        {avg_sharpe:>10.2f}")
            report.append(f"Avg Profit Factor: {avg_pf:>10.2f}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def generate_monthly_returns(
        self,
        result: BacktestResult,
    ) -> pd.DataFrame:
        """
        Generate monthly returns table.

        Args:
            result: Backtest result

        Returns:
            DataFrame with monthly returns
        """
        if not result.trades:
            return pd.DataFrame()

        # Group trades by month
        monthly_pnl = {}

        for trade in result.trades:
            if trade.exit_time:
                month_key = trade.exit_time.strftime("%Y-%m")
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0.0
                monthly_pnl[month_key] += trade.pnl_pct

        # Create DataFrame
        df = pd.DataFrame([
            {"month": k, "return_pct": v}
            for k, v in sorted(monthly_pnl.items())
        ])

        return df

    def generate_trade_distribution(
        self,
        result: BacktestResult,
    ) -> Dict[str, Any]:
        """
        Analyze trade P&L distribution.

        Args:
            result: Backtest result

        Returns:
            Distribution statistics
        """
        if not result.trades:
            return {}

        pnl_values = [t.pnl_pct for t in result.trades]

        return {
            "mean": np.mean(pnl_values),
            "median": np.median(pnl_values),
            "std": np.std(pnl_values),
            "skewness": float(pd.Series(pnl_values).skew()),
            "kurtosis": float(pd.Series(pnl_values).kurtosis()),
            "percentile_5": np.percentile(pnl_values, 5),
            "percentile_25": np.percentile(pnl_values, 25),
            "percentile_75": np.percentile(pnl_values, 75),
            "percentile_95": np.percentile(pnl_values, 95),
            "positive_ratio": sum(1 for p in pnl_values if p > 0) / len(pnl_values) * 100,
        }


def print_quick_stats(result: BacktestResult) -> None:
    """Print quick statistics to console."""
    m = result.metrics
    if not m:
        calc = MetricsCalculator()
        m = calc.calculate_all(result)

    print(f"\n📊 {result.symbol} {result.timeframe}")
    print(f"   Trades: {m.get('total_trades', 0)} | "
          f"WR: {m.get('win_rate', 0):.1f}% | "
          f"PF: {m.get('profit_factor', 0):.2f} | "
          f"Sharpe: {m.get('sharpe_ratio', 0):.2f}")
    print(f"   Return: {m.get('total_return_pct', 0):.2f}% | "
          f"MaxDD: {m.get('max_drawdown_pct', 0):.2f}%")
