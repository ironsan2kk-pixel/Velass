"""
VELAS Trading System - Backtest Engine

Full-featured backtesting engine with partial TP closes and position management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np

from core.strategy import VelasStrategy, VelasParams
from core.signals import Signal, SignalSide, SignalGenerator
from core.filters import FilterManager
from core.portfolio import PortfolioManager


class TradeResult(Enum):
    """Trade outcome type."""
    WIN = "WIN"          # At least TP1 hit
    LOSS = "LOSS"        # SL hit before any TP
    BREAKEVEN = "BREAKEVEN"  # Closed at entry


@dataclass
class Trade:
    """Completed trade record."""

    id: str = ""
    symbol: str = ""
    timeframe: str = ""
    side: str = ""  # "LONG" or "SHORT"

    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl_price: float = 0.0
    tp_levels: List[float] = field(default_factory=list)

    # Results
    result: TradeResult = TradeResult.LOSS
    tps_hit: List[int] = field(default_factory=list)
    exit_type: str = ""  # "SL", "TP1", "TP2", etc.

    # P&L
    pnl_pct: float = 0.0
    pnl_r: float = 0.0  # In terms of risk units

    # Timing
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    duration_bars: int = 0

    # Position tracking
    partial_closes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "sl_price": self.sl_price,
            "result": self.result.value,
            "tps_hit": self.tps_hit,
            "exit_type": self.exit_type,
            "pnl_pct": round(self.pnl_pct, 4),
            "pnl_r": round(self.pnl_r, 2),
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "duration_bars": self.duration_bars,
        }


@dataclass
class BacktestConfig:
    """Backtest configuration."""

    # Date range
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Strategy params
    strategy_params: Optional[VelasParams] = None

    # TP/SL
    tp_percents: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 7.5, 14.0])
    tp_distribution: List[int] = field(default_factory=lambda: [17, 17, 17, 17, 16, 16])
    sl_percent: float = 8.5

    # Stop management
    stop_management: str = "cascade"  # "cascade" or "breakeven"
    breakeven_after_tp: int = 1  # Move to BE after this TP

    # Position sizing
    position_size_pct: float = 2.0

    # Filters
    use_filters: bool = True
    filter_config: Optional[Dict[str, Any]] = None

    # Slippage and fees
    slippage_pct: float = 0.05
    fee_pct: float = 0.04  # 0.04% taker fee


@dataclass
class BacktestResult:
    """Backtest results container."""

    config: BacktestConfig = field(default_factory=BacktestConfig)
    symbol: str = ""
    timeframe: str = ""

    # Trade results
    trades: List[Trade] = field(default_factory=list)

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)

    # Summary metrics (populated by MetricsCalculator)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "total_trades": len(self.trades),
            "metrics": self.metrics,
            "trades": [t.to_dict() for t in self.trades],
        }


class Backtester:
    """
    Full-featured backtesting engine.

    Features:
    - Partial position closes at each TP
    - Cascade or breakeven stop management
    - Filter integration
    - Slippage and fee simulation
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        strategy: Optional[VelasStrategy] = None,
    ) -> None:
        """
        Initialize backtester.

        Args:
            config: Backtest configuration
            strategy: Velas strategy instance
        """
        self.config = config or BacktestConfig()

        if strategy:
            self.strategy = strategy
        else:
            self.strategy = VelasStrategy(
                params=self.config.strategy_params,
                tp_percents=self.config.tp_percents,
                tp_distribution=self.config.tp_distribution,
                sl_percent=self.config.sl_percent,
            )

        self.filter_manager = None
        if self.config.use_filters and self.config.filter_config:
            self.filter_manager = FilterManager(self.config.filter_config)

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "1h",
        htf_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df: OHLCV DataFrame
            symbol: Symbol name
            timeframe: Timeframe string
            htf_df: Higher timeframe data for MTF filter

        Returns:
            BacktestResult with trades and metrics
        """
        result = BacktestResult(
            config=self.config,
            symbol=symbol,
            timeframe=timeframe,
        )

        # Filter by date range
        df = self._filter_date_range(df)
        if len(df) < 100:
            return result

        # Calculate indicators and signals
        df = self.strategy.calculate_indicators(df)
        df = self.strategy.generate_entry_signals(df)

        # Initialize tracking
        trades: List[Trade] = []
        equity = [100.0]  # Start with 100%
        current_equity = 100.0

        # Active position tracking
        in_position = False
        position: Optional[Dict[str, Any]] = None

        # Iterate through bars
        for idx in range(1, len(df)):
            row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]
            timestamp = row.name if isinstance(row.name, datetime) else None

            # Update equity curve
            if in_position and position:
                unrealized_pnl = self._calculate_unrealized_pnl(
                    position, row["close"]
                )
                equity.append(current_equity + unrealized_pnl)
            else:
                equity.append(current_equity)

            # Check exit conditions for open position
            if in_position and position:
                exit_result = self._check_exit(position, row)

                if exit_result:
                    trade = self._close_trade(position, exit_result, idx)
                    trades.append(trade)
                    current_equity += trade.pnl_pct
                    in_position = False
                    position = None

            # Check for new entry signal
            if not in_position:
                signal_value = row.get("signal", 0)
                prev_signal = prev_row.get("signal", 0)

                # Only enter on fresh signal
                if signal_value != 0 and signal_value != prev_signal:
                    # Create temporary signal for filter check
                    side = "LONG" if signal_value == 1 else "SHORT"
                    entry_price = row["close"]

                    # Apply slippage
                    if side == "LONG":
                        entry_price *= (1 + self.config.slippage_pct / 100)
                    else:
                        entry_price *= (1 - self.config.slippage_pct / 100)

                    # Get TP/SL levels
                    levels = self.strategy.calculate_tp_sl(
                        entry_price, side, row.get("atr")
                    )

                    # Create signal for filter evaluation
                    temp_signal = Signal(
                        symbol=symbol,
                        timeframe=timeframe,
                        side=SignalSide.LONG if side == "LONG" else SignalSide.SHORT,
                        entry_price=entry_price,
                        tp_levels=levels.tp_levels,
                        sl_price=levels.sl_price,
                    )

                    # Apply filters
                    if self.filter_manager:
                        passed, _ = self.filter_manager.evaluate_signal(
                            temp_signal,
                            df.iloc[:idx + 1],
                            htf_df=htf_df,
                        )
                        if not passed:
                            continue

                    # Open position
                    position = {
                        "id": f"trade_{len(trades) + 1}",
                        "side": side,
                        "entry_price": entry_price,
                        "entry_idx": idx,
                        "entry_time": timestamp,
                        "sl_price": levels.sl_price,
                        "original_sl": levels.sl_price,
                        "tp_levels": levels.tp_levels.copy(),
                        "tp_distribution": self.config.tp_distribution.copy(),
                        "tps_hit": [],
                        "remaining_pct": 100.0,
                        "realized_pnl": 0.0,
                        "partial_closes": [],
                    }
                    in_position = True

        # Close any remaining position at end
        if in_position and position:
            exit_result = {
                "type": "END",
                "price": df.iloc[-1]["close"],
                "idx": len(df) - 1,
            }
            trade = self._close_trade(position, exit_result, len(df) - 1)
            trades.append(trade)
            current_equity += trade.pnl_pct

        result.trades = trades
        result.equity_curve = equity

        # Calculate drawdown curve
        result.drawdown_curve = self._calculate_drawdown(equity)

        return result

    def _filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if self.config.start_date:
            start = pd.to_datetime(self.config.start_date)
            df = df[df.index >= start]

        if self.config.end_date:
            end = pd.to_datetime(self.config.end_date)
            df = df[df.index <= end]

        return df

    def _check_exit(
        self,
        position: Dict[str, Any],
        row: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if position should be exited.

        Returns exit info dict or None.
        """
        high = row["high"]
        low = row["low"]
        side = position["side"]
        remaining = position["remaining_pct"]

        if remaining <= 0:
            return {"type": "CLOSED", "price": row["close"], "idx": 0}

        # Check SL first
        sl_hit = False
        if side == "LONG" and low <= position["sl_price"]:
            sl_hit = True
            exit_price = position["sl_price"]
        elif side == "SHORT" and high >= position["sl_price"]:
            sl_hit = True
            exit_price = position["sl_price"]

        if sl_hit:
            # Apply slippage on SL
            if side == "LONG":
                exit_price *= (1 - self.config.slippage_pct / 100)
            else:
                exit_price *= (1 + self.config.slippage_pct / 100)

            return {"type": "SL", "price": exit_price, "idx": 0}

        # Check TPs
        for i, tp in enumerate(position["tp_levels"]):
            if i in position["tps_hit"]:
                continue

            tp_hit = False
            if side == "LONG" and high >= tp:
                tp_hit = True
            elif side == "SHORT" and low <= tp:
                tp_hit = True

            if tp_hit:
                # Partial close at TP
                tp_dist = position["tp_distribution"][i]
                close_pct = tp_dist

                # Calculate P&L for this portion
                if side == "LONG":
                    pnl_pct = ((tp - position["entry_price"]) / position["entry_price"]) * 100
                else:
                    pnl_pct = ((position["entry_price"] - tp) / position["entry_price"]) * 100

                # Account for fees
                pnl_pct -= self.config.fee_pct * 2  # Entry + exit fee

                portion_pnl = pnl_pct * (close_pct / 100) * (self.config.position_size_pct / 100)

                position["tps_hit"].append(i)
                position["remaining_pct"] -= close_pct
                position["realized_pnl"] += portion_pnl
                position["partial_closes"].append({
                    "tp": i + 1,
                    "price": tp,
                    "pnl_pct": portion_pnl,
                    "closed_pct": close_pct,
                })

                # Update SL based on stop management
                if self.config.stop_management == "cascade":
                    # Move SL to previous TP level or entry
                    if i == 0:
                        position["sl_price"] = position["entry_price"]
                    else:
                        position["sl_price"] = position["tp_levels"][i - 1]

                elif self.config.stop_management == "breakeven":
                    # Move to breakeven after specified TP
                    if i + 1 >= self.config.breakeven_after_tp:
                        position["sl_price"] = position["entry_price"]

                # Check if fully closed
                if position["remaining_pct"] <= 0:
                    return {"type": f"TP{i + 1}_FULL", "price": tp, "idx": 0}

        return None

    def _close_trade(
        self,
        position: Dict[str, Any],
        exit_result: Dict[str, Any],
        exit_idx: int,
    ) -> Trade:
        """Close trade and create Trade record."""
        side = position["side"]
        entry_price = position["entry_price"]
        exit_price = exit_result["price"]
        remaining = position["remaining_pct"]

        # Calculate remaining P&L
        if remaining > 0:
            if side == "LONG":
                remaining_pnl = ((exit_price - entry_price) / entry_price) * 100
            else:
                remaining_pnl = ((entry_price - exit_price) / entry_price) * 100

            remaining_pnl -= self.config.fee_pct * 2
            remaining_pnl *= (remaining / 100) * (self.config.position_size_pct / 100)
            position["realized_pnl"] += remaining_pnl

        total_pnl = position["realized_pnl"]

        # Calculate risk for R multiple
        if side == "LONG":
            initial_risk = ((entry_price - position["original_sl"]) / entry_price) * 100
        else:
            initial_risk = ((position["original_sl"] - entry_price) / entry_price) * 100

        initial_risk *= self.config.position_size_pct / 100
        pnl_r = total_pnl / initial_risk if initial_risk > 0 else 0

        # Determine result
        if len(position["tps_hit"]) > 0:
            result = TradeResult.WIN
        elif abs(total_pnl) < 0.01:
            result = TradeResult.BREAKEVEN
        else:
            result = TradeResult.LOSS

        trade = Trade(
            id=position["id"],
            symbol="",
            timeframe="",
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            sl_price=position["original_sl"],
            tp_levels=position["tp_levels"],
            result=result,
            tps_hit=position["tps_hit"],
            exit_type=exit_result["type"],
            pnl_pct=total_pnl,
            pnl_r=pnl_r,
            entry_time=position.get("entry_time"),
            duration_bars=exit_idx - position["entry_idx"],
            partial_closes=position["partial_closes"],
        )

        return trade

    def _calculate_unrealized_pnl(
        self,
        position: Dict[str, Any],
        current_price: float,
    ) -> float:
        """Calculate unrealized P&L for position."""
        if position["remaining_pct"] <= 0:
            return 0.0

        entry = position["entry_price"]
        side = position["side"]

        if side == "LONG":
            pnl_pct = ((current_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - current_price) / entry) * 100

        return pnl_pct * (position["remaining_pct"] / 100) * (self.config.position_size_pct / 100)

    def _calculate_drawdown(self, equity: List[float]) -> List[float]:
        """Calculate drawdown curve from equity curve."""
        drawdown = []
        peak = equity[0]

        for value in equity:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100 if peak > 0 else 0
            drawdown.append(dd)

        return drawdown

    def run_multi(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]],
        htf_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Dict[str, BacktestResult]]:
        """
        Run backtest on multiple symbols and timeframes.

        Args:
            data: {symbol: {timeframe: DataFrame}}
            htf_data: {symbol: DataFrame} for MTF filter

        Returns:
            {symbol: {timeframe: BacktestResult}}
        """
        results = {}

        for symbol, timeframes in data.items():
            results[symbol] = {}
            htf_df = htf_data.get(symbol) if htf_data else None

            for timeframe, df in timeframes.items():
                result = self.run(df, symbol, timeframe, htf_df)
                results[symbol][timeframe] = result

        return results


class WalkForwardOptimizer:
    """
    Walk-forward optimization for parameter selection.
    """

    def __init__(
        self,
        optimization_window_months: int = 6,
        test_window_months: int = 2,
        step_months: int = 2,
    ) -> None:
        self.opt_window = optimization_window_months
        self.test_window = test_window_months
        self.step = step_months

    def generate_windows(
        self,
        df: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate optimization and test windows.

        Returns:
            List of (optimization_df, test_df) tuples
        """
        windows = []

        if len(df) == 0:
            return windows

        start_date = df.index.min()
        end_date = df.index.max()

        current_start = start_date

        while True:
            opt_end = current_start + pd.DateOffset(months=self.opt_window)
            test_start = opt_end
            test_end = test_start + pd.DateOffset(months=self.test_window)

            if test_end > end_date:
                break

            opt_df = df[(df.index >= current_start) & (df.index < opt_end)]
            test_df = df[(df.index >= test_start) & (df.index < test_end)]

            if len(opt_df) > 50 and len(test_df) > 20:
                windows.append((opt_df, test_df))

            current_start += pd.DateOffset(months=self.step)

        return windows

    def optimize_params(
        self,
        df: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        metric: str = "sharpe",
    ) -> Tuple[VelasParams, Dict[str, Any]]:
        """
        Find optimal parameters using grid search.

        Args:
            df: Training DataFrame
            param_grid: {param_name: [values]} to test
            metric: Optimization metric

        Returns:
            (best_params, results_dict)
        """
        from backtest.metrics import MetricsCalculator

        best_score = float("-inf")
        best_params = VelasParams()
        all_results = []

        # Generate parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for combo in itertools.product(*param_values):
            params_dict = dict(zip(param_names, combo))
            params = VelasParams(**params_dict)

            # Run backtest
            config = BacktestConfig(strategy_params=params)
            backtester = Backtester(config)
            result = backtester.run(df)

            # Calculate metrics
            calc = MetricsCalculator()
            metrics = calc.calculate_all(result)

            score = metrics.get(metric, 0)
            all_results.append({
                "params": params_dict,
                "metrics": metrics,
                "score": score,
            })

            if score > best_score:
                best_score = score
                best_params = params

        return best_params, {"results": all_results, "best_score": best_score}
