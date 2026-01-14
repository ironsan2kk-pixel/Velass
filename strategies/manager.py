"""
Strategy Manager - Manages strategy-pair assignments and execution

Provides:
- Assign strategies to trading pairs
- Run backtests for pair-strategy combinations
- Manage live trading sessions
- Compare strategies on same pair
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseStrategy, StrategyConfig, StrategyResult, SignalType
from .loader import StrategyRegistry, StrategyLoader, load_builtin_strategies
from data.storage import DataStorage

logger = logging.getLogger(__name__)


@dataclass
class BacktestTask:
    """Task for backtest execution."""
    task_id: str
    pair: str
    strategy_name: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class LiveSession:
    """Live trading session."""
    session_id: str
    pair: str
    strategy_name: str
    timeframe: str
    strategy: BaseStrategy
    status: str = "stopped"  # running, stopped, paused
    started_at: Optional[datetime] = None
    is_paper: bool = True


class StrategyManager:
    """
    Central manager for strategies, backtests, and live trading.

    Usage:
        manager = StrategyManager()

        # Register pair-strategy
        manager.assign_strategy("BTCUSDT", "velas", params={...})

        # Run backtest
        result = manager.backtest("BTCUSDT", "velas", days=30)

        # Start live trading
        manager.start_live("BTCUSDT", "velas")
    """

    def __init__(
        self,
        storage: Optional[DataStorage] = None,
        max_workers: int = 4,
    ):
        """
        Initialize strategy manager.

        Args:
            storage: Data storage instance
            max_workers: Max parallel backtest workers
        """
        self.storage = storage or DataStorage()
        self.max_workers = max_workers

        # Load strategies
        self.registry = load_builtin_strategies()
        self.loader = StrategyLoader(self.registry)

        # Active sessions
        self._live_sessions: Dict[str, LiveSession] = {}

        # Backtest queue
        self._backtest_tasks: Dict[str, BacktestTask] = {}

        # Callbacks
        self._on_signal: Optional[Callable[[StrategyResult], None]] = None
        self._on_backtest_complete: Optional[Callable[[Dict], None]] = None

        # Sync strategies to database
        self._sync_strategies_to_db()

    def _sync_strategies_to_db(self) -> None:
        """Sync registered strategies to database."""
        for name in self.registry.list_strategies():
            info = self.registry.get_strategy_info(name)
            if info:
                self.storage.save_strategy({
                    "name": info["name"],
                    "description": info["description"],
                    "default_params": info["default_params"],
                    "params_schema": info["params_schema"],
                    "is_builtin": True,
                    "is_active": True,
                })

    # ==================== Strategy Info ====================

    def list_strategies(self) -> List[Dict[str, Any]]:
        """Get all available strategies with info."""
        strategies = []
        for name in self.registry.list_strategies():
            info = self.registry.get_strategy_info(name)
            if info:
                strategies.append({
                    "name": info["name"],
                    "description": info["description"],
                    "params": list(info["default_params"].keys()),
                })
        return strategies

    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed strategy info."""
        return self.registry.get_strategy_info(name)

    def get_strategy_params(self, name: str) -> Dict[str, Any]:
        """Get strategy default parameters."""
        info = self.registry.get_strategy_info(name)
        if info:
            return info["default_params"]
        return {}

    # ==================== Pair-Strategy Assignment ====================

    def assign_strategy(
        self,
        pair: str,
        strategy_name: str,
        timeframe: str = "1h",
        params: Optional[Dict[str, Any]] = None,
        enable_live: bool = False,
        enable_backtest: bool = True,
        risk_per_trade: float = 0.02,
    ) -> bool:
        """
        Assign a strategy to a trading pair.

        Args:
            pair: Trading pair (e.g., "BTCUSDT")
            strategy_name: Strategy identifier
            timeframe: Timeframe for analysis
            params: Custom parameters (overrides defaults)
            enable_live: Enable live trading
            enable_backtest: Enable backtesting
            risk_per_trade: Risk per trade (0.02 = 2%)

        Returns:
            True if successful
        """
        # Validate strategy exists
        if strategy_name not in self.registry.list_strategies():
            logger.error(f"Strategy '{strategy_name}' not found")
            return False

        # Save configuration
        config = {
            "pair": pair,
            "strategy_name": strategy_name,
            "timeframe": timeframe,
            "params": params or {},
            "is_live": enable_live,
            "is_backtest": enable_backtest,
            "risk_per_trade": risk_per_trade,
        }

        self.storage.save_pair_config(config)
        logger.info(f"Assigned {strategy_name} to {pair} ({timeframe})")
        return True

    def unassign_strategy(self, pair: str, strategy_name: str) -> bool:
        """Remove strategy assignment from pair."""
        # Stop live session if running
        session_key = f"{pair}_{strategy_name}"
        if session_key in self._live_sessions:
            self.stop_live(pair, strategy_name)

        return self.storage.delete_pair_config(pair, strategy_name)

    def get_pair_strategies(self, pair: str) -> List[Dict[str, Any]]:
        """Get all strategies assigned to a pair."""
        return self.storage.get_pair_configs(pair=pair)

    def get_all_assignments(self) -> List[Dict[str, Any]]:
        """Get all pair-strategy assignments."""
        return self.storage.get_pair_configs()

    # ==================== Backtesting ====================

    def backtest(
        self,
        pair: str,
        strategy_name: str,
        days: int = 30,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        params: Optional[Dict[str, Any]] = None,
        initial_balance: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        Run backtest for a pair-strategy combination.

        Args:
            pair: Trading pair
            strategy_name: Strategy to test
            days: Number of days to backtest (if dates not specified)
            start_date: Start date (optional)
            end_date: End date (optional)
            params: Custom parameters
            initial_balance: Starting balance

        Returns:
            Backtest results dictionary
        """
        # Get or create config
        configs = self.storage.get_pair_configs(pair=pair, strategy_name=strategy_name)
        if configs:
            config_data = configs[0]
            timeframe = config_data["timeframe"]
            merged_params = {**config_data["params"], **(params or {})}
        else:
            timeframe = "1h"
            merged_params = params or {}

        # Set dates
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=days)

        # Create strategy
        config = StrategyConfig(
            strategy_name=strategy_name,
            pair=pair,
            timeframe=timeframe,
            params=merged_params,
        )
        strategy = self.registry.create(config)

        # Load data
        df = self.storage.load_ohlcv(pair, timeframe)
        if df is None or df.empty:
            logger.error(f"No data for {pair} {timeframe}")
            return {"error": "No data available"}

        # Filter by date
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        if df.empty:
            return {"error": "No data in date range"}

        # Run backtest
        result = self._run_backtest(strategy, df, initial_balance)

        # Save result
        backtest_id = f"bt_{uuid.uuid4().hex[:8]}"
        result_record = {
            "backtest_id": backtest_id,
            "pair": pair,
            "strategy_name": strategy_name,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "params": merged_params,
            **result,
            "full_metrics": result,
        }
        self.storage.save_backtest_result(result_record)

        if self._on_backtest_complete:
            self._on_backtest_complete(result_record)

        return result_record

    def _run_backtest(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        initial_balance: float,
    ) -> Dict[str, Any]:
        """Execute backtest on data."""
        # Get signals
        df = strategy.backtest_signals(df)

        trades = []
        balance = initial_balance
        position = None
        equity_curve = [initial_balance]

        for i in range(len(df)):
            row = df.iloc[i]
            price = row['close']

            # Check exit conditions if in position
            if position:
                exit_result = self._check_exit(position, row)
                if exit_result:
                    trade = self._close_position(position, exit_result, row)
                    trades.append(trade)
                    balance += trade['pnl']
                    position = None

            # Check entry signal
            if position is None and row.get('signal', 0) != 0:
                signal_type = SignalType.BUY if row['signal'] > 0 else SignalType.SELL
                stop_loss, take_profits = strategy.calculate_tp_sl(
                    df.iloc[:i+1], signal_type, price
                )

                position = {
                    'entry_price': price,
                    'entry_time': row.name,
                    'side': signal_type,
                    'stop_loss': stop_loss,
                    'take_profits': take_profits,
                    'size': balance * strategy.config.risk_per_trade / abs(price - stop_loss),
                }

            equity_curve.append(balance + (self._calc_unrealized_pnl(position, price) if position else 0))

        # Close any open position at end
        if position:
            trade = self._close_position(position, {'type': 'end', 'price': df.iloc[-1]['close']}, df.iloc[-1])
            trades.append(trade)
            balance += trade['pnl']

        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve, initial_balance)

    def _check_exit(self, position: Dict, row: pd.Series) -> Optional[Dict]:
        """Check if position should exit."""
        price = row['close']
        high = row['high']
        low = row['low']

        if position['side'] == SignalType.BUY:
            # Check stop loss
            if low <= position['stop_loss']:
                return {'type': 'sl', 'price': position['stop_loss']}
            # Check take profits
            for tp in position['take_profits']:
                if not tp.hit and high >= tp.price:
                    tp.hit = True
                    return {'type': f'tp{tp.level}', 'price': tp.price}
        else:
            if high >= position['stop_loss']:
                return {'type': 'sl', 'price': position['stop_loss']}
            for tp in position['take_profits']:
                if not tp.hit and low <= tp.price:
                    tp.hit = True
                    return {'type': f'tp{tp.level}', 'price': tp.price}

        return None

    def _close_position(self, position: Dict, exit_result: Dict, row: pd.Series) -> Dict:
        """Close position and return trade record."""
        exit_price = exit_result['price']
        if position['side'] == SignalType.BUY:
            pnl = (exit_price - position['entry_price']) * position['size']
            pnl_pct = (exit_price / position['entry_price'] - 1) * 100
        else:
            pnl = (position['entry_price'] - exit_price) * position['size']
            pnl_pct = (1 - exit_price / position['entry_price']) * 100

        return {
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_time': position['entry_time'],
            'exit_time': row.name,
            'side': position['side'].value,
            'exit_type': exit_result['type'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
        }

    def _calc_unrealized_pnl(self, position: Dict, price: float) -> float:
        """Calculate unrealized P&L."""
        if position is None:
            return 0.0
        if position['side'] == SignalType.BUY:
            return (price - position['entry_price']) * position['size']
        else:
            return (position['entry_price'] - price) * position['size']

    def _calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[float],
        initial_balance: float,
    ) -> Dict[str, Any]:
        """Calculate backtest metrics."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
            }

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]

        win_rate = len(wins) / len(trades) * 100 if trades else 0
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        final_balance = equity_curve[-1] if equity_curve else initial_balance
        total_return = (final_balance / initial_balance - 1) * 100

        # Max drawdown
        peak = initial_balance
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (simplified)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0

        # Sortino (only downside)
        neg_returns = returns[returns < 0]
        sortino = returns.mean() / neg_returns.std() * (252 ** 0.5) if len(neg_returns) > 0 and neg_returns.std() > 0 else 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_return": round(total_return, 2),
            "max_drawdown": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "avg_win": round(gross_profit / len(wins), 2) if wins else 0,
            "avg_loss": round(gross_loss / len(losses), 2) if losses else 0,
            "trades": trades,
        }

    def backtest_multiple(
        self,
        pairs: List[str],
        strategy_name: str,
        days: int = 30,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run backtest for multiple pairs in parallel.

        Returns:
            {pair: results}
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.backtest, pair, strategy_name, days, params=params): pair
                for pair in pairs
            }

            for future in as_completed(futures):
                pair = futures[future]
                try:
                    results[pair] = future.result()
                except Exception as e:
                    logger.error(f"Backtest failed for {pair}: {e}")
                    results[pair] = {"error": str(e)}

        return results

    def compare_strategies(
        self,
        pair: str,
        strategy_names: List[str],
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple strategies on the same pair.

        Returns:
            List of results sorted by total return
        """
        results = []

        for strategy_name in strategy_names:
            try:
                result = self.backtest(pair, strategy_name, days)
                result["strategy_name"] = strategy_name
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to backtest {strategy_name}: {e}")
                results.append({
                    "strategy_name": strategy_name,
                    "error": str(e),
                    "total_return": -999,
                })

        # Sort by total return
        results.sort(key=lambda x: x.get("total_return", -999), reverse=True)
        return results

    # ==================== Live Trading ====================

    def start_live(
        self,
        pair: str,
        strategy_name: str,
        is_paper: bool = True,
    ) -> bool:
        """
        Start live trading session.

        Args:
            pair: Trading pair
            strategy_name: Strategy to use
            is_paper: Paper trading mode

        Returns:
            True if started successfully
        """
        session_key = f"{pair}_{strategy_name}"

        if session_key in self._live_sessions:
            logger.warning(f"Session already exists for {session_key}")
            return False

        # Get config
        configs = self.storage.get_pair_configs(pair=pair, strategy_name=strategy_name)
        if not configs:
            logger.error(f"No config for {pair} {strategy_name}")
            return False

        config_data = configs[0]

        # Create strategy instance
        config = StrategyConfig(
            strategy_name=strategy_name,
            pair=pair,
            timeframe=config_data["timeframe"],
            params=config_data["params"],
            risk_per_trade=config_data["risk_per_trade"],
        )
        strategy = self.registry.create(config)

        # Create session
        session_id = f"live_{uuid.uuid4().hex[:8]}"
        session = LiveSession(
            session_id=session_id,
            pair=pair,
            strategy_name=strategy_name,
            timeframe=config_data["timeframe"],
            strategy=strategy,
            status="running",
            started_at=datetime.utcnow(),
            is_paper=is_paper,
        )

        self._live_sessions[session_key] = session

        # Save to database
        self.storage.save_live_session({
            "session_id": session_id,
            "pair": pair,
            "strategy_name": strategy_name,
            "timeframe": config_data["timeframe"],
            "params": config_data["params"],
            "status": "running",
            "started_at": session.started_at,
            "is_paper": is_paper,
        })

        logger.info(f"Started live session {session_id} for {pair} with {strategy_name}")
        return True

    def stop_live(self, pair: str, strategy_name: str) -> bool:
        """Stop live trading session."""
        session_key = f"{pair}_{strategy_name}"

        if session_key not in self._live_sessions:
            logger.warning(f"No session for {session_key}")
            return False

        session = self._live_sessions[session_key]
        session.status = "stopped"

        # Update database
        self.storage.save_live_session({
            "session_id": session.session_id,
            "status": "stopped",
            "stopped_at": datetime.utcnow(),
        })

        del self._live_sessions[session_key]
        logger.info(f"Stopped live session for {pair}")
        return True

    def get_live_sessions(self) -> List[Dict[str, Any]]:
        """Get all active live sessions."""
        return [
            {
                "session_id": s.session_id,
                "pair": s.pair,
                "strategy_name": s.strategy_name,
                "timeframe": s.timeframe,
                "status": s.status,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "is_paper": s.is_paper,
            }
            for s in self._live_sessions.values()
        ]

    def process_candle(self, pair: str, candle: Dict[str, Any]) -> Optional[StrategyResult]:
        """
        Process new candle for live sessions.

        Args:
            pair: Trading pair
            candle: OHLCV candle data

        Returns:
            Signal result if generated
        """
        for session_key, session in self._live_sessions.items():
            if session.pair == pair and session.status == "running":
                # Build DataFrame from candle
                # In real implementation, maintain rolling window
                df = pd.DataFrame([candle])

                result = session.strategy.analyze(df)
                if result and result.is_valid:
                    if self._on_signal:
                        self._on_signal(result)
                    return result

        return None

    # ==================== Callbacks ====================

    def on_signal(self, callback: Callable[[StrategyResult], None]) -> None:
        """Set callback for new signals."""
        self._on_signal = callback

    def on_backtest_complete(self, callback: Callable[[Dict], None]) -> None:
        """Set callback for backtest completion."""
        self._on_backtest_complete = callback

    # ==================== Results ====================

    def get_backtest_history(
        self,
        pair: Optional[str] = None,
        strategy_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get historical backtest results."""
        return self.storage.get_backtest_results(pair, strategy_name, limit)
