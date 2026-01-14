"""
VELAS Trading System - Portfolio Management

Manages positions, correlations, and risk allocation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from enum import Enum
import pandas as pd
import numpy as np

from core.signals import Signal, SignalSide, SignalStatus


class PositionStatus(Enum):
    """Position lifecycle status."""
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"  # Some TPs hit
    CLOSED = "CLOSED"


@dataclass
class Position:
    """Active trading position."""

    id: str = ""
    signal_id: str = ""
    symbol: str = ""
    timeframe: str = ""
    side: SignalSide = SignalSide.LONG

    # Prices
    entry_price: float = 0.0
    current_price: float = 0.0
    sl_price: float = 0.0
    tp_levels: List[float] = field(default_factory=list)

    # Size and P&L
    initial_size_pct: float = 2.0  # % of portfolio
    remaining_size_pct: float = 2.0
    realized_pnl_pct: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Status
    status: PositionStatus = PositionStatus.OPEN
    tps_hit: List[int] = field(default_factory=list)

    # Timestamps
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None

    # Group for correlation
    group: str = ""

    def update_pnl(self, current_price: float) -> None:
        """Update unrealized P&L based on current price."""
        self.current_price = current_price

        if self.side == SignalSide.LONG:
            pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100

        self.unrealized_pnl_pct = pnl_pct * (self.remaining_size_pct / self.initial_size_pct)

    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl_pct + self.unrealized_pnl_pct

    def get_risk_amount(self) -> float:
        """Get current risk in % of portfolio."""
        if self.entry_price == 0:
            return 0.0

        if self.side == SignalSide.LONG:
            risk_pct = ((self.entry_price - self.sl_price) / self.entry_price) * 100
        else:
            risk_pct = ((self.sl_price - self.entry_price) / self.entry_price) * 100

        return risk_pct * (self.remaining_size_pct / 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "id": self.id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "sl_price": self.sl_price,
            "tp_levels": self.tp_levels,
            "initial_size_pct": self.initial_size_pct,
            "remaining_size_pct": self.remaining_size_pct,
            "realized_pnl_pct": self.realized_pnl_pct,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "status": self.status.value,
            "tps_hit": self.tps_hit,
            "opened_at": self.opened_at.isoformat(),
            "group": self.group,
        }


@dataclass
class CorrelationGroup:
    """Group of correlated symbols."""
    name: str
    symbols: List[str] = field(default_factory=list)
    max_positions: int = 2


class PortfolioManager:
    """
    Manages portfolio of positions with correlation and risk controls.
    """

    def __init__(
        self,
        max_positions: int = 5,
        max_per_group: int = 2,
        position_size_pct: float = 2.0,
        max_portfolio_heat: float = 15.0,
        groups: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Initialize portfolio manager.

        Args:
            max_positions: Maximum simultaneous positions
            max_per_group: Maximum positions per correlation group
            position_size_pct: Default position size in % of portfolio
            max_portfolio_heat: Maximum total portfolio risk
            groups: Correlation groups {group_name: [symbols]}
        """
        self.max_positions = max_positions
        self.max_per_group = max_per_group
        self.position_size_pct = position_size_pct
        self.max_portfolio_heat = max_portfolio_heat

        # Initialize correlation groups
        self.groups: Dict[str, CorrelationGroup] = {}
        if groups:
            for name, symbols in groups.items():
                self.groups[name] = CorrelationGroup(
                    name=name,
                    symbols=symbols,
                    max_positions=max_per_group,
                )

        # Symbol to group mapping
        self.symbol_to_group: Dict[str, str] = {}
        for group_name, group in self.groups.items():
            for symbol in group.symbols:
                self.symbol_to_group[symbol] = group_name

        # Active positions
        self.positions: Dict[str, Position] = {}

        # Correlation matrix (updated periodically)
        self.correlation_matrix: Optional[pd.DataFrame] = None

    def get_symbol_group(self, symbol: str) -> str:
        """Get correlation group for symbol."""
        return self.symbol_to_group.get(symbol, "unknown")

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [
            p for p in self.positions.values()
            if p.status in [PositionStatus.OPEN, PositionStatus.PARTIAL]
        ]

    def get_positions_by_group(self, group: str) -> List[Position]:
        """Get open positions for a specific group."""
        return [
            p for p in self.get_open_positions()
            if p.group == group
        ]

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get open positions for a specific symbol."""
        return [
            p for p in self.get_open_positions()
            if p.symbol == symbol
        ]

    def get_portfolio_heat(self) -> float:
        """Calculate total portfolio risk (heat)."""
        return sum(p.get_risk_amount() for p in self.get_open_positions())

    def get_total_exposure(self) -> float:
        """Calculate total position size exposure."""
        return sum(p.remaining_size_pct for p in self.get_open_positions())

    def can_open_position(self, signal: Signal) -> tuple:
        """
        Check if a new position can be opened.

        Args:
            signal: Signal to evaluate

        Returns:
            Tuple of (can_open: bool, reason: str)
        """
        open_positions = self.get_open_positions()

        # Check max positions
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        # Check if already have position in same symbol
        symbol_positions = self.get_positions_by_symbol(signal.symbol)
        if symbol_positions:
            return False, f"Already have position in {signal.symbol}"

        # Check group limits
        group = self.get_symbol_group(signal.symbol)
        if group != "unknown":
            group_positions = self.get_positions_by_group(group)
            if len(group_positions) >= self.max_per_group:
                return False, f"Max positions for group '{group}' reached ({self.max_per_group})"

        # Check portfolio heat
        current_heat = self.get_portfolio_heat()
        new_position_risk = self._estimate_position_risk(signal)
        if current_heat + new_position_risk > self.max_portfolio_heat:
            return False, f"Would exceed max portfolio heat ({self.max_portfolio_heat}%)"

        return True, "OK"

    def _estimate_position_risk(self, signal: Signal) -> float:
        """Estimate risk for a potential position."""
        if signal.entry_price == 0:
            return 0.0

        if signal.side == SignalSide.LONG:
            risk_pct = ((signal.entry_price - signal.sl_price) / signal.entry_price) * 100
        else:
            risk_pct = ((signal.sl_price - signal.entry_price) / signal.entry_price) * 100

        return risk_pct * (self.position_size_pct / 100)

    def open_position(self, signal: Signal) -> Optional[Position]:
        """
        Open a new position from signal.

        Args:
            signal: Trading signal

        Returns:
            Position if opened, None if rejected
        """
        can_open, reason = self.can_open_position(signal)
        if not can_open:
            return None

        position = Position(
            id=f"pos_{signal.id}",
            signal_id=signal.id,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            side=signal.side,
            entry_price=signal.entry_price,
            current_price=signal.entry_price,
            sl_price=signal.sl_price,
            tp_levels=signal.tp_levels.copy(),
            initial_size_pct=self.position_size_pct,
            remaining_size_pct=self.position_size_pct,
            group=self.get_symbol_group(signal.symbol),
        )

        self.positions[position.id] = position
        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_type: str = "MANUAL",
    ) -> Optional[Position]:
        """
        Close a position completely.

        Args:
            position_id: Position ID
            exit_price: Exit price
            exit_type: Exit reason (SL, TP, MANUAL)

        Returns:
            Closed position or None if not found
        """
        if position_id not in self.positions:
            return None

        position = self.positions[position_id]
        position.current_price = exit_price

        # Calculate final P&L
        if position.side == SignalSide.LONG:
            pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_pct = ((position.entry_price - exit_price) / position.entry_price) * 100

        position.realized_pnl_pct += pnl_pct * (position.remaining_size_pct / position.initial_size_pct)
        position.unrealized_pnl_pct = 0.0
        position.remaining_size_pct = 0.0
        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.now()

        return position

    def partial_close(
        self,
        position_id: str,
        exit_price: float,
        close_pct: float,
        tp_index: int,
    ) -> Optional[Position]:
        """
        Partially close a position at TP.

        Args:
            position_id: Position ID
            exit_price: Exit price
            close_pct: Percentage of position to close
            tp_index: TP level index

        Returns:
            Updated position or None if not found
        """
        if position_id not in self.positions:
            return None

        position = self.positions[position_id]

        # Calculate P&L for this portion
        if position.side == SignalSide.LONG:
            pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_pct = ((position.entry_price - exit_price) / position.entry_price) * 100

        # Add to realized P&L
        portion_of_initial = (close_pct / 100) * (position.remaining_size_pct / position.initial_size_pct)
        position.realized_pnl_pct += pnl_pct * portion_of_initial

        # Reduce remaining size
        position.remaining_size_pct *= (1 - close_pct / 100)
        position.tps_hit.append(tp_index)
        position.current_price = exit_price

        if position.remaining_size_pct < 0.01:
            position.status = PositionStatus.CLOSED
            position.closed_at = datetime.now()
        else:
            position.status = PositionStatus.PARTIAL

        return position

    def update_stop_loss(
        self,
        position_id: str,
        new_sl: float,
    ) -> Optional[Position]:
        """
        Update position stop loss (for trailing/breakeven).

        Args:
            position_id: Position ID
            new_sl: New stop loss price

        Returns:
            Updated position or None if not found
        """
        if position_id not in self.positions:
            return None

        position = self.positions[position_id]
        position.sl_price = new_sl
        return position

    def update_all_prices(
        self,
        prices: Dict[str, float],
    ) -> None:
        """
        Update current prices for all positions.

        Args:
            prices: {symbol: current_price}
        """
        for position in self.get_open_positions():
            if position.symbol in prices:
                position.update_pnl(prices[position.symbol])

    def prioritize_signals(
        self,
        signals: List[Signal],
    ) -> List[Signal]:
        """
        Prioritize signals when we can't take all.

        Args:
            signals: List of candidate signals

        Returns:
            Prioritized list of signals
        """
        # Filter out signals we can't take
        valid_signals = []
        for signal in signals:
            can_open, _ = self.can_open_position(signal)
            if can_open:
                valid_signals.append(signal)

        # Sort by score (higher is better)
        valid_signals.sort(key=lambda s: s.score, reverse=True)

        # Limit to available slots
        available_slots = self.max_positions - len(self.get_open_positions())
        return valid_signals[:available_slots]

    def calculate_correlation_matrix(
        self,
        price_data: Dict[str, pd.DataFrame],
        lookback: int = 50,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for all symbols.

        Args:
            price_data: {symbol: OHLCV DataFrame}
            lookback: Lookback period for correlation

        Returns:
            Correlation matrix DataFrame
        """
        returns = {}

        for symbol, df in price_data.items():
            if len(df) >= lookback:
                returns[symbol] = df["close"].pct_change().tail(lookback)

        if not returns:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns)
        self.correlation_matrix = returns_df.corr()
        return self.correlation_matrix

    def find_correlated_pairs(
        self,
        threshold: float = 0.7,
    ) -> List[tuple]:
        """
        Find highly correlated symbol pairs.

        Args:
            threshold: Correlation threshold

        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        if self.correlation_matrix is None:
            return []

        correlated = []
        symbols = self.correlation_matrix.columns.tolist()

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                corr = self.correlation_matrix.loc[sym1, sym2]
                if abs(corr) >= threshold:
                    correlated.append((sym1, sym2, corr))

        return sorted(correlated, key=lambda x: abs(x[2]), reverse=True)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary statistics."""
        open_positions = self.get_open_positions()

        total_realized = sum(p.realized_pnl_pct for p in self.positions.values())
        total_unrealized = sum(p.unrealized_pnl_pct for p in open_positions)

        by_group = {}
        for group_name in self.groups:
            group_positions = self.get_positions_by_group(group_name)
            by_group[group_name] = {
                "count": len(group_positions),
                "exposure": sum(p.remaining_size_pct for p in group_positions),
            }

        return {
            "open_positions": len(open_positions),
            "max_positions": self.max_positions,
            "total_exposure": self.get_total_exposure(),
            "portfolio_heat": self.get_portfolio_heat(),
            "max_heat": self.max_portfolio_heat,
            "total_realized_pnl": round(total_realized, 2),
            "total_unrealized_pnl": round(total_unrealized, 2),
            "total_pnl": round(total_realized + total_unrealized, 2),
            "by_group": by_group,
            "positions": [p.to_dict() for p in open_positions],
        }
