"""
VELAS Trading System - Position Tracker

Tracks and manages live trading positions.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from core.signals import Signal, SignalSide
from core.portfolio import PortfolioManager, Position, PositionStatus
from data.storage import DataStorage


@dataclass
class PositionEvent:
    """Position lifecycle event."""
    event_type: str  # OPEN, TP_HIT, SL_HIT, PARTIAL_CLOSE, CLOSE, UPDATE
    position_id: str
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


class PositionTracker:
    """
    Tracks position lifecycle and emits events.
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        storage: Optional[DataStorage] = None,
    ) -> None:
        """
        Initialize position tracker.

        Args:
            portfolio_manager: Portfolio manager instance
            storage: Storage for persistence
        """
        self.portfolio = portfolio_manager
        self.storage = storage

        # Event history
        self.events: List[PositionEvent] = []
        self._max_events = 1000

        # Callbacks
        self._event_callbacks: List[callable] = []

    def open_position(
        self,
        signal: Signal,
    ) -> Optional[Position]:
        """
        Open new position from signal.

        Args:
            signal: Trading signal

        Returns:
            Position if opened
        """
        position = self.portfolio.open_position(signal)

        if position:
            event = PositionEvent(
                event_type="OPEN",
                position_id=position.id,
                symbol=position.symbol,
                data={
                    "side": position.side.value,
                    "entry_price": position.entry_price,
                    "sl_price": position.sl_price,
                    "tp_levels": position.tp_levels,
                    "size_pct": position.initial_size_pct,
                },
            )
            self._emit_event(event)

            if self.storage:
                self.storage.save_position(position.to_dict())

        return position

    def check_tp_sl(
        self,
        symbol: str,
        high: float,
        low: float,
        current_price: float,
    ) -> List[PositionEvent]:
        """
        Check and process TP/SL hits.

        Args:
            symbol: Trading pair
            high: Bar high
            low: Bar low
            current_price: Current price

        Returns:
            List of events generated
        """
        events = []
        positions = self.portfolio.get_positions_by_symbol(symbol)

        for position in positions:
            if position.status == PositionStatus.CLOSED:
                continue

            # Update current price
            position.update_pnl(current_price)

            # Check Stop Loss
            sl_hit = False
            if position.side == SignalSide.LONG and low <= position.sl_price:
                sl_hit = True
            elif position.side == SignalSide.SHORT and high >= position.sl_price:
                sl_hit = True

            if sl_hit:
                event = self._process_sl_hit(position)
                events.append(event)
                continue

            # Check Take Profits
            for i, tp in enumerate(position.tp_levels):
                if i in position.tps_hit:
                    continue

                tp_hit = False
                if position.side == SignalSide.LONG and high >= tp:
                    tp_hit = True
                elif position.side == SignalSide.SHORT and low <= tp:
                    tp_hit = True

                if tp_hit:
                    event = self._process_tp_hit(position, i, tp)
                    events.append(event)

            # Save updated position
            if self.storage:
                self.storage.save_position(position.to_dict())

        return events

    def _process_tp_hit(
        self,
        position: Position,
        tp_index: int,
        tp_price: float,
    ) -> PositionEvent:
        """Process TP hit."""
        # Calculate P&L for this portion
        if position.side == SignalSide.LONG:
            pnl_pct = ((tp_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_pct = ((position.entry_price - tp_price) / position.entry_price) * 100

        # Get distribution
        tp_distribution = [17, 17, 17, 17, 16, 16]  # Default
        close_pct = tp_distribution[tp_index] if tp_index < len(tp_distribution) else 16

        # Partial close
        self.portfolio.partial_close(
            position.id,
            tp_price,
            close_pct,
            tp_index,
        )

        event = PositionEvent(
            event_type="TP_HIT",
            position_id=position.id,
            symbol=position.symbol,
            data={
                "tp_level": tp_index + 1,
                "tp_price": tp_price,
                "pnl_pct": pnl_pct,
                "closed_pct": close_pct,
                "remaining_pct": position.remaining_size_pct,
            },
        )
        self._emit_event(event)

        return event

    def _process_sl_hit(self, position: Position) -> PositionEvent:
        """Process SL hit."""
        if position.side == SignalSide.LONG:
            pnl_pct = ((position.sl_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_pct = ((position.entry_price - position.sl_price) / position.entry_price) * 100

        # Close position
        self.portfolio.close_position(position.id, position.sl_price, "SL")

        event = PositionEvent(
            event_type="SL_HIT",
            position_id=position.id,
            symbol=position.symbol,
            data={
                "sl_price": position.sl_price,
                "pnl_pct": pnl_pct,
                "tps_hit": position.tps_hit,
                "realized_pnl": position.realized_pnl_pct,
            },
        )
        self._emit_event(event)

        return event

    def update_stop_loss(
        self,
        position_id: str,
        new_sl: float,
        reason: str = "Manual",
    ) -> Optional[PositionEvent]:
        """
        Update position stop loss.

        Args:
            position_id: Position ID
            new_sl: New SL price
            reason: Reason for update

        Returns:
            Event if successful
        """
        position = self.portfolio.update_stop_loss(position_id, new_sl)

        if position:
            event = PositionEvent(
                event_type="UPDATE",
                position_id=position_id,
                symbol=position.symbol,
                data={
                    "field": "sl_price",
                    "old_value": position.sl_price,
                    "new_value": new_sl,
                    "reason": reason,
                },
            )
            self._emit_event(event)

            if self.storage:
                self.storage.save_position(position.to_dict())

            return event

        return None

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "Manual",
    ) -> Optional[PositionEvent]:
        """
        Close position manually.

        Args:
            position_id: Position ID
            exit_price: Exit price
            reason: Close reason

        Returns:
            Event if successful
        """
        position = self.portfolio.close_position(position_id, exit_price, reason)

        if position:
            event = PositionEvent(
                event_type="CLOSE",
                position_id=position_id,
                symbol=position.symbol,
                data={
                    "exit_price": exit_price,
                    "reason": reason,
                    "realized_pnl": position.realized_pnl_pct,
                    "tps_hit": position.tps_hit,
                },
            )
            self._emit_event(event)

            if self.storage:
                self.storage.save_position(position.to_dict())

            return event

        return None

    def _emit_event(self, event: PositionEvent) -> None:
        """Emit event to callbacks."""
        self.events.append(event)

        # Trim events
        if len(self.events) > self._max_events:
            self.events = self.events[-self._max_events:]

        # Notify callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Event callback error: {e}")

    def on_event(self, callback: callable) -> None:
        """Register event callback."""
        self._event_callbacks.append(callback)

    def get_recent_events(
        self,
        limit: int = 50,
        event_type: Optional[str] = None,
    ) -> List[PositionEvent]:
        """Get recent events."""
        events = self.events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def get_position_events(self, position_id: str) -> List[PositionEvent]:
        """Get all events for a position."""
        return [e for e in self.events if e.position_id == position_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        open_positions = self.portfolio.get_open_positions()

        total_pnl = sum(
            p.realized_pnl_pct + p.unrealized_pnl_pct
            for p in self.portfolio.positions.values()
        )

        tp_events = [e for e in self.events if e.event_type == "TP_HIT"]
        sl_events = [e for e in self.events if e.event_type == "SL_HIT"]

        return {
            "open_positions": len(open_positions),
            "total_positions": len(self.portfolio.positions),
            "total_events": len(self.events),
            "tp_hits": len(tp_events),
            "sl_hits": len(sl_events),
            "total_pnl_pct": round(total_pnl, 2),
            "portfolio_heat": round(self.portfolio.get_portfolio_heat(), 2),
        }
