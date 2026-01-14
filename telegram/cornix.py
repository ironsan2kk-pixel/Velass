"""
VELAS Trading System - Cornix Formatter

Formats signals for Cornix trading bot.
"""

from typing import Optional, List, Dict, Any

from core.signals import Signal, SignalSide


class CornixFormatter:
    """
    Formats signals in Cornix-compatible format.

    Cornix is a trading bot that reads signals from Telegram
    and executes them on exchanges.
    """

    def __init__(
        self,
        exchange: str = "Binance Futures",
        leverage: int = 10,
        margin_type: str = "Cross",
        risk_percent: float = 2.0,
    ) -> None:
        """
        Initialize Cornix formatter.

        Args:
            exchange: Exchange name
            leverage: Default leverage
            margin_type: Margin type (Cross/Isolated)
            risk_percent: Risk per trade
        """
        self.exchange = exchange
        self.leverage = leverage
        self.margin_type = margin_type
        self.risk_percent = risk_percent

    def format_signal(
        self,
        signal: Signal,
        include_stats: bool = True,
    ) -> str:
        """
        Format signal for Cornix.

        Args:
            signal: Trading signal
            include_stats: Include win rate stats

        Returns:
            Cornix-formatted message
        """
        side_emoji = "🟢" if signal.side == SignalSide.LONG else "🔴"
        side_text = "LONG" if signal.side == SignalSide.LONG else "SHORT"

        lines = [
            f"{side_emoji} {side_text} #{signal.symbol}",
            "",
        ]

        # Entry zone
        if signal.entry_zone[0] != signal.entry_zone[1]:
            lines.append(f"Entry: {signal.entry_zone[0]:.4f} - {signal.entry_zone[1]:.4f}")
        else:
            lines.append(f"Entry: {signal.entry_price:.4f}")

        # Leverage
        lines.append(f"Leverage: {self.margin_type} {self.leverage}x")
        lines.append("")

        # Take Profits
        for i, tp in enumerate(signal.tp_levels):
            dist = signal.tp_distribution[i] if i < len(signal.tp_distribution) else 0
            lines.append(f"TP{i + 1}: {tp:.4f} ({dist}%)")

        lines.append("")

        # Stop Loss
        lines.append(f"SL: {signal.sl_price:.4f}")
        lines.append("")

        # Risk and stats
        lines.append(f"Risk: {self.risk_percent}%")

        if include_stats and signal.score > 0:
            lines.append(f"Signal Score: {signal.score:.0f}")

        return "\n".join(lines)

    def format_entry_update(
        self,
        symbol: str,
        new_entry: float,
        reason: str = "Price adjusted",
    ) -> str:
        """Format entry price update."""
        return (
            f"📝 #{symbol} Entry Update\n\n"
            f"New Entry: {new_entry:.4f}\n"
            f"Reason: {reason}"
        )

    def format_tp_update(
        self,
        symbol: str,
        tp_level: int,
        new_price: float,
    ) -> str:
        """Format TP level update."""
        return (
            f"📝 #{symbol} TP{tp_level} Update\n\n"
            f"New TP{tp_level}: {new_price:.4f}"
        )

    def format_sl_update(
        self,
        symbol: str,
        new_sl: float,
        reason: str = "Trailing stop",
    ) -> str:
        """Format SL update."""
        return (
            f"📝 #{symbol} SL Update\n\n"
            f"New SL: {new_sl:.4f}\n"
            f"Reason: {reason}"
        )

    def format_cancel(
        self,
        symbol: str,
        reason: str = "Signal cancelled",
    ) -> str:
        """Format signal cancellation."""
        return (
            f"❌ #{symbol} CANCELLED\n\n"
            f"Reason: {reason}"
        )

    def format_close(
        self,
        symbol: str,
        close_percent: int = 100,
        reason: str = "Manual close",
    ) -> str:
        """Format position close command."""
        return (
            f"🔒 #{symbol} CLOSE {close_percent}%\n\n"
            f"Reason: {reason}"
        )

    def format_breakeven(self, symbol: str) -> str:
        """Format breakeven update."""
        return f"📝 #{symbol} Move SL to Breakeven"


class CornixSignalBuilder:
    """
    Builder for creating Cornix signals with validation.
    """

    def __init__(self) -> None:
        self._symbol: str = ""
        self._side: str = "LONG"
        self._entry_min: float = 0.0
        self._entry_max: float = 0.0
        self._tps: List[tuple] = []  # [(price, percent), ...]
        self._sl: float = 0.0
        self._leverage: int = 10
        self._margin: str = "Cross"

    def symbol(self, symbol: str) -> "CornixSignalBuilder":
        """Set trading symbol."""
        self._symbol = symbol.upper()
        return self

    def long(self) -> "CornixSignalBuilder":
        """Set signal as LONG."""
        self._side = "LONG"
        return self

    def short(self) -> "CornixSignalBuilder":
        """Set signal as SHORT."""
        self._side = "SHORT"
        return self

    def entry(
        self,
        price: float,
        zone_pct: float = 0.0,
    ) -> "CornixSignalBuilder":
        """
        Set entry price/zone.

        Args:
            price: Entry price
            zone_pct: Zone width in percent (optional)
        """
        if zone_pct > 0:
            offset = price * (zone_pct / 100)
            if self._side == "LONG":
                self._entry_min = price
                self._entry_max = price + offset
            else:
                self._entry_min = price - offset
                self._entry_max = price
        else:
            self._entry_min = price
            self._entry_max = price

        return self

    def add_tp(
        self,
        price: float,
        percent: int,
    ) -> "CornixSignalBuilder":
        """Add take profit level."""
        self._tps.append((price, percent))
        return self

    def stop_loss(self, price: float) -> "CornixSignalBuilder":
        """Set stop loss."""
        self._sl = price
        return self

    def leverage(
        self,
        value: int,
        margin: str = "Cross",
    ) -> "CornixSignalBuilder":
        """Set leverage."""
        self._leverage = value
        self._margin = margin
        return self

    def validate(self) -> List[str]:
        """
        Validate signal configuration.

        Returns:
            List of validation errors
        """
        errors = []

        if not self._symbol:
            errors.append("Symbol is required")

        if self._entry_min <= 0:
            errors.append("Entry price is required")

        if not self._tps:
            errors.append("At least one TP is required")

        if self._sl <= 0:
            errors.append("Stop loss is required")

        # Check TP distribution
        total_pct = sum(tp[1] for tp in self._tps)
        if total_pct != 100:
            errors.append(f"TP distribution must equal 100% (got {total_pct}%)")

        # Check TP order
        if self._side == "LONG":
            for i, tp in enumerate(self._tps):
                if tp[0] <= self._entry_min:
                    errors.append(f"TP{i+1} must be above entry for LONG")
            if self._sl >= self._entry_min:
                errors.append("SL must be below entry for LONG")
        else:
            for i, tp in enumerate(self._tps):
                if tp[0] >= self._entry_max:
                    errors.append(f"TP{i+1} must be below entry for SHORT")
            if self._sl <= self._entry_max:
                errors.append("SL must be above entry for SHORT")

        return errors

    def build(self) -> str:
        """
        Build Cornix signal message.

        Returns:
            Formatted signal string

        Raises:
            ValueError: If validation fails
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Signal validation failed: {'; '.join(errors)}")

        side_emoji = "🟢" if self._side == "LONG" else "🔴"

        lines = [
            f"{side_emoji} {self._side} #{self._symbol}",
            "",
        ]

        # Entry
        if self._entry_min != self._entry_max:
            lines.append(f"Entry: {self._entry_min:.4f} - {self._entry_max:.4f}")
        else:
            lines.append(f"Entry: {self._entry_min:.4f}")

        lines.append(f"Leverage: {self._margin} {self._leverage}x")
        lines.append("")

        # TPs
        for i, (price, pct) in enumerate(self._tps):
            lines.append(f"TP{i+1}: {price:.4f} ({pct}%)")

        lines.append("")
        lines.append(f"SL: {self._sl:.4f}")

        return "\n".join(lines)

    def from_signal(self, signal: Signal) -> "CornixSignalBuilder":
        """
        Initialize builder from Signal object.

        Args:
            signal: Trading signal

        Returns:
            Self for chaining
        """
        self._symbol = signal.symbol

        if signal.side == SignalSide.LONG:
            self.long()
        else:
            self.short()

        self._entry_min = signal.entry_zone[0]
        self._entry_max = signal.entry_zone[1]

        self._tps = list(zip(
            signal.tp_levels,
            signal.tp_distribution
        ))

        self._sl = signal.sl_price

        return self
