"""
VELAS Trading System - Message Formatter

Formats trading data into readable messages.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from core.signals import Signal, SignalSide


class MessageFormatter:
    """
    Formats trading data into Telegram messages.
    """

    def __init__(self, use_html: bool = True) -> None:
        """
        Initialize formatter.

        Args:
            use_html: Use HTML formatting
        """
        self.use_html = use_html

    def _bold(self, text: str) -> str:
        """Apply bold formatting."""
        if self.use_html:
            return f"<b>{text}</b>"
        return f"*{text}*"

    def _code(self, text: str) -> str:
        """Apply code formatting."""
        if self.use_html:
            return f"<code>{text}</code>"
        return f"`{text}`"

    def format_signal_basic(self, signal: Signal) -> str:
        """
        Format signal as basic message.

        Args:
            signal: Trading signal

        Returns:
            Formatted message
        """
        side_emoji = "🟢" if signal.side == SignalSide.LONG else "🔴"
        side_text = "LONG" if signal.side == SignalSide.LONG else "SHORT"

        lines = [
            f"{side_emoji} {self._bold(f'{side_text} #{signal.symbol}')}",
            "",
            f"Entry: {self._code(f'{signal.entry_price:.4f}')}",
        ]

        # Entry zone
        if signal.entry_zone[0] != signal.entry_zone[1]:
            lines.append(f"Zone: {signal.entry_zone[0]:.4f} - {signal.entry_zone[1]:.4f}")

        lines.append("")

        # TPs
        for i, tp in enumerate(signal.tp_levels):
            dist = signal.tp_distribution[i] if i < len(signal.tp_distribution) else 0
            lines.append(f"TP{i + 1}: {self._code(f'{tp:.4f}')} ({dist}%)")

        lines.append("")
        lines.append(f"SL: {self._code(f'{signal.sl_price:.4f}')}")

        # Score and metadata
        if signal.score > 0:
            lines.append("")
            lines.append(f"Score: {signal.score:.1f}")

        lines.append("")
        lines.append(f"TF: {signal.timeframe}")

        return "\n".join(lines)

    def format_tp_hit(
        self,
        symbol: str,
        tp_level: int,
        entry_price: float,
        hit_price: float,
        remaining_pct: float,
    ) -> str:
        """Format TP hit notification."""
        pnl_pct = ((hit_price - entry_price) / entry_price) * 100

        lines = [
            f"✅ {self._bold(f'#{symbol} TP{tp_level} Hit!')}",
            "",
            f"Entry: {entry_price:.4f}",
            f"TP{tp_level}: {hit_price:.4f} ({pnl_pct:+.2f}%)",
            "",
            f"Remaining: {remaining_pct:.0f}%",
        ]

        return "\n".join(lines)

    def format_sl_hit(
        self,
        symbol: str,
        entry_price: float,
        sl_price: float,
        side: str,
    ) -> str:
        """Format SL hit notification."""
        if side == "LONG":
            pnl_pct = ((sl_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - sl_price) / entry_price) * 100

        lines = [
            f"🔴 {self._bold(f'#{symbol} Stopped Out')}",
            "",
            f"Entry: {entry_price:.4f}",
            f"SL: {sl_price:.4f} ({pnl_pct:.2f}%)",
            "",
            f"Result: {pnl_pct:.2f}%",
        ]

        return "\n".join(lines)

    def format_position_list(self, positions: List[Dict[str, Any]]) -> str:
        """Format list of open positions."""
        if not positions:
            return f"📍 {self._bold('Open Positions')}\n\nNo open positions"

        lines = [f"📍 {self._bold('Open Positions')}", ""]

        total_pnl = 0

        for pos in positions:
            pnl = pos.get("unrealized_pnl_pct", 0)
            total_pnl += pnl
            emoji = "🟢" if pnl >= 0 else "🔴"

            lines.append(f"{emoji} {self._bold(pos['symbol'])} {pos['side']}")
            lines.append(f"   Entry: {pos['entry_price']:.4f}")
            lines.append(f"   Current: {pos.get('current_price', 0):.4f}")
            lines.append(f"   P&L: {pnl:+.2f}%")
            lines.append("")

        lines.append(f"Total Unrealized: {total_pnl:+.2f}%")

        return "\n".join(lines)

    def format_daily_summary(
        self,
        date: datetime,
        signals: int,
        winners: int,
        losers: int,
        total_pnl: float,
        open_positions: int,
    ) -> str:
        """Format daily trading summary."""
        win_rate = (winners / signals * 100) if signals > 0 else 0
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        date_str = date.strftime("%Y-%m-%d")

        lines = [
            f"📊 {self._bold(f'Daily Summary - {date_str}')}",
            "",
            f"Signals: {signals}",
            f"Winners: {winners} | Losers: {losers}",
            f"Win Rate: {win_rate:.1f}%",
            "",
            f"{pnl_emoji} Total P&L: {total_pnl:+.2f}%",
            f"Open Positions: {open_positions}",
        ]

        return "\n".join(lines)

    def format_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "WARNING",
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format system alert."""
        emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "CRITICAL": "🔴",
        }.get(severity, "⚠️")

        lines = [
            f"{emoji} {self._bold(alert_type.upper())}",
            "",
            message,
        ]

        if details:
            lines.append("")
            lines.append(self._bold("Details:"))
            for key, value in details.items():
                lines.append(f"• {key}: {value}")

        return "\n".join(lines)

    def format_trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        exit_type: str,
        tps_hit: List[int],
    ) -> str:
        """Format trade closed notification."""
        emoji = "✅" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"

        lines = [
            f"{emoji} {self._bold(f'#{symbol} Trade Closed')}",
            "",
            f"Side: {side}",
            f"Entry: {entry_price:.4f}",
            f"Exit: {exit_price:.4f}",
            "",
            f"Exit Type: {exit_type}",
            f"TPs Hit: {', '.join(f'TP{i+1}' for i in tps_hit) if tps_hit else 'None'}",
            "",
            f"P&L: {pnl_pct:+.2f}%",
        ]

        return "\n".join(lines)

    def format_stats(self, stats: Dict[str, Any]) -> str:
        """Format trading statistics."""
        lines = [
            f"📈 {self._bold('Trading Statistics')}",
            "",
            f"Total Trades: {stats.get('total_trades', 0)}",
            f"Win Rate: {stats.get('win_rate', 0):.1f}%",
            f"Profit Factor: {stats.get('profit_factor', 0):.2f}",
            "",
            f"Total Return: {stats.get('total_return_pct', 0):.2f}%",
            f"Max Drawdown: {stats.get('max_drawdown_pct', 0):.2f}%",
            f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}",
        ]

        return "\n".join(lines)
