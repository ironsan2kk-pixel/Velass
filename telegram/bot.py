"""
VELAS Trading System - Telegram Bot

Handles sending signals and notifications to Telegram channels.
"""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from telegram import Bot
from telegram.error import TelegramError
from telegram.constants import ParseMode


class TelegramBot:
    """
    Telegram bot for sending trading signals and notifications.

    Supports:
    - Signal messages in Cornix format
    - TP/SL hit notifications
    - System alerts
    - Position updates
    """

    def __init__(
        self,
        token: str,
        channel_id: str,
        alert_channel_id: Optional[str] = None,
    ) -> None:
        """
        Initialize Telegram bot.

        Args:
            token: Telegram bot token
            channel_id: Main signal channel ID
            alert_channel_id: Separate channel for alerts (optional)
        """
        self.token = token
        self.channel_id = channel_id
        self.alert_channel_id = alert_channel_id or channel_id

        self.bot = Bot(token=token)
        self._rate_limit_delay = 0.05  # 50ms between messages

    async def send_message(
        self,
        text: str,
        channel_id: Optional[str] = None,
        parse_mode: str = ParseMode.HTML,
        disable_notification: bool = False,
    ) -> Optional[int]:
        """
        Send message to channel.

        Args:
            text: Message text
            channel_id: Target channel (defaults to main channel)
            parse_mode: Message parse mode
            disable_notification: Silent message

        Returns:
            Message ID if successful, None otherwise
        """
        target_channel = channel_id or self.channel_id

        try:
            message = await self.bot.send_message(
                chat_id=target_channel,
                text=text,
                parse_mode=parse_mode,
                disable_notification=disable_notification,
            )
            await asyncio.sleep(self._rate_limit_delay)
            return message.message_id

        except TelegramError as e:
            print(f"Telegram error: {e}")
            return None

    async def send_signal(
        self,
        signal_text: str,
        pin_message: bool = False,
    ) -> Optional[int]:
        """
        Send trading signal to channel.

        Args:
            signal_text: Formatted signal message
            pin_message: Whether to pin the message

        Returns:
            Message ID if successful
        """
        message_id = await self.send_message(signal_text)

        if message_id and pin_message:
            try:
                await self.bot.pin_chat_message(
                    chat_id=self.channel_id,
                    message_id=message_id,
                    disable_notification=True,
                )
            except TelegramError:
                pass

        return message_id

    async def send_tp_notification(
        self,
        symbol: str,
        tp_level: int,
        entry_price: float,
        tp_price: float,
        pnl_pct: float,
        remaining_pct: float,
    ) -> Optional[int]:
        """
        Send TP hit notification.

        Args:
            symbol: Trading pair
            tp_level: TP level hit (1-6)
            entry_price: Entry price
            tp_price: TP price
            pnl_pct: Profit percentage
            remaining_pct: Remaining position

        Returns:
            Message ID
        """
        text = (
            f"✅ <b>#{symbol} TP{tp_level} Hit!</b>\n\n"
            f"Entry: {entry_price}\n"
            f"TP{tp_level}: {tp_price} (+{pnl_pct:.2f}%)\n\n"
            f"Remaining: {remaining_pct:.0f}%"
        )

        return await self.send_message(text)

    async def send_sl_notification(
        self,
        symbol: str,
        entry_price: float,
        sl_price: float,
        pnl_pct: float,
    ) -> Optional[int]:
        """
        Send SL hit notification.

        Args:
            symbol: Trading pair
            entry_price: Entry price
            sl_price: SL price
            pnl_pct: Loss percentage

        Returns:
            Message ID
        """
        text = (
            f"🔴 <b>#{symbol} Stopped Out</b>\n\n"
            f"Entry: {entry_price}\n"
            f"SL: {sl_price} ({pnl_pct:.2f}%)\n\n"
            f"Result: {pnl_pct:.2f}%"
        )

        return await self.send_message(text)

    async def send_alert(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "WARNING",
    ) -> Optional[int]:
        """
        Send system alert.

        Args:
            alert_type: Type of alert
            message: Alert message
            details: Additional details
            severity: Alert severity (INFO, WARNING, CRITICAL)

        Returns:
            Message ID
        """
        emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "CRITICAL": "🔴",
        }.get(severity, "⚠️")

        text = f"{emoji} <b>{alert_type.upper()}</b>\n\n{message}"

        if details:
            text += "\n\n<b>Details:</b>\n"
            for key, value in details.items():
                text += f"• {key}: {value}\n"

        return await self.send_message(
            text,
            channel_id=self.alert_channel_id,
        )

    async def send_daily_summary(
        self,
        date: datetime,
        total_signals: int,
        winners: int,
        losers: int,
        total_pnl: float,
        open_positions: int,
    ) -> Optional[int]:
        """
        Send daily trading summary.

        Args:
            date: Summary date
            total_signals: Total signals generated
            winners: Winning trades
            losers: Losing trades
            total_pnl: Total P&L
            open_positions: Current open positions

        Returns:
            Message ID
        """
        win_rate = (winners / total_signals * 100) if total_signals > 0 else 0
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"

        text = (
            f"📊 <b>Daily Summary - {date.strftime('%Y-%m-%d')}</b>\n\n"
            f"Signals: {total_signals}\n"
            f"Winners: {winners} | Losers: {losers}\n"
            f"Win Rate: {win_rate:.1f}%\n\n"
            f"{pnl_emoji} Total P&L: {total_pnl:+.2f}%\n"
            f"Open Positions: {open_positions}"
        )

        return await self.send_message(text)

    async def send_position_update(
        self,
        positions: List[Dict[str, Any]],
    ) -> Optional[int]:
        """
        Send current positions status.

        Args:
            positions: List of position dictionaries

        Returns:
            Message ID
        """
        if not positions:
            text = "📍 <b>Open Positions</b>\n\nNo open positions"
        else:
            text = "📍 <b>Open Positions</b>\n\n"

            for pos in positions:
                pnl = pos.get("unrealized_pnl_pct", 0)
                emoji = "🟢" if pnl >= 0 else "🔴"

                text += (
                    f"{emoji} <b>{pos['symbol']}</b> {pos['side']}\n"
                    f"   Entry: {pos['entry_price']:.4f}\n"
                    f"   P&L: {pnl:+.2f}%\n\n"
                )

        return await self.send_message(text)

    async def test_connection(self) -> bool:
        """Test bot connection."""
        try:
            me = await self.bot.get_me()
            print(f"Bot connected: @{me.username}")
            return True
        except TelegramError as e:
            print(f"Connection test failed: {e}")
            return False


class TelegramNotifier:
    """
    High-level notifier that integrates with trading system.
    """

    def __init__(
        self,
        bot: TelegramBot,
        enabled: bool = True,
        send_signals: bool = True,
        send_tp_notifications: bool = True,
        send_sl_notifications: bool = True,
        send_alerts: bool = True,
    ) -> None:
        """
        Initialize notifier.

        Args:
            bot: TelegramBot instance
            enabled: Global enable/disable
            send_signals: Send signal messages
            send_tp_notifications: Send TP hit messages
            send_sl_notifications: Send SL hit messages
            send_alerts: Send alert messages
        """
        self.bot = bot
        self.enabled = enabled
        self.send_signals = send_signals
        self.send_tp_notifications = send_tp_notifications
        self.send_sl_notifications = send_sl_notifications
        self.send_alerts = send_alerts

        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self) -> None:
        """Start notification queue processor."""
        self._running = True
        asyncio.create_task(self._process_queue())

    async def stop(self) -> None:
        """Stop notification queue processor."""
        self._running = False

    async def _process_queue(self) -> None:
        """Process queued notifications."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )

                msg_type = message.get("type")
                data = message.get("data", {})

                if msg_type == "signal" and self.send_signals:
                    await self.bot.send_signal(data.get("text", ""))

                elif msg_type == "tp" and self.send_tp_notifications:
                    await self.bot.send_tp_notification(**data)

                elif msg_type == "sl" and self.send_sl_notifications:
                    await self.bot.send_sl_notification(**data)

                elif msg_type == "alert" and self.send_alerts:
                    await self.bot.send_alert(**data)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Notification error: {e}")

    def queue_signal(self, signal_text: str) -> None:
        """Queue signal notification."""
        if self.enabled:
            self._message_queue.put_nowait({
                "type": "signal",
                "data": {"text": signal_text},
            })

    def queue_tp_notification(self, **kwargs: Any) -> None:
        """Queue TP notification."""
        if self.enabled:
            self._message_queue.put_nowait({
                "type": "tp",
                "data": kwargs,
            })

    def queue_sl_notification(self, **kwargs: Any) -> None:
        """Queue SL notification."""
        if self.enabled:
            self._message_queue.put_nowait({
                "type": "sl",
                "data": kwargs,
            })

    def queue_alert(self, **kwargs: Any) -> None:
        """Queue alert notification."""
        if self.enabled:
            self._message_queue.put_nowait({
                "type": "alert",
                "data": kwargs,
            })
