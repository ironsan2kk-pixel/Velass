"""
VELAS Trading System - Telegram Module

Contains Telegram bot, message formatting, and Cornix integration.

Note: Requires python-telegram-bot package.
Import directly from submodules if telegram is not installed:
    from tg_bot.formatter import MessageFormatter
    from tg_bot.cornix import CornixFormatter
"""

# Safe imports (no external dependencies)
from tg_bot.formatter import MessageFormatter
from tg_bot.cornix import CornixFormatter

# TelegramBot requires python-telegram-bot
# Import directly: from tg_bot.bot import TelegramBot
# Strategy Bot: from tg_bot.strategy_bot import StrategyBot, create_bot

__all__ = [
    "MessageFormatter",
    "CornixFormatter",
]
