"""
Strategy Bot - Telegram Bot for Strategy Management

Commands:
- /start - Welcome message
- /help - List all commands
- /strategies - List available strategies
- /strategy <name> - Get strategy details
- /assign <pair> <strategy> - Assign strategy to pair
- /unassign <pair> <strategy> - Remove assignment
- /list - List all assignments
- /backtest <pair> <strategy> [days] - Run backtest
- /compare <pair> <strategy1,strategy2,...> - Compare strategies
- /results [pair] - Show recent backtest results
- /live start <pair> <strategy> - Start live trading
- /live stop <pair> <strategy> - Stop live trading
- /live status - Show active sessions
- /pnl - Show P&L summary
- /status - System status
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from functools import wraps

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        CallbackQueryHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None
    ContextTypes = None

from strategies.manager import StrategyManager
from data.storage import DataStorage

logger = logging.getLogger(__name__)


def telegram_required(func):
    """Decorator to check if telegram is available."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not TELEGRAM_AVAILABLE:
            logger.error("Telegram library not available")
            return None
        return await func(*args, **kwargs)
    return wrapper


class StrategyBot:
    """
    Telegram bot for managing trading strategies.

    Features:
    - List and inspect strategies
    - Assign strategies to pairs
    - Run backtests
    - Compare strategies
    - Manage live trading sessions
    - View results and P&L
    """

    def __init__(
        self,
        token: str,
        allowed_users: Optional[List[int]] = None,
        manager: Optional[StrategyManager] = None,
    ):
        """
        Initialize Strategy Bot.

        Args:
            token: Telegram bot token
            allowed_users: List of allowed user IDs (None = all)
            manager: Strategy manager instance
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot not installed")

        self.token = token
        self.allowed_users = allowed_users or []
        self.manager = manager or StrategyManager()

        # Running backtest tasks
        self._running_tasks: Dict[str, asyncio.Task] = {}

        # Build application
        self.app = Application.builder().token(token).build()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup command handlers."""
        # Basic commands
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("status", self.cmd_status))

        # Strategy commands
        self.app.add_handler(CommandHandler("strategies", self.cmd_strategies))
        self.app.add_handler(CommandHandler("strategy", self.cmd_strategy_info))

        # Assignment commands
        self.app.add_handler(CommandHandler("assign", self.cmd_assign))
        self.app.add_handler(CommandHandler("unassign", self.cmd_unassign))
        self.app.add_handler(CommandHandler("list", self.cmd_list_assignments))

        # Backtest commands
        self.app.add_handler(CommandHandler("backtest", self.cmd_backtest))
        self.app.add_handler(CommandHandler("compare", self.cmd_compare))
        self.app.add_handler(CommandHandler("results", self.cmd_results))

        # Live trading commands
        self.app.add_handler(CommandHandler("live", self.cmd_live))
        self.app.add_handler(CommandHandler("pnl", self.cmd_pnl))

        # Callback query handler for inline buttons
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))

        # Error handler
        self.app.add_error_handler(self.error_handler)

    def _check_access(self, user_id: int) -> bool:
        """Check if user has access."""
        if not self.allowed_users:
            return True
        return user_id in self.allowed_users

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors."""
        logger.error(f"Error: {context.error}", exc_info=context.error)
        if update and update.effective_message:
            await update.effective_message.reply_text(
                f"Error: {str(context.error)}"
            )

    # ==================== Basic Commands ====================

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not self._check_access(update.effective_user.id):
            await update.message.reply_text("Access denied.")
            return

        welcome = """
Welcome to Strategy Bot!

I help you manage trading strategies, run backtests, and control live trading.

Quick commands:
/strategies - List available strategies
/backtest BTCUSDT velas 30 - Run backtest
/live start BTCUSDT velas - Start live trading

Use /help for full command list.
        """
        await update.message.reply_text(welcome.strip())

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not self._check_access(update.effective_user.id):
            return

        help_text = """
Strategy Bot Commands:

STRATEGIES
/strategies - List all available strategies
/strategy <name> - Get strategy details and parameters

ASSIGNMENTS
/assign <pair> <strategy> - Assign strategy to pair
/unassign <pair> <strategy> - Remove assignment
/list - Show all assignments

BACKTESTING
/backtest <pair> <strategy> [days] - Run backtest
/compare <pair> <s1,s2,...> - Compare strategies
/results [pair] - Show recent results

LIVE TRADING
/live start <pair> <strategy> - Start live session
/live stop <pair> <strategy> - Stop session
/live status - Show active sessions
/pnl - Show P&L summary

SYSTEM
/status - System status
/help - This message

Examples:
/backtest BTCUSDT velas 30
/compare BTCUSDT velas,ema_cross,bollinger
/live start ETHUSDT rsi_divergence
        """
        await update.message.reply_text(help_text.strip())

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not self._check_access(update.effective_user.id):
            return

        strategies = self.manager.list_strategies()
        assignments = self.manager.get_all_assignments()
        live_sessions = self.manager.get_live_sessions()

        status = f"""
System Status

Strategies: {len(strategies)} available
Assignments: {len(assignments)} configured
Live Sessions: {len(live_sessions)} active
Running Backtests: {len(self._running_tasks)}

Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """
        await update.message.reply_text(status.strip())

    # ==================== Strategy Commands ====================

    async def cmd_strategies(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /strategies command - list all strategies."""
        if not self._check_access(update.effective_user.id):
            return

        strategies = self.manager.list_strategies()

        if not strategies:
            await update.message.reply_text("No strategies available.")
            return

        text = "Available Strategies:\n\n"
        for s in strategies:
            text += f"• {s['name']}\n"
            text += f"  {s['description'][:60]}...\n\n"

        # Add inline buttons for details
        keyboard = [
            [InlineKeyboardButton(s['name'], callback_data=f"strategy_{s['name']}")]
            for s in strategies
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(text.strip(), reply_markup=reply_markup)

    async def cmd_strategy_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /strategy <name> command."""
        if not self._check_access(update.effective_user.id):
            return

        if not context.args:
            await update.message.reply_text("Usage: /strategy <name>")
            return

        name = context.args[0].lower()
        info = self.manager.get_strategy_info(name)

        if not info:
            await update.message.reply_text(f"Strategy '{name}' not found.")
            return

        text = f"""
Strategy: {info['name']}

{info['description']}

Default Parameters:
"""
        for param, value in info['default_params'].items():
            text += f"• {param}: {value}\n"

        await update.message.reply_text(text.strip())

    # ==================== Assignment Commands ====================

    async def cmd_assign(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /assign <pair> <strategy> command."""
        if not self._check_access(update.effective_user.id):
            return

        if len(context.args) < 2:
            await update.message.reply_text("Usage: /assign <pair> <strategy> [timeframe]")
            return

        pair = context.args[0].upper()
        strategy = context.args[1].lower()
        timeframe = context.args[2] if len(context.args) > 2 else "1h"

        success = self.manager.assign_strategy(
            pair=pair,
            strategy_name=strategy,
            timeframe=timeframe,
        )

        if success:
            await update.message.reply_text(f"Assigned {strategy} to {pair} ({timeframe})")
        else:
            await update.message.reply_text(f"Failed to assign. Strategy '{strategy}' not found.")

    async def cmd_unassign(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /unassign <pair> <strategy> command."""
        if not self._check_access(update.effective_user.id):
            return

        if len(context.args) < 2:
            await update.message.reply_text("Usage: /unassign <pair> <strategy>")
            return

        pair = context.args[0].upper()
        strategy = context.args[1].lower()

        success = self.manager.unassign_strategy(pair, strategy)

        if success:
            await update.message.reply_text(f"Removed {strategy} from {pair}")
        else:
            await update.message.reply_text("Assignment not found.")

    async def cmd_list_assignments(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /list command - show all assignments."""
        if not self._check_access(update.effective_user.id):
            return

        assignments = self.manager.get_all_assignments()

        if not assignments:
            await update.message.reply_text("No assignments configured.")
            return

        text = "Strategy Assignments:\n\n"
        for a in assignments:
            live_indicator = " [LIVE]" if a['is_live'] else ""
            text += f"• {a['pair']} - {a['strategy_name']} ({a['timeframe']}){live_indicator}\n"

        await update.message.reply_text(text.strip())

    # ==================== Backtest Commands ====================

    async def cmd_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /backtest <pair> <strategy> [days] command."""
        if not self._check_access(update.effective_user.id):
            return

        if len(context.args) < 2:
            await update.message.reply_text("Usage: /backtest <pair> <strategy> [days]")
            return

        pair = context.args[0].upper()
        strategy = context.args[1].lower()
        days = int(context.args[2]) if len(context.args) > 2 else 30

        await update.message.reply_text(f"Running backtest for {pair} with {strategy} ({days} days)...")

        # Run backtest in background
        task_id = f"{pair}_{strategy}_{datetime.utcnow().timestamp()}"

        async def run_backtest():
            try:
                result = self.manager.backtest(pair, strategy, days)
                return result
            except Exception as e:
                logger.error(f"Backtest error: {e}")
                return {"error": str(e)}

        # Execute and send result
        result = await run_backtest()

        if "error" in result:
            await update.message.reply_text(f"Backtest failed: {result['error']}")
            return

        # Format result
        text = f"""
Backtest Results: {pair} - {strategy}
Period: {days} days

Total Trades: {result.get('total_trades', 0)}
Win Rate: {result.get('win_rate', 0):.1f}%
Profit Factor: {result.get('profit_factor', 0):.2f}
Total Return: {result.get('total_return', 0):.2f}%
Max Drawdown: {result.get('max_drawdown', 0):.2f}%
Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}

Gross Profit: ${result.get('gross_profit', 0):.2f}
Gross Loss: ${result.get('gross_loss', 0):.2f}
        """
        await update.message.reply_text(text.strip())

    async def cmd_compare(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /compare <pair> <strategy1,strategy2,...> command."""
        if not self._check_access(update.effective_user.id):
            return

        if len(context.args) < 2:
            await update.message.reply_text("Usage: /compare <pair> <strategy1,strategy2,...>")
            return

        pair = context.args[0].upper()
        strategies = context.args[1].lower().split(',')
        days = int(context.args[2]) if len(context.args) > 2 else 30

        await update.message.reply_text(f"Comparing {len(strategies)} strategies on {pair}...")

        results = self.manager.compare_strategies(pair, strategies, days)

        text = f"Strategy Comparison: {pair} ({days} days)\n\n"
        text += f"{'Strategy':<15} {'Return':<10} {'WR':<8} {'PF':<8} {'DD':<8}\n"
        text += "-" * 50 + "\n"

        for r in results:
            if "error" not in r:
                text += f"{r['strategy_name']:<15} "
                text += f"{r.get('total_return', 0):>7.1f}% "
                text += f"{r.get('win_rate', 0):>6.1f}% "
                text += f"{r.get('profit_factor', 0):>6.2f} "
                text += f"{r.get('max_drawdown', 0):>6.1f}%\n"
            else:
                text += f"{r['strategy_name']:<15} ERROR\n"

        await update.message.reply_text(f"```\n{text}```", parse_mode='Markdown')

    async def cmd_results(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /results [pair] command."""
        if not self._check_access(update.effective_user.id):
            return

        pair = context.args[0].upper() if context.args else None
        results = self.manager.get_backtest_history(pair=pair, limit=10)

        if not results:
            await update.message.reply_text("No backtest results found.")
            return

        text = "Recent Backtest Results:\n\n"
        for r in results[:10]:
            text += f"• {r['pair']} - {r['strategy_name']}\n"
            text += f"  Return: {r.get('total_return', 0):.1f}%, "
            text += f"WR: {r.get('win_rate', 0):.1f}%, "
            text += f"Trades: {r.get('total_trades', 0)}\n"
            created = r.get('created_at', '')[:10] if r.get('created_at') else 'N/A'
            text += f"  Date: {created}\n\n"

        await update.message.reply_text(text.strip())

    # ==================== Live Trading Commands ====================

    async def cmd_live(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /live <action> <pair> <strategy> command."""
        if not self._check_access(update.effective_user.id):
            return

        if not context.args:
            await update.message.reply_text("Usage: /live <start|stop|status> [pair] [strategy]")
            return

        action = context.args[0].lower()

        if action == "status":
            sessions = self.manager.get_live_sessions()
            if not sessions:
                await update.message.reply_text("No active live sessions.")
                return

            text = "Active Live Sessions:\n\n"
            for s in sessions:
                paper_tag = " [PAPER]" if s['is_paper'] else " [REAL]"
                text += f"• {s['pair']} - {s['strategy_name']}{paper_tag}\n"
                text += f"  Status: {s['status']}\n"
                text += f"  Started: {s.get('started_at', 'N/A')}\n\n"

            await update.message.reply_text(text.strip())
            return

        if len(context.args) < 3:
            await update.message.reply_text("Usage: /live <start|stop> <pair> <strategy>")
            return

        pair = context.args[1].upper()
        strategy = context.args[2].lower()

        if action == "start":
            # Check if strategy is assigned
            assignments = self.manager.get_pair_strategies(pair)
            strategy_assigned = any(a['strategy_name'] == strategy for a in assignments)

            if not strategy_assigned:
                # Auto-assign
                self.manager.assign_strategy(pair, strategy, enable_live=True)

            success = self.manager.start_live(pair, strategy, is_paper=True)
            if success:
                await update.message.reply_text(f"Started live session for {pair} with {strategy} [PAPER]")
            else:
                await update.message.reply_text("Failed to start live session.")

        elif action == "stop":
            success = self.manager.stop_live(pair, strategy)
            if success:
                await update.message.reply_text(f"Stopped live session for {pair}")
            else:
                await update.message.reply_text("Session not found.")

        else:
            await update.message.reply_text("Unknown action. Use: start, stop, status")

    async def cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pnl command - show P&L summary."""
        if not self._check_access(update.effective_user.id):
            return

        sessions = self.manager.get_live_sessions()
        db_sessions = self.manager.storage.get_live_sessions()

        if not sessions and not db_sessions:
            await update.message.reply_text("No trading sessions found.")
            return

        text = "P&L Summary:\n\n"
        total_pnl = 0.0

        for s in db_sessions:
            realized = s.get('realized_pnl', 0)
            unrealized = s.get('unrealized_pnl', 0)
            total = realized + unrealized
            total_pnl += total

            text += f"• {s['pair']} - {s['strategy_name']}\n"
            text += f"  Realized: ${realized:.2f}\n"
            text += f"  Unrealized: ${unrealized:.2f}\n"
            text += f"  Total: ${total:.2f}\n\n"

        text += f"Total P&L: ${total_pnl:.2f}"

        await update.message.reply_text(text.strip())

    # ==================== Callback Handlers ====================

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()

        data = query.data

        if data.startswith("strategy_"):
            strategy_name = data.replace("strategy_", "")
            info = self.manager.get_strategy_info(strategy_name)

            if info:
                text = f"Strategy: {info['name']}\n\n{info['description']}\n\nParams:\n"
                for param, value in info['default_params'].items():
                    text += f"• {param}: {value}\n"

                await query.edit_message_text(text)

    # ==================== Bot Control ====================

    def run(self) -> None:
        """Start the bot (blocking)."""
        logger.info("Starting Strategy Bot...")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

    async def start_async(self) -> None:
        """Start the bot asynchronously."""
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

    async def stop_async(self) -> None:
        """Stop the bot asynchronously."""
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()


def create_bot(
    token: str,
    allowed_users: Optional[List[int]] = None,
) -> Optional[StrategyBot]:
    """
    Create Strategy Bot instance.

    Args:
        token: Telegram bot token
        allowed_users: Allowed user IDs

    Returns:
        StrategyBot instance or None if telegram not available
    """
    if not TELEGRAM_AVAILABLE:
        logger.warning("Telegram library not available")
        return None

    return StrategyBot(token, allowed_users)
