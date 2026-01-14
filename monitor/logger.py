"""
VELAS Trading System - Logger

Comprehensive logging for trading system.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict

from loguru import logger


class VelasLogger:
    """
    Centralized logging for VELAS trading system.

    Features:
    - File rotation
    - Colored console output
    - Structured logging
    - Trade-specific logs
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        rotation: str = "10 MB",
        retention: str = "1 month",
        console_enabled: bool = True,
    ) -> None:
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files
            log_level: Minimum log level
            rotation: File rotation size/time
            retention: Log retention period
            console_enabled: Enable console output
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Remove default handler
        logger.remove()

        # Console handler
        if console_enabled:
            logger.add(
                sys.stdout,
                level=log_level,
                format=(
                    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                    "<level>{message}</level>"
                ),
                colorize=True,
            )

        # Main log file
        logger.add(
            self.log_dir / "velas.log",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )

        # Separate file for trades
        logger.add(
            self.log_dir / "trades.log",
            level="INFO",
            rotation=rotation,
            retention=retention,
            filter=lambda record: "trade" in record["extra"],
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        )

        # Separate file for signals
        logger.add(
            self.log_dir / "signals.log",
            level="INFO",
            rotation=rotation,
            retention=retention,
            filter=lambda record: "signal" in record["extra"],
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        )

        # Separate file for errors
        logger.add(
            self.log_dir / "errors.log",
            level="ERROR",
            rotation=rotation,
            retention=retention,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}\n{exception}",
        )

        self._logger = logger

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, **kwargs)

    # === Trade Logging ===

    def log_signal(
        self,
        symbol: str,
        side: str,
        entry: float,
        sl: float,
        tp_levels: list,
        score: float = 0,
    ) -> None:
        """Log trading signal."""
        message = (
            f"SIGNAL | {symbol} {side} | "
            f"Entry: {entry:.4f} | SL: {sl:.4f} | "
            f"TPs: {len(tp_levels)} | Score: {score:.1f}"
        )
        self._logger.bind(signal=True).info(message)

    def log_trade_open(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry: float,
        size_pct: float,
    ) -> None:
        """Log trade open."""
        message = (
            f"TRADE OPEN | ID: {trade_id} | {symbol} {side} | "
            f"Entry: {entry:.4f} | Size: {size_pct:.1f}%"
        )
        self._logger.bind(trade=True).info(message)

    def log_tp_hit(
        self,
        trade_id: str,
        symbol: str,
        tp_level: int,
        price: float,
        pnl_pct: float,
    ) -> None:
        """Log TP hit."""
        message = (
            f"TP HIT | ID: {trade_id} | {symbol} | "
            f"TP{tp_level}: {price:.4f} | P&L: {pnl_pct:+.2f}%"
        )
        self._logger.bind(trade=True).info(message)

    def log_sl_hit(
        self,
        trade_id: str,
        symbol: str,
        price: float,
        pnl_pct: float,
    ) -> None:
        """Log SL hit."""
        message = (
            f"SL HIT | ID: {trade_id} | {symbol} | "
            f"SL: {price:.4f} | P&L: {pnl_pct:+.2f}%"
        )
        self._logger.bind(trade=True).warning(message)

    def log_trade_close(
        self,
        trade_id: str,
        symbol: str,
        exit_type: str,
        pnl_pct: float,
        duration_bars: int,
    ) -> None:
        """Log trade close."""
        result = "WIN" if pnl_pct > 0 else "LOSS" if pnl_pct < 0 else "BE"
        message = (
            f"TRADE CLOSE | ID: {trade_id} | {symbol} | "
            f"{result} | {exit_type} | P&L: {pnl_pct:+.2f}% | "
            f"Duration: {duration_bars} bars"
        )
        self._logger.bind(trade=True).info(message)

    # === System Logging ===

    def log_system_start(self, config: Dict[str, Any]) -> None:
        """Log system start."""
        pairs = config.get("pairs", [])
        timeframes = config.get("timeframes", [])

        message = (
            f"SYSTEM START | Pairs: {len(pairs)} | "
            f"Timeframes: {timeframes} | "
            f"Mode: {'DRY RUN' if config.get('dry_run') else 'LIVE'}"
        )
        self._logger.info(message)

    def log_system_stop(self, reason: str = "Manual") -> None:
        """Log system stop."""
        self._logger.info(f"SYSTEM STOP | Reason: {reason}")

    def log_websocket_event(self, event: str, details: str = "") -> None:
        """Log WebSocket event."""
        self._logger.debug(f"WEBSOCKET | {event} | {details}")

    def log_api_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        duration_ms: float,
    ) -> None:
        """Log API request."""
        self._logger.debug(
            f"API | {method} {endpoint} | Status: {status} | {duration_ms:.0f}ms"
        )

    # === Performance Logging ===

    def log_daily_summary(
        self,
        date: datetime,
        signals: int,
        winners: int,
        losers: int,
        total_pnl: float,
    ) -> None:
        """Log daily summary."""
        win_rate = (winners / signals * 100) if signals > 0 else 0
        message = (
            f"DAILY SUMMARY | {date.strftime('%Y-%m-%d')} | "
            f"Signals: {signals} | W/L: {winners}/{losers} | "
            f"WR: {win_rate:.1f}% | P&L: {total_pnl:+.2f}%"
        )
        self._logger.info(message)


# Global logger instance
_velas_logger: Optional[VelasLogger] = None


def get_logger() -> VelasLogger:
    """Get or create global logger instance."""
    global _velas_logger
    if _velas_logger is None:
        _velas_logger = VelasLogger()
    return _velas_logger


def setup_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    **kwargs: Any,
) -> VelasLogger:
    """Set up global logger with custom settings."""
    global _velas_logger
    _velas_logger = VelasLogger(log_dir=log_dir, log_level=log_level, **kwargs)
    return _velas_logger
