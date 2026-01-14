"""
VELAS Trading System - Core Module

Contains strategy logic, signal generation, filters, and portfolio management.
"""

from core.strategy import VelasStrategy
from core.signals import Signal, SignalGenerator
from core.filters import FilterManager
from core.portfolio import PortfolioManager

# Task queue (async operations)
try:
    from core.task_queue import TaskQueue, BacktestQueue, BatchBacktestQueue, TaskStatus
except ImportError:
    TaskQueue = None
    BacktestQueue = None
    BatchBacktestQueue = None
    TaskStatus = None

__all__ = [
    "VelasStrategy",
    "Signal",
    "SignalGenerator",
    "FilterManager",
    "PortfolioManager",
    "TaskQueue",
    "BacktestQueue",
    "BatchBacktestQueue",
    "TaskStatus",
]
