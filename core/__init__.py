"""
VELAS Trading System - Core Module

Contains strategy logic, signal generation, filters, and portfolio management.
"""

from core.strategy import VelasStrategy
from core.signals import Signal, SignalGenerator
from core.filters import FilterManager
from core.portfolio import PortfolioManager

__all__ = [
    "VelasStrategy",
    "Signal",
    "SignalGenerator",
    "FilterManager",
    "PortfolioManager",
]
