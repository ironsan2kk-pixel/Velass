"""
VELAS Trading System - Monitor Module

Contains logging, alerts, and web dashboard.
"""

from monitor.logger import VelasLogger
from monitor.alerts import AlertManager
from monitor.dashboard import Dashboard

__all__ = [
    "VelasLogger",
    "AlertManager",
    "Dashboard",
]
