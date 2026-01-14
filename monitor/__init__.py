"""
VELAS Trading System - Monitor Module

Contains logging, alerts, and web dashboard.
"""

from monitor.logger import VelasLogger
from monitor.alerts import AlertManager

# Dashboard requires FastAPI/uvicorn
# Import directly: from monitor.dashboard import Dashboard

__all__ = [
    "VelasLogger",
    "AlertManager",
]
