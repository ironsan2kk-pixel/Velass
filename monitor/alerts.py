"""
VELAS Trading System - Alert Manager

Monitors system health and triggers alerts.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from enum import Enum

from data.storage import DataStorage


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Alert types."""
    LOSS_STREAK = "LOSS_STREAK"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    SIGNAL_FREQUENCY = "SIGNAL_FREQUENCY"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    CONNECTION_LOST = "CONNECTION_LOST"
    POSITION_HEAT = "POSITION_HEAT"
    DRAWDOWN = "DRAWDOWN"
    CUSTOM = "CUSTOM"


@dataclass
class Alert:
    """Alert record."""
    id: str = ""
    type: AlertType = AlertType.CUSTOM
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


class AlertManager:
    """
    Manages system alerts and anomaly detection.

    Features:
    - Loss streak detection
    - Volatility spike alerts
    - Signal frequency monitoring
    - System health checks
    """

    def __init__(
        self,
        storage: Optional[DataStorage] = None,
        on_alert: Optional[Callable[[Alert], None]] = None,
    ) -> None:
        """
        Initialize alert manager.

        Args:
            storage: Storage for persistence
            on_alert: Callback for new alerts
        """
        self.storage = storage
        self.on_alert = on_alert

        # Alert history
        self.alerts: List[Alert] = []
        self._max_alerts = 500

        # Counters for anomaly detection
        self._consecutive_losses = 0
        self._recent_signals: List[datetime] = []
        self._last_volatility: Dict[str, float] = {}

        # Configuration
        self.loss_streak_threshold = 3
        self.volatility_spike_multiplier = 2.5
        self.max_signals_per_period = 5
        self.signal_period_hours = 4
        self.max_portfolio_heat_warning = 12.0
        self.max_drawdown_warning = 10.0

        self._alert_counter = 0

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure alert thresholds."""
        alerts_config = config.get("alerts", {})

        self.loss_streak_threshold = alerts_config.get("loss_streak_threshold", 3)
        self.volatility_spike_multiplier = alerts_config.get("volatility_spike_multiplier", 2.5)
        self.max_signals_per_period = alerts_config.get("max_signals_per_period", 5)
        self.signal_period_hours = alerts_config.get("signal_period_hours", 4)

    def _create_alert(
        self,
        alert_type: AlertType,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        details: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create and emit alert."""
        self._alert_counter += 1

        alert = Alert(
            id=f"alert_{self._alert_counter}",
            type=alert_type,
            severity=severity,
            message=message,
            details=details or {},
        )

        self.alerts.append(alert)

        # Trim history
        if len(self.alerts) > self._max_alerts:
            self.alerts = self.alerts[-self._max_alerts:]

        # Save to storage
        if self.storage:
            self.storage.save_alert(
                alert_type=alert.type.value,
                message=alert.message,
                severity=alert.severity.value,
                data=alert.details,
            )

        # Trigger callback
        if self.on_alert:
            self.on_alert(alert)

        return alert

    # === Trade Monitoring ===

    def on_trade_result(self, is_win: bool, pnl_pct: float) -> Optional[Alert]:
        """
        Process trade result for loss streak detection.

        Args:
            is_win: Whether trade was profitable
            pnl_pct: Trade P&L percentage

        Returns:
            Alert if loss streak detected
        """
        if is_win:
            self._consecutive_losses = 0
            return None

        self._consecutive_losses += 1

        if self._consecutive_losses >= self.loss_streak_threshold:
            return self._create_alert(
                alert_type=AlertType.LOSS_STREAK,
                message=f"{self._consecutive_losses} consecutive losses",
                severity=AlertSeverity.WARNING,
                details={
                    "consecutive_losses": self._consecutive_losses,
                    "last_loss_pct": pnl_pct,
                },
            )

        return None

    # === Signal Frequency Monitoring ===

    def on_signal_generated(self) -> Optional[Alert]:
        """
        Track signal frequency for anomaly detection.

        Returns:
            Alert if too many signals
        """
        now = datetime.now()
        self._recent_signals.append(now)

        # Clean old signals
        cutoff = now - timedelta(hours=self.signal_period_hours)
        self._recent_signals = [s for s in self._recent_signals if s > cutoff]

        if len(self._recent_signals) > self.max_signals_per_period:
            return self._create_alert(
                alert_type=AlertType.SIGNAL_FREQUENCY,
                message=f"{len(self._recent_signals)} signals in {self.signal_period_hours} hours",
                severity=AlertSeverity.WARNING,
                details={
                    "signal_count": len(self._recent_signals),
                    "period_hours": self.signal_period_hours,
                    "normal_max": self.max_signals_per_period,
                },
            )

        return None

    # === Volatility Monitoring ===

    def check_volatility(
        self,
        symbol: str,
        current_atr: float,
        avg_atr: float,
    ) -> Optional[Alert]:
        """
        Check for volatility spikes.

        Args:
            symbol: Trading pair
            current_atr: Current ATR value
            avg_atr: Average ATR value

        Returns:
            Alert if volatility spike detected
        """
        if avg_atr == 0:
            return None

        ratio = current_atr / avg_atr

        if ratio >= self.volatility_spike_multiplier:
            # Check if already alerted recently
            last = self._last_volatility.get(symbol, 0)
            if ratio <= last * 1.1:  # Don't re-alert for same level
                return None

            self._last_volatility[symbol] = ratio

            return self._create_alert(
                alert_type=AlertType.HIGH_VOLATILITY,
                message=f"{symbol} volatility spike: {ratio:.1f}x normal",
                severity=AlertSeverity.WARNING,
                details={
                    "symbol": symbol,
                    "current_atr": current_atr,
                    "avg_atr": avg_atr,
                    "ratio": ratio,
                },
            )

        return None

    # === Portfolio Monitoring ===

    def check_portfolio_heat(self, heat_pct: float) -> Optional[Alert]:
        """
        Check portfolio heat level.

        Args:
            heat_pct: Current portfolio heat percentage

        Returns:
            Alert if heat is too high
        """
        if heat_pct >= self.max_portfolio_heat_warning:
            return self._create_alert(
                alert_type=AlertType.POSITION_HEAT,
                message=f"Portfolio heat at {heat_pct:.1f}%",
                severity=AlertSeverity.WARNING,
                details={
                    "heat_pct": heat_pct,
                    "threshold": self.max_portfolio_heat_warning,
                },
            )

        return None

    def check_drawdown(
        self,
        current_dd: float,
        max_dd: float,
    ) -> Optional[Alert]:
        """
        Check drawdown level.

        Args:
            current_dd: Current drawdown percentage
            max_dd: Maximum drawdown percentage

        Returns:
            Alert if drawdown is concerning
        """
        if current_dd >= self.max_drawdown_warning:
            severity = (
                AlertSeverity.CRITICAL
                if current_dd >= max_dd * 0.8
                else AlertSeverity.WARNING
            )

            return self._create_alert(
                alert_type=AlertType.DRAWDOWN,
                message=f"Drawdown at {current_dd:.1f}%",
                severity=severity,
                details={
                    "current_drawdown": current_dd,
                    "max_drawdown": max_dd,
                },
            )

        return None

    # === System Monitoring ===

    def on_system_error(
        self,
        error_type: str,
        message: str,
        critical: bool = False,
    ) -> Alert:
        """
        Record system error.

        Args:
            error_type: Type of error
            message: Error message
            critical: Whether error is critical

        Returns:
            Created alert
        """
        return self._create_alert(
            alert_type=AlertType.SYSTEM_ERROR,
            message=f"{error_type}: {message}",
            severity=AlertSeverity.CRITICAL if critical else AlertSeverity.WARNING,
            details={
                "error_type": error_type,
            },
        )

    def on_connection_lost(self, service: str) -> Alert:
        """Record connection loss."""
        return self._create_alert(
            alert_type=AlertType.CONNECTION_LOST,
            message=f"Connection lost: {service}",
            severity=AlertSeverity.CRITICAL,
            details={"service": service},
        )

    # === Custom Alerts ===

    def create_custom_alert(
        self,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create custom alert."""
        return self._create_alert(
            alert_type=AlertType.CUSTOM,
            message=message,
            severity=severity,
            details=details,
        )

    # === Alert Management ===

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def acknowledge_all(self) -> int:
        """Acknowledge all alerts."""
        count = 0
        for alert in self.alerts:
            if not alert.acknowledged:
                alert.acknowledged = True
                count += 1
        return count

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get unacknowledged alerts."""
        alerts = [a for a in self.alerts if not a.acknowledged]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def get_recent_alerts(
        self,
        hours: int = 24,
        limit: int = 50,
    ) -> List[Alert]:
        """Get recent alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [a for a in self.alerts if a.timestamp > cutoff]
        return recent[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        active = self.get_active_alerts()

        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active),
            "critical": len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
            "warnings": len([a for a in active if a.severity == AlertSeverity.WARNING]),
            "consecutive_losses": self._consecutive_losses,
            "recent_signals": len(self._recent_signals),
        }
