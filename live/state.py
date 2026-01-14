"""
VELAS Trading System - State Management

Handles system state persistence and recovery.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from data.storage import DataStorage


class StateManager:
    """
    Manages system state for persistence and recovery.

    Handles:
    - Position state
    - Signal history
    - Configuration snapshots
    - Recovery after restart
    """

    def __init__(
        self,
        storage: Optional[DataStorage] = None,
        state_file: str = "data_store/state.json",
    ) -> None:
        """
        Initialize state manager.

        Args:
            storage: DataStorage instance
            state_file: Path to state file
        """
        self.storage = storage
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self._state: Dict[str, Any] = {}
        self._checkpoints: List[Dict[str, Any]] = []
        self._max_checkpoints = 10

    def save(self, state: Optional[Dict[str, Any]] = None) -> None:
        """
        Save current state.

        Args:
            state: State to save (uses internal state if None)
        """
        if state:
            self._state = state

        self._state["saved_at"] = datetime.now().isoformat()

        # Save to file
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

        # Save to storage if available
        if self.storage:
            self.storage.save_state(self._state)

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load state from storage.

        Returns:
            Loaded state or None
        """
        # Try file first
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    self._state = json.load(f)
                    return self._state
            except json.JSONDecodeError:
                pass

        # Try storage
        if self.storage:
            state = self.storage.load_state()
            if state:
                self._state = state
                return state

        return None

    def checkpoint(self, name: str = "") -> None:
        """
        Create state checkpoint.

        Args:
            name: Checkpoint name/description
        """
        checkpoint = {
            "name": name or f"checkpoint_{len(self._checkpoints) + 1}",
            "timestamp": datetime.now().isoformat(),
            "state": self._state.copy(),
        }

        self._checkpoints.append(checkpoint)

        # Trim old checkpoints
        if len(self._checkpoints) > self._max_checkpoints:
            self._checkpoints = self._checkpoints[-self._max_checkpoints:]

    def restore_checkpoint(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """
        Restore state from checkpoint.

        Args:
            index: Checkpoint index (-1 for latest)

        Returns:
            Restored state or None
        """
        if not self._checkpoints:
            return None

        checkpoint = self._checkpoints[index]
        self._state = checkpoint["state"].copy()
        return self._state

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self._state.get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        """Set value in state."""
        self._state[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple state values."""
        self._state.update(updates)

    def clear(self) -> None:
        """Clear state."""
        self._state = {}
        if self.state_file.exists():
            self.state_file.unlink()

    # === Position State ===

    def save_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Save position state."""
        self._state["positions"] = positions
        self._state["positions_updated_at"] = datetime.now().isoformat()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get saved positions."""
        return self._state.get("positions", [])

    # === Signal State ===

    def save_active_signals(self, signals: List[Dict[str, Any]]) -> None:
        """Save active signals."""
        self._state["active_signals"] = signals

    def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get active signals."""
        return self._state.get("active_signals", [])

    def add_signal_history(self, signal: Dict[str, Any]) -> None:
        """Add signal to history."""
        if "signal_history" not in self._state:
            self._state["signal_history"] = []

        self._state["signal_history"].append({
            **signal,
            "recorded_at": datetime.now().isoformat(),
        })

        # Trim history
        max_history = 500
        if len(self._state["signal_history"]) > max_history:
            self._state["signal_history"] = self._state["signal_history"][-max_history:]

    # === Configuration State ===

    def save_config_snapshot(self, config: Dict[str, Any]) -> None:
        """Save configuration snapshot."""
        self._state["config_snapshot"] = {
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }

    def get_config_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get configuration snapshot."""
        snapshot = self._state.get("config_snapshot")
        return snapshot.get("config") if snapshot else None

    # === Runtime State ===

    def set_running(self, running: bool) -> None:
        """Set running state."""
        self._state["running"] = running
        self._state["running_since"] = datetime.now().isoformat() if running else None

    def is_running(self) -> bool:
        """Check if system was running."""
        return self._state.get("running", False)

    def set_last_processed(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
    ) -> None:
        """Set last processed candle timestamp."""
        if "last_processed" not in self._state:
            self._state["last_processed"] = {}

        key = f"{symbol}_{timeframe}"
        self._state["last_processed"][key] = timestamp.isoformat()

    def get_last_processed(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[datetime]:
        """Get last processed timestamp."""
        key = f"{symbol}_{timeframe}"
        timestamp_str = self._state.get("last_processed", {}).get(key)

        if timestamp_str:
            return datetime.fromisoformat(timestamp_str)
        return None

    # === Statistics State ===

    def update_statistics(self, stats: Dict[str, Any]) -> None:
        """Update trading statistics."""
        if "statistics" not in self._state:
            self._state["statistics"] = {}

        self._state["statistics"].update(stats)
        self._state["statistics"]["updated_at"] = datetime.now().isoformat()

    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics."""
        return self._state.get("statistics", {})

    def increment_counter(self, name: str, amount: int = 1) -> int:
        """Increment a counter."""
        if "counters" not in self._state:
            self._state["counters"] = {}

        current = self._state["counters"].get(name, 0)
        self._state["counters"][name] = current + amount
        return self._state["counters"][name]

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self._state.get("counters", {}).get(name, 0)

    # === Export/Import ===

    def export_state(self) -> str:
        """Export state as JSON string."""
        return json.dumps(self._state, indent=2, default=str)

    def import_state(self, state_json: str) -> None:
        """Import state from JSON string."""
        self._state = json.loads(state_json)

    def get_summary(self) -> Dict[str, Any]:
        """Get state summary."""
        return {
            "saved_at": self._state.get("saved_at"),
            "running": self._state.get("running", False),
            "positions_count": len(self._state.get("positions", [])),
            "active_signals_count": len(self._state.get("active_signals", [])),
            "checkpoints_count": len(self._checkpoints),
            "state_keys": list(self._state.keys()),
        }
