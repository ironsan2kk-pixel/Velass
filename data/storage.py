"""
VELAS Trading System - Data Storage

Handles data persistence with SQLite/PostgreSQL and file storage.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


Base = declarative_base()


class SignalRecord(Base):
    """Database model for signals."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(50), unique=True, index=True)
    symbol = Column(String(20), index=True)
    timeframe = Column(String(10))
    side = Column(String(10))
    entry_price = Column(Float)
    sl_price = Column(Float)
    tp_levels = Column(Text)  # JSON string
    status = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    score = Column(Float, default=0.0)
    filters_passed = Column(Text)  # JSON string
    indicators = Column(Text)  # JSON string


class TradeRecord(Base):
    """Database model for trades."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, index=True)
    signal_id = Column(String(50), index=True)
    symbol = Column(String(20), index=True)
    timeframe = Column(String(10))
    side = Column(String(10))
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    sl_price = Column(Float)
    exit_type = Column(String(20), nullable=True)
    pnl_pct = Column(Float, default=0.0)
    pnl_r = Column(Float, default=0.0)
    tps_hit = Column(Text)  # JSON string
    entry_time = Column(DateTime)
    exit_time = Column(DateTime, nullable=True)
    status = Column(String(20))


class PositionRecord(Base):
    """Database model for positions."""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(String(50), unique=True, index=True)
    signal_id = Column(String(50), index=True)
    symbol = Column(String(20), index=True)
    timeframe = Column(String(10))
    side = Column(String(10))
    entry_price = Column(Float)
    current_price = Column(Float)
    sl_price = Column(Float)
    tp_levels = Column(Text)  # JSON string
    initial_size_pct = Column(Float)
    remaining_size_pct = Column(Float)
    realized_pnl_pct = Column(Float, default=0.0)
    unrealized_pnl_pct = Column(Float, default=0.0)
    status = Column(String(20))
    group_name = Column(String(50), nullable=True)
    opened_at = Column(DateTime)
    closed_at = Column(DateTime, nullable=True)


class AlertRecord(Base):
    """Database model for alerts."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), index=True)
    severity = Column(String(20))
    message = Column(Text)
    data = Column(Text)  # JSON string
    acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DataStorage:
    """
    Unified data storage handler.

    Supports:
    - SQLite for signals, trades, positions
    - Parquet files for OHLCV data
    - JSON for configuration and state
    """

    def __init__(
        self,
        db_url: str = "sqlite:///data_store/velas.db",
        data_dir: str = "data_store",
    ) -> None:
        """
        Initialize storage.

        Args:
            db_url: SQLAlchemy database URL
            data_dir: Directory for file storage
        """
        self.db_url = db_url
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "results").mkdir(exist_ok=True)

        # Initialize database
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()

    # === Signal Operations ===

    def save_signal(self, signal_data: Dict[str, Any]) -> None:
        """Save signal to database."""
        session = self.get_session()
        try:
            record = SignalRecord(
                signal_id=signal_data.get("id", ""),
                symbol=signal_data.get("symbol", ""),
                timeframe=signal_data.get("timeframe", ""),
                side=signal_data.get("side", ""),
                entry_price=signal_data.get("entry_price", 0.0),
                sl_price=signal_data.get("sl_price", 0.0),
                tp_levels=json.dumps(signal_data.get("tp_levels", [])),
                status=signal_data.get("status", "PENDING"),
                score=signal_data.get("score", 0.0),
                filters_passed=json.dumps(signal_data.get("filters_passed", [])),
                indicators=json.dumps(signal_data.get("indicators", {})),
            )
            session.add(record)
            session.commit()
        finally:
            session.close()

    def update_signal_status(
        self,
        signal_id: str,
        status: str,
        closed_at: Optional[datetime] = None,
    ) -> None:
        """Update signal status."""
        session = self.get_session()
        try:
            record = session.query(SignalRecord).filter_by(signal_id=signal_id).first()
            if record:
                record.status = status
                if closed_at:
                    record.closed_at = closed_at
                session.commit()
        finally:
            session.close()

    def get_signals(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get signals from database."""
        session = self.get_session()
        try:
            query = session.query(SignalRecord)
            if symbol:
                query = query.filter_by(symbol=symbol)
            if status:
                query = query.filter_by(status=status)
            query = query.order_by(SignalRecord.created_at.desc()).limit(limit)

            return [
                {
                    "id": r.signal_id,
                    "symbol": r.symbol,
                    "timeframe": r.timeframe,
                    "side": r.side,
                    "entry_price": r.entry_price,
                    "sl_price": r.sl_price,
                    "tp_levels": json.loads(r.tp_levels) if r.tp_levels else [],
                    "status": r.status,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "score": r.score,
                }
                for r in query.all()
            ]
        finally:
            session.close()

    # === Trade Operations ===

    def save_trade(self, trade_data: Dict[str, Any]) -> None:
        """Save trade to database."""
        session = self.get_session()
        try:
            record = TradeRecord(
                trade_id=trade_data.get("id", ""),
                signal_id=trade_data.get("signal_id", ""),
                symbol=trade_data.get("symbol", ""),
                timeframe=trade_data.get("timeframe", ""),
                side=trade_data.get("side", ""),
                entry_price=trade_data.get("entry_price", 0.0),
                exit_price=trade_data.get("exit_price"),
                sl_price=trade_data.get("sl_price", 0.0),
                exit_type=trade_data.get("exit_type"),
                pnl_pct=trade_data.get("pnl_pct", 0.0),
                pnl_r=trade_data.get("pnl_r", 0.0),
                tps_hit=json.dumps(trade_data.get("tps_hit", [])),
                entry_time=trade_data.get("entry_time"),
                exit_time=trade_data.get("exit_time"),
                status=trade_data.get("status", "OPEN"),
            )
            session.add(record)
            session.commit()
        finally:
            session.close()

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get trades from database."""
        session = self.get_session()
        try:
            query = session.query(TradeRecord)
            if symbol:
                query = query.filter_by(symbol=symbol)
            if start_date:
                query = query.filter(TradeRecord.entry_time >= start_date)
            if end_date:
                query = query.filter(TradeRecord.entry_time <= end_date)
            query = query.order_by(TradeRecord.entry_time.desc()).limit(limit)

            return [
                {
                    "id": r.trade_id,
                    "signal_id": r.signal_id,
                    "symbol": r.symbol,
                    "side": r.side,
                    "entry_price": r.entry_price,
                    "exit_price": r.exit_price,
                    "pnl_pct": r.pnl_pct,
                    "exit_type": r.exit_type,
                    "entry_time": r.entry_time.isoformat() if r.entry_time else None,
                    "exit_time": r.exit_time.isoformat() if r.exit_time else None,
                }
                for r in query.all()
            ]
        finally:
            session.close()

    # === Position Operations ===

    def save_position(self, position_data: Dict[str, Any]) -> None:
        """Save or update position."""
        session = self.get_session()
        try:
            existing = session.query(PositionRecord).filter_by(
                position_id=position_data.get("id", "")
            ).first()

            if existing:
                # Update
                existing.current_price = position_data.get("current_price", existing.current_price)
                existing.remaining_size_pct = position_data.get("remaining_size_pct", existing.remaining_size_pct)
                existing.realized_pnl_pct = position_data.get("realized_pnl_pct", existing.realized_pnl_pct)
                existing.unrealized_pnl_pct = position_data.get("unrealized_pnl_pct", existing.unrealized_pnl_pct)
                existing.status = position_data.get("status", existing.status)
                existing.sl_price = position_data.get("sl_price", existing.sl_price)
            else:
                # Insert
                record = PositionRecord(
                    position_id=position_data.get("id", ""),
                    signal_id=position_data.get("signal_id", ""),
                    symbol=position_data.get("symbol", ""),
                    timeframe=position_data.get("timeframe", ""),
                    side=position_data.get("side", ""),
                    entry_price=position_data.get("entry_price", 0.0),
                    current_price=position_data.get("current_price", 0.0),
                    sl_price=position_data.get("sl_price", 0.0),
                    tp_levels=json.dumps(position_data.get("tp_levels", [])),
                    initial_size_pct=position_data.get("initial_size_pct", 0.0),
                    remaining_size_pct=position_data.get("remaining_size_pct", 0.0),
                    status=position_data.get("status", "OPEN"),
                    group_name=position_data.get("group", ""),
                    opened_at=position_data.get("opened_at", datetime.utcnow()),
                )
                session.add(record)

            session.commit()
        finally:
            session.close()

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        session = self.get_session()
        try:
            records = session.query(PositionRecord).filter(
                PositionRecord.status.in_(["OPEN", "PARTIAL"])
            ).all()

            return [
                {
                    "id": r.position_id,
                    "signal_id": r.signal_id,
                    "symbol": r.symbol,
                    "side": r.side,
                    "entry_price": r.entry_price,
                    "current_price": r.current_price,
                    "sl_price": r.sl_price,
                    "remaining_size_pct": r.remaining_size_pct,
                    "unrealized_pnl_pct": r.unrealized_pnl_pct,
                    "status": r.status,
                }
                for r in records
            ]
        finally:
            session.close()

    # === Alert Operations ===

    def save_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "INFO",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save alert to database."""
        session = self.get_session()
        try:
            record = AlertRecord(
                alert_type=alert_type,
                severity=severity,
                message=message,
                data=json.dumps(data or {}),
            )
            session.add(record)
            session.commit()
        finally:
            session.close()

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        session = self.get_session()
        try:
            records = session.query(AlertRecord).order_by(
                AlertRecord.created_at.desc()
            ).limit(limit).all()

            return [
                {
                    "id": r.id,
                    "type": r.alert_type,
                    "severity": r.severity,
                    "message": r.message,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "acknowledged": r.acknowledged,
                }
                for r in records
            ]
        finally:
            session.close()

    # === OHLCV File Operations ===

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        subdir: str = "raw",
    ) -> str:
        """
        Save OHLCV data to parquet file.

        Returns:
            Path to saved file
        """
        filepath = self.data_dir / subdir / f"{symbol}_{timeframe}.parquet"
        df.to_parquet(filepath)
        return str(filepath)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        subdir: str = "raw",
    ) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data from parquet file.

        Returns:
            DataFrame or None if not found
        """
        filepath = self.data_dir / subdir / f"{symbol}_{timeframe}.parquet"
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None

    def load_all_ohlcv(
        self,
        symbols: List[str],
        timeframes: List[str],
        subdir: str = "raw",
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load OHLCV data for multiple symbols and timeframes.

        Returns:
            {symbol: {timeframe: DataFrame}}
        """
        data = {}

        for symbol in symbols:
            data[symbol] = {}
            for tf in timeframes:
                df = self.load_ohlcv(symbol, tf, subdir)
                if df is not None:
                    data[symbol][tf] = df

        return data

    # === State Management ===

    def save_state(self, state: Dict[str, Any], filename: str = "state.json") -> None:
        """Save system state to JSON file."""
        filepath = self.data_dir / filename
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, filename: str = "state.json") -> Optional[Dict[str, Any]]:
        """Load system state from JSON file."""
        filepath = self.data_dir / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
        return None
