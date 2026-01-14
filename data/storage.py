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


class StrategyRecord(Base):
    """Database model for registered strategies."""
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, index=True)
    description = Column(Text)
    default_params = Column(Text)  # JSON string
    params_schema = Column(Text)   # JSON string
    is_builtin = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PairConfigRecord(Base):
    """Database model for pair-strategy configurations."""
    __tablename__ = "pair_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(20), index=True)
    strategy_name = Column(String(50), index=True)
    timeframe = Column(String(10), default="1h")
    params = Column(Text)  # JSON string - override params
    is_live = Column(Boolean, default=False)  # Live trading enabled
    is_backtest = Column(Boolean, default=True)  # Backtest enabled
    risk_per_trade = Column(Float, default=0.02)
    max_positions = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BacktestResultRecord(Base):
    """Database model for backtest results."""
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(String(50), unique=True, index=True)
    pair = Column(String(20), index=True)
    strategy_name = Column(String(50), index=True)
    timeframe = Column(String(10))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    params = Column(Text)  # JSON string

    # Metrics
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    total_return = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)

    # Full metrics JSON
    full_metrics = Column(Text)  # JSON string

    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="completed")


class LiveSessionRecord(Base):
    """Database model for live trading sessions."""
    __tablename__ = "live_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), unique=True, index=True)
    pair = Column(String(20), index=True)
    strategy_name = Column(String(50), index=True)
    timeframe = Column(String(10))
    params = Column(Text)  # JSON string

    # Session state
    status = Column(String(20), default="stopped")  # running, stopped, paused
    started_at = Column(DateTime, nullable=True)
    stopped_at = Column(DateTime, nullable=True)

    # Performance
    total_trades = Column(Integer, default=0)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)

    # Config
    is_paper = Column(Boolean, default=True)  # Paper or real trading
    initial_balance = Column(Float, default=0.0)
    current_balance = Column(Float, default=0.0)


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

    # === Strategy Operations ===

    def save_strategy(self, strategy_info: Dict[str, Any]) -> None:
        """Save or update strategy in database."""
        session = self.get_session()
        try:
            existing = session.query(StrategyRecord).filter_by(
                name=strategy_info.get("name", "")
            ).first()

            if existing:
                existing.description = strategy_info.get("description", existing.description)
                existing.default_params = json.dumps(strategy_info.get("default_params", {}))
                existing.params_schema = json.dumps(strategy_info.get("params_schema", {}))
                existing.is_active = strategy_info.get("is_active", existing.is_active)
            else:
                record = StrategyRecord(
                    name=strategy_info.get("name", ""),
                    description=strategy_info.get("description", ""),
                    default_params=json.dumps(strategy_info.get("default_params", {})),
                    params_schema=json.dumps(strategy_info.get("params_schema", {})),
                    is_builtin=strategy_info.get("is_builtin", True),
                    is_active=strategy_info.get("is_active", True),
                )
                session.add(record)

            session.commit()
        finally:
            session.close()

    def get_strategies(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all strategies from database."""
        session = self.get_session()
        try:
            query = session.query(StrategyRecord)
            if active_only:
                query = query.filter_by(is_active=True)

            return [
                {
                    "name": r.name,
                    "description": r.description,
                    "default_params": json.loads(r.default_params) if r.default_params else {},
                    "params_schema": json.loads(r.params_schema) if r.params_schema else {},
                    "is_builtin": r.is_builtin,
                    "is_active": r.is_active,
                }
                for r in query.all()
            ]
        finally:
            session.close()

    def get_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """Get single strategy by name."""
        session = self.get_session()
        try:
            r = session.query(StrategyRecord).filter_by(name=name).first()
            if r:
                return {
                    "name": r.name,
                    "description": r.description,
                    "default_params": json.loads(r.default_params) if r.default_params else {},
                    "params_schema": json.loads(r.params_schema) if r.params_schema else {},
                    "is_builtin": r.is_builtin,
                    "is_active": r.is_active,
                }
            return None
        finally:
            session.close()

    # === Pair Config Operations ===

    def save_pair_config(self, config: Dict[str, Any]) -> None:
        """Save or update pair-strategy configuration."""
        session = self.get_session()
        try:
            existing = session.query(PairConfigRecord).filter_by(
                pair=config.get("pair", ""),
                strategy_name=config.get("strategy_name", "")
            ).first()

            if existing:
                existing.timeframe = config.get("timeframe", existing.timeframe)
                existing.params = json.dumps(config.get("params", {}))
                existing.is_live = config.get("is_live", existing.is_live)
                existing.is_backtest = config.get("is_backtest", existing.is_backtest)
                existing.risk_per_trade = config.get("risk_per_trade", existing.risk_per_trade)
                existing.max_positions = config.get("max_positions", existing.max_positions)
            else:
                record = PairConfigRecord(
                    pair=config.get("pair", ""),
                    strategy_name=config.get("strategy_name", ""),
                    timeframe=config.get("timeframe", "1h"),
                    params=json.dumps(config.get("params", {})),
                    is_live=config.get("is_live", False),
                    is_backtest=config.get("is_backtest", True),
                    risk_per_trade=config.get("risk_per_trade", 0.02),
                    max_positions=config.get("max_positions", 1),
                )
                session.add(record)

            session.commit()
        finally:
            session.close()

    def get_pair_configs(
        self,
        pair: Optional[str] = None,
        strategy_name: Optional[str] = None,
        live_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get pair configurations."""
        session = self.get_session()
        try:
            query = session.query(PairConfigRecord)
            if pair:
                query = query.filter_by(pair=pair)
            if strategy_name:
                query = query.filter_by(strategy_name=strategy_name)
            if live_only:
                query = query.filter_by(is_live=True)

            return [
                {
                    "pair": r.pair,
                    "strategy_name": r.strategy_name,
                    "timeframe": r.timeframe,
                    "params": json.loads(r.params) if r.params else {},
                    "is_live": r.is_live,
                    "is_backtest": r.is_backtest,
                    "risk_per_trade": r.risk_per_trade,
                    "max_positions": r.max_positions,
                }
                for r in query.all()
            ]
        finally:
            session.close()

    def delete_pair_config(self, pair: str, strategy_name: str) -> bool:
        """Delete a pair configuration."""
        session = self.get_session()
        try:
            record = session.query(PairConfigRecord).filter_by(
                pair=pair, strategy_name=strategy_name
            ).first()
            if record:
                session.delete(record)
                session.commit()
                return True
            return False
        finally:
            session.close()

    # === Backtest Result Operations ===

    def save_backtest_result(self, result: Dict[str, Any]) -> None:
        """Save backtest result to database."""
        session = self.get_session()
        try:
            record = BacktestResultRecord(
                backtest_id=result.get("backtest_id", ""),
                pair=result.get("pair", ""),
                strategy_name=result.get("strategy_name", ""),
                timeframe=result.get("timeframe", ""),
                start_date=result.get("start_date"),
                end_date=result.get("end_date"),
                params=json.dumps(result.get("params", {})),
                total_trades=result.get("total_trades", 0),
                win_rate=result.get("win_rate", 0.0),
                profit_factor=result.get("profit_factor", 0.0),
                total_return=result.get("total_return", 0.0),
                max_drawdown=result.get("max_drawdown", 0.0),
                sharpe_ratio=result.get("sharpe_ratio", 0.0),
                sortino_ratio=result.get("sortino_ratio", 0.0),
                full_metrics=json.dumps(result.get("full_metrics", {})),
                status=result.get("status", "completed"),
            )
            session.add(record)
            session.commit()
        finally:
            session.close()

    def get_backtest_results(
        self,
        pair: Optional[str] = None,
        strategy_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get backtest results."""
        session = self.get_session()
        try:
            query = session.query(BacktestResultRecord)
            if pair:
                query = query.filter_by(pair=pair)
            if strategy_name:
                query = query.filter_by(strategy_name=strategy_name)
            query = query.order_by(BacktestResultRecord.created_at.desc()).limit(limit)

            return [
                {
                    "backtest_id": r.backtest_id,
                    "pair": r.pair,
                    "strategy_name": r.strategy_name,
                    "timeframe": r.timeframe,
                    "start_date": r.start_date.isoformat() if r.start_date else None,
                    "end_date": r.end_date.isoformat() if r.end_date else None,
                    "params": json.loads(r.params) if r.params else {},
                    "total_trades": r.total_trades,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "total_return": r.total_return,
                    "max_drawdown": r.max_drawdown,
                    "sharpe_ratio": r.sharpe_ratio,
                    "sortino_ratio": r.sortino_ratio,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in query.all()
            ]
        finally:
            session.close()

    # === Live Session Operations ===

    def save_live_session(self, session_data: Dict[str, Any]) -> None:
        """Save or update live trading session."""
        session = self.get_session()
        try:
            existing = session.query(LiveSessionRecord).filter_by(
                session_id=session_data.get("session_id", "")
            ).first()

            if existing:
                existing.status = session_data.get("status", existing.status)
                existing.stopped_at = session_data.get("stopped_at", existing.stopped_at)
                existing.total_trades = session_data.get("total_trades", existing.total_trades)
                existing.realized_pnl = session_data.get("realized_pnl", existing.realized_pnl)
                existing.unrealized_pnl = session_data.get("unrealized_pnl", existing.unrealized_pnl)
                existing.current_balance = session_data.get("current_balance", existing.current_balance)
            else:
                record = LiveSessionRecord(
                    session_id=session_data.get("session_id", ""),
                    pair=session_data.get("pair", ""),
                    strategy_name=session_data.get("strategy_name", ""),
                    timeframe=session_data.get("timeframe", "1h"),
                    params=json.dumps(session_data.get("params", {})),
                    status=session_data.get("status", "stopped"),
                    started_at=session_data.get("started_at"),
                    is_paper=session_data.get("is_paper", True),
                    initial_balance=session_data.get("initial_balance", 0.0),
                    current_balance=session_data.get("current_balance", 0.0),
                )
                session.add(record)

            session.commit()
        finally:
            session.close()

    def get_live_sessions(
        self,
        status: Optional[str] = None,
        pair: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get live trading sessions."""
        session = self.get_session()
        try:
            query = session.query(LiveSessionRecord)
            if status:
                query = query.filter_by(status=status)
            if pair:
                query = query.filter_by(pair=pair)

            return [
                {
                    "session_id": r.session_id,
                    "pair": r.pair,
                    "strategy_name": r.strategy_name,
                    "timeframe": r.timeframe,
                    "params": json.loads(r.params) if r.params else {},
                    "status": r.status,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "stopped_at": r.stopped_at.isoformat() if r.stopped_at else None,
                    "total_trades": r.total_trades,
                    "realized_pnl": r.realized_pnl,
                    "unrealized_pnl": r.unrealized_pnl,
                    "is_paper": r.is_paper,
                    "current_balance": r.current_balance,
                }
                for r in query.all()
            ]
        finally:
            session.close()
