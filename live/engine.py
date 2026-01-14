"""
VELAS Trading System - Live Trading Engine

Main engine for live signal generation and position management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
import pandas as pd

from core.strategy import VelasStrategy, VelasParams
from core.signals import Signal, SignalSide, SignalStatus, SignalGenerator, calculate_signal_score
from core.filters import FilterManager
from core.portfolio import PortfolioManager, Position

from data.binance_client import BinanceClient
from data.websocket import BinanceWebSocket, Candle, KlineAggregator
from data.storage import DataStorage

from tg_bot.bot import TelegramBot, TelegramNotifier
from tg_bot.cornix import CornixFormatter

from live.position_tracker import PositionTracker
from live.state import StateManager


class LiveEngine:
    """
    Main live trading engine.

    Coordinates:
    - Real-time data streaming
    - Signal generation
    - Position management
    - Telegram notifications
    - State persistence
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dry_run: bool = False,
    ) -> None:
        """
        Initialize live engine.

        Args:
            config: Configuration dictionary
            dry_run: If True, don't send actual signals
        """
        self.config = config
        self.dry_run = dry_run
        self._running = False

        # Trading parameters
        self.pairs = config.get("pairs", [])
        self.timeframes = config.get("timeframes", ["30m", "1h", "2h"])

        # Initialize strategy
        strategy_config = config.get("strategy", {})
        velas_config = config.get("velas_params", {})

        self.strategy = VelasStrategy(
            params=VelasParams(**velas_config),
            tp_percents=strategy_config.get("tp_levels", [1.0, 2.0, 3.0, 4.0, 7.5, 14.0]),
            tp_distribution=strategy_config.get("tp_distribution", [17, 17, 17, 17, 16, 16]),
            sl_percent=strategy_config.get("sl_percent", 8.5),
        )

        self.signal_generator = SignalGenerator(strategy=self.strategy)

        # Initialize filters
        filter_config = config.get("filters", {})
        self.filter_manager = FilterManager(filter_config)

        # Initialize portfolio manager
        portfolio_config = config.get("portfolio", {})
        self.portfolio_manager = PortfolioManager(
            max_positions=portfolio_config.get("max_positions", 5),
            max_per_group=portfolio_config.get("max_per_group", 2),
            position_size_pct=portfolio_config.get("position_size_pct", 2.0),
            max_portfolio_heat=portfolio_config.get("max_portfolio_heat", 15.0),
            groups=portfolio_config.get("groups", {}),
        )

        # Initialize components (set up later)
        self.binance_client: Optional[BinanceClient] = None
        self.websocket: Optional[BinanceWebSocket] = None
        self.telegram_bot: Optional[TelegramBot] = None
        self.notifier: Optional[TelegramNotifier] = None
        self.storage: Optional[DataStorage] = None
        self.state_manager: Optional[StateManager] = None
        self.position_tracker: Optional[PositionTracker] = None

        # Data aggregator
        self.aggregator = KlineAggregator(max_candles=500)

        # Historical data cache
        self.historical_data: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Cornix formatter
        exchange_config = config.get("exchange", {})
        self.cornix = CornixFormatter(
            leverage=exchange_config.get("leverage", 10),
            margin_type=exchange_config.get("margin_type", "Cross").title(),
        )

        # Callbacks
        self._on_signal_callbacks: List[Callable[[Signal], None]] = []
        self._on_position_update_callbacks: List[Callable[[Position], None]] = []

    async def initialize(self) -> None:
        """Initialize all components."""
        print("Initializing Live Engine...")

        # Initialize Binance client
        secrets = self.config.get("secrets", {})
        self.binance_client = BinanceClient(
            api_key=secrets.get("binance_api_key"),
            api_secret=secrets.get("binance_api_secret"),
        )

        # Initialize WebSocket
        self.websocket = BinanceWebSocket(
            on_candle=self._on_candle,
            on_error=self._on_ws_error,
        )

        # Initialize Telegram
        telegram_config = self.config.get("telegram", {})
        if telegram_config.get("enabled", True) and not self.dry_run:
            self.telegram_bot = TelegramBot(
                token=secrets.get("telegram_token", ""),
                channel_id=secrets.get("telegram_channel_id", ""),
                alert_channel_id=secrets.get("telegram_alert_channel_id"),
            )

            self.notifier = TelegramNotifier(
                bot=self.telegram_bot,
                send_signals=telegram_config.get("send_signals", True),
                send_tp_notifications=telegram_config.get("send_tp_notifications", True),
                send_sl_notifications=telegram_config.get("send_sl_notifications", True),
                send_alerts=telegram_config.get("send_anomaly_alerts", True),
            )

        # Initialize storage
        db_config = self.config.get("database", {})
        self.storage = DataStorage(
            db_url=f"sqlite:///{db_config.get('path', 'data_store/velas.db')}",
        )

        # Initialize state manager
        self.state_manager = StateManager(storage=self.storage)

        # Initialize position tracker
        self.position_tracker = PositionTracker(
            portfolio_manager=self.portfolio_manager,
            storage=self.storage,
        )

        # Load historical data
        await self._load_historical_data()

        # Restore state
        await self._restore_state()

        print("Live Engine initialized")

    async def _load_historical_data(self) -> None:
        """Load historical data for indicators."""
        print("Loading historical data...")

        for pair in self.pairs:
            self.historical_data[pair] = {}

            for tf in self.timeframes:
                # Try to load from storage first
                df = self.storage.load_ohlcv(pair, tf)

                if df is None or len(df) < 200:
                    # Download from Binance
                    print(f"  Downloading {pair} {tf}...")
                    df = await self.binance_client.get_historical_klines(
                        symbol=pair,
                        interval=tf,
                        start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                    )

                    if df is not None and len(df) > 0:
                        self.storage.save_ohlcv(df, pair, tf)

                if df is not None:
                    self.historical_data[pair][tf] = df

        print(f"Loaded data for {len(self.historical_data)} pairs")

    async def _restore_state(self) -> None:
        """Restore state from storage."""
        state = self.state_manager.load()

        if state:
            print("Restoring previous state...")

            # Restore open positions
            positions = state.get("positions", [])
            for pos_data in positions:
                self.portfolio_manager.positions[pos_data["id"]] = Position(
                    id=pos_data["id"],
                    signal_id=pos_data.get("signal_id", ""),
                    symbol=pos_data["symbol"],
                    side=SignalSide[pos_data["side"]],
                    entry_price=pos_data["entry_price"],
                    sl_price=pos_data["sl_price"],
                    tp_levels=pos_data.get("tp_levels", []),
                    remaining_size_pct=pos_data.get("remaining_size_pct", 100),
                    group=pos_data.get("group", ""),
                )

            print(f"Restored {len(positions)} positions")

    def _on_candle(self, candle: Candle) -> None:
        """Handle incoming candle from WebSocket."""
        self.aggregator.add_candle(candle)

        # Process closed candles
        if candle.is_closed:
            asyncio.create_task(self._process_candle(candle))

    def _on_ws_error(self, error: Exception) -> None:
        """Handle WebSocket error."""
        print(f"WebSocket error: {error}")

        if self.notifier:
            self.notifier.queue_alert(
                alert_type="WEBSOCKET_ERROR",
                message=str(error),
                severity="WARNING",
            )

    async def _process_candle(self, candle: Candle) -> None:
        """Process closed candle for signals."""
        symbol = candle.symbol
        interval = candle.interval

        # Update historical data
        if symbol in self.historical_data and interval in self.historical_data[symbol]:
            new_row = pd.DataFrame([{
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }], index=[candle.open_time])

            self.historical_data[symbol][interval] = pd.concat([
                self.historical_data[symbol][interval],
                new_row
            ]).tail(500)

        # Generate signal
        df = self.historical_data.get(symbol, {}).get(interval)
        if df is None or len(df) < 100:
            return

        signal = self.signal_generator.generate_signal(df, symbol, interval)

        if signal:
            await self._process_signal(signal, df)

        # Update position status
        await self._update_positions(symbol, candle)

    async def _process_signal(
        self,
        signal: Signal,
        df: pd.DataFrame,
    ) -> None:
        """Process and potentially send a signal."""
        # Apply filters
        passed, results = self.filter_manager.evaluate_signal(
            signal=signal,
            df=df,
        )

        if not passed:
            print(f"Signal {signal.symbol} filtered out")
            return

        # Calculate score
        calculate_signal_score(signal)

        # Check portfolio constraints
        can_open, reason = self.portfolio_manager.can_open_position(signal)

        if not can_open:
            print(f"Signal {signal.symbol} rejected: {reason}")
            return

        # Save signal
        self.storage.save_signal(signal.to_dict())

        # Send to Telegram
        if self.notifier and not self.dry_run:
            message = self.cornix.format_signal(signal)
            self.notifier.queue_signal(message)

        # Open position in portfolio
        position = self.portfolio_manager.open_position(signal)

        if position:
            self.storage.save_position(position.to_dict())
            print(f"Opened position: {signal.symbol} {signal.side.value}")

        # Trigger callbacks
        for callback in self._on_signal_callbacks:
            callback(signal)

    async def _update_positions(
        self,
        symbol: str,
        candle: Candle,
    ) -> None:
        """Update position status based on price."""
        positions = self.portfolio_manager.get_positions_by_symbol(symbol)

        for position in positions:
            # Check TP/SL hits
            for i, tp in enumerate(position.tp_levels):
                if i in position.tps_hit:
                    continue

                tp_hit = False
                if position.side == SignalSide.LONG and candle.high >= tp:
                    tp_hit = True
                elif position.side == SignalSide.SHORT and candle.low <= tp:
                    tp_hit = True

                if tp_hit:
                    # Partial close
                    tp_dist = self.strategy.tp_distribution[i] if i < len(self.strategy.tp_distribution) else 16
                    self.portfolio_manager.partial_close(
                        position.id, tp, tp_dist, i
                    )

                    # Notify
                    if self.notifier:
                        self.notifier.queue_tp_notification(
                            symbol=symbol,
                            tp_level=i + 1,
                            entry_price=position.entry_price,
                            tp_price=tp,
                            pnl_pct=((tp - position.entry_price) / position.entry_price) * 100,
                            remaining_pct=position.remaining_size_pct,
                        )

                    # Move SL
                    config = self.config.get("strategy", {})
                    if config.get("stop_management") == "cascade":
                        if i == 0:
                            new_sl = position.entry_price
                        else:
                            new_sl = position.tp_levels[i - 1]
                        self.portfolio_manager.update_stop_loss(position.id, new_sl)

            # Check SL
            sl_hit = False
            if position.side == SignalSide.LONG and candle.low <= position.sl_price:
                sl_hit = True
            elif position.side == SignalSide.SHORT and candle.high >= position.sl_price:
                sl_hit = True

            if sl_hit:
                self.portfolio_manager.close_position(
                    position.id, position.sl_price, "SL"
                )

                if self.notifier:
                    pnl = ((position.sl_price - position.entry_price) / position.entry_price) * 100
                    if position.side == SignalSide.SHORT:
                        pnl = -pnl

                    self.notifier.queue_sl_notification(
                        symbol=symbol,
                        entry_price=position.entry_price,
                        sl_price=position.sl_price,
                        pnl_pct=pnl,
                    )

            # Update position in storage
            self.storage.save_position(position.to_dict())

    async def run(self) -> None:
        """Main run loop."""
        await self.initialize()

        print("Starting Live Engine...")
        self._running = True

        # Start notifier if available
        if self.notifier:
            await self.notifier.start()

        # Subscribe to WebSocket streams
        await self.websocket.subscribe_klines(
            symbols=self.pairs,
            intervals=self.timeframes,
        )

        try:
            # Start WebSocket
            ws_task = asyncio.create_task(self.websocket.run())

            # Start state save loop
            state_task = asyncio.create_task(self._state_save_loop())

            # Start monitoring loop
            monitor_task = asyncio.create_task(self._monitoring_loop())

            # Wait for shutdown
            await asyncio.gather(ws_task, state_task, monitor_task)

        except asyncio.CancelledError:
            print("Shutting down...")

        finally:
            await self.shutdown()

    async def _state_save_loop(self) -> None:
        """Periodically save state."""
        while self._running:
            await asyncio.sleep(60)  # Save every minute

            state = {
                "positions": [
                    p.to_dict() for p in self.portfolio_manager.get_open_positions()
                ],
                "last_update": datetime.now().isoformat(),
            }
            self.state_manager.save(state)

    async def _monitoring_loop(self) -> None:
        """Monitor system health."""
        while self._running:
            await asyncio.sleep(300)  # Check every 5 minutes

            # Check for anomalies
            positions = self.portfolio_manager.get_open_positions()
            heat = self.portfolio_manager.get_portfolio_heat()

            if heat > self.portfolio_manager.max_portfolio_heat * 0.9:
                if self.notifier:
                    self.notifier.queue_alert(
                        alert_type="HIGH_PORTFOLIO_HEAT",
                        message=f"Portfolio heat at {heat:.1f}%",
                        severity="WARNING",
                    )

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False

        # Save final state
        state = {
            "positions": [
                p.to_dict() for p in self.portfolio_manager.get_open_positions()
            ],
            "last_update": datetime.now().isoformat(),
        }
        self.state_manager.save(state)

        # Close connections
        if self.websocket:
            await self.websocket.disconnect()

        if self.binance_client:
            await self.binance_client.close()

        if self.notifier:
            await self.notifier.stop()

        print("Live Engine shutdown complete")

    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        """Register signal callback."""
        self._on_signal_callbacks.append(callback)

    def on_position_update(self, callback: Callable[[Position], None]) -> None:
        """Register position update callback."""
        self._on_position_update_callbacks.append(callback)
