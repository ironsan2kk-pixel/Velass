"""
VELAS Trading System - Binance WebSocket Client

Handles real-time data streaming from Binance.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Set
from dataclasses import dataclass, field
import websockets
from websockets.client import WebSocketClientProtocol


@dataclass
class Candle:
    """Real-time candle data."""
    symbol: str
    interval: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "open_time": self.open_time.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "is_closed": self.is_closed,
        }


class BinanceWebSocket:
    """
    Binance Futures WebSocket client for real-time data.

    Handles:
    - Multiple kline streams
    - Automatic reconnection
    - Callback-based updates
    """

    WS_URL = "wss://fstream.binance.com/ws"
    STREAM_URL = "wss://fstream.binance.com/stream"

    def __init__(
        self,
        on_candle: Optional[Callable[[Candle], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """
        Initialize WebSocket client.

        Args:
            on_candle: Callback for candle updates
            on_error: Callback for errors
        """
        self.on_candle = on_candle
        self.on_error = on_error

        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._subscriptions: Set[str] = set()
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

        # Latest candles cache
        self.candles: Dict[str, Dict[str, Candle]] = {}  # {symbol: {interval: Candle}}

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        try:
            self._ws = await websockets.connect(
                self.STREAM_URL,
                ping_interval=20,
                ping_timeout=10,
            )
            self._running = True
            self._reconnect_delay = 1.0
            print("WebSocket connected")
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        print("WebSocket disconnected")

    async def subscribe_klines(
        self,
        symbols: List[str],
        intervals: List[str],
    ) -> None:
        """
        Subscribe to kline streams.

        Args:
            symbols: List of trading pairs
            intervals: List of intervals
        """
        streams = []

        for symbol in symbols:
            for interval in intervals:
                stream_name = f"{symbol.lower()}@kline_{interval}"
                streams.append(stream_name)
                self._subscriptions.add(stream_name)

        if not self._ws:
            await self.connect()

        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }

        await self._ws.send(json.dumps(subscribe_msg))
        print(f"Subscribed to {len(streams)} streams")

    async def unsubscribe_klines(
        self,
        symbols: List[str],
        intervals: List[str],
    ) -> None:
        """Unsubscribe from kline streams."""
        streams = []

        for symbol in symbols:
            for interval in intervals:
                stream_name = f"{symbol.lower()}@kline_{interval}"
                streams.append(stream_name)
                self._subscriptions.discard(stream_name)

        if self._ws:
            unsubscribe_msg = {
                "method": "UNSUBSCRIBE",
                "params": streams,
                "id": 2
            }
            await self._ws.send(json.dumps(unsubscribe_msg))

    async def _handle_message(self, message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Handle stream data
            if "stream" in data and "data" in data:
                stream_data = data["data"]
                event_type = stream_data.get("e")

                if event_type == "kline":
                    candle = self._parse_kline(stream_data)
                    self._update_candle_cache(candle)

                    if self.on_candle:
                        self.on_candle(candle)

        except json.JSONDecodeError as e:
            if self.on_error:
                self.on_error(e)
        except Exception as e:
            if self.on_error:
                self.on_error(e)

    def _parse_kline(self, data: Dict[str, Any]) -> Candle:
        """Parse kline data from WebSocket message."""
        k = data["k"]

        return Candle(
            symbol=data["s"],
            interval=k["i"],
            open_time=datetime.fromtimestamp(k["t"] / 1000),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            is_closed=k["x"],
        )

    def _update_candle_cache(self, candle: Candle) -> None:
        """Update internal candle cache."""
        if candle.symbol not in self.candles:
            self.candles[candle.symbol] = {}

        self.candles[candle.symbol][candle.interval] = candle

    def get_latest_candle(
        self,
        symbol: str,
        interval: str,
    ) -> Optional[Candle]:
        """Get latest candle from cache."""
        return self.candles.get(symbol, {}).get(interval)

    async def run(self) -> None:
        """Main WebSocket event loop with auto-reconnection."""
        while self._running:
            try:
                if not self._ws:
                    await self.connect()

                    # Resubscribe after reconnection
                    if self._subscriptions:
                        subscribe_msg = {
                            "method": "SUBSCRIBE",
                            "params": list(self._subscriptions),
                            "id": 1
                        }
                        await self._ws.send(json.dumps(subscribe_msg))

                async for message in self._ws:
                    if not self._running:
                        break
                    await self._handle_message(message)

            except websockets.ConnectionClosed:
                print(f"WebSocket connection closed, reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )
                self._ws = None

            except Exception as e:
                if self.on_error:
                    self.on_error(e)
                print(f"WebSocket error: {e}, reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )
                self._ws = None

    async def run_with_callback(
        self,
        symbols: List[str],
        intervals: List[str],
        callback: Callable[[Candle], None],
    ) -> None:
        """
        Convenience method to run with specific subscriptions.

        Args:
            symbols: List of trading pairs
            intervals: List of intervals
            callback: Function to call on each candle update
        """
        self.on_candle = callback
        await self.subscribe_klines(symbols, intervals)
        await self.run()


class KlineAggregator:
    """
    Aggregates WebSocket klines into DataFrame format.
    """

    def __init__(self, max_candles: int = 1000) -> None:
        """
        Initialize aggregator.

        Args:
            max_candles: Maximum candles to keep in memory per stream
        """
        self.max_candles = max_candles
        self.data: Dict[str, Dict[str, List[Candle]]] = {}  # {symbol: {interval: [candles]}}

    def add_candle(self, candle: Candle) -> None:
        """Add candle to aggregator."""
        if candle.symbol not in self.data:
            self.data[candle.symbol] = {}

        if candle.interval not in self.data[candle.symbol]:
            self.data[candle.symbol][candle.interval] = []

        candles = self.data[candle.symbol][candle.interval]

        # Only add closed candles
        if candle.is_closed:
            # Check for duplicates
            if not candles or candles[-1].open_time != candle.open_time:
                candles.append(candle)

                # Trim to max size
                if len(candles) > self.max_candles:
                    self.data[candle.symbol][candle.interval] = candles[-self.max_candles:]

    def get_dataframe(
        self,
        symbol: str,
        interval: str,
    ) -> Optional[Any]:
        """
        Get DataFrame for symbol/interval.

        Returns:
            pandas DataFrame or None
        """
        import pandas as pd

        candles = self.data.get(symbol, {}).get(interval, [])

        if not candles:
            return None

        df = pd.DataFrame([
            {
                "timestamp": c.open_time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ])

        df.set_index("timestamp", inplace=True)
        return df

    def get_latest_n(
        self,
        symbol: str,
        interval: str,
        n: int = 100,
    ) -> List[Candle]:
        """Get latest N candles."""
        candles = self.data.get(symbol, {}).get(interval, [])
        return candles[-n:] if candles else []
