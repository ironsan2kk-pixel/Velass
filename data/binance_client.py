"""
VELAS Trading System - Binance REST API Client

Handles historical data fetching and account operations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import aiohttp
from pathlib import Path


class BinanceClient:
    """
    Binance Futures REST API client.

    Handles:
    - Historical klines/candles
    - Account information
    - Exchange information
    """

    BASE_URL = "https://fapi.binance.com"
    KLINES_LIMIT = 1500  # Max candles per request

    TIMEFRAME_MAP = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        """
        Initialize Binance client.

        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Make HTTP request to Binance API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response
        """
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"

        try:
            async with session.request(method, url, params=params) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Binance API error: {response.status} - {text}")
                return await response.json()
        except aiohttp.ClientError as e:
            raise Exception(f"Request failed: {e}")

    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information."""
        return await self._request("GET", "/fapi/v1/exchangeInfo")

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500,
    ) -> List[List[Any]]:
        """
        Get kline/candlestick data.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 1h, etc.)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of candles (max 1500)

        Returns:
            List of kline data
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, self.KLINES_LIMIT),
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return await self._request("GET", "/fapi/v1/klines", params)

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get historical klines with pagination.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (optional, defaults to now)

        Returns:
            DataFrame with OHLCV data
        """
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)

        if end_date:
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        all_klines = []
        current_start = start_ts

        while current_start < end_ts:
            klines = await self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_ts,
                limit=self.KLINES_LIMIT,
            )

            if not klines:
                break

            all_klines.extend(klines)

            # Move to next batch
            last_time = klines[-1][0]
            current_start = last_time + 1

            # Rate limiting
            await asyncio.sleep(0.1)

        return self._klines_to_dataframe(all_klines)

    def _klines_to_dataframe(self, klines: List[List[Any]]) -> pd.DataFrame:
        """Convert klines to DataFrame."""
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)

        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]]

        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]

        return df

    async def get_ticker_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        result = await self._request(
            "GET", "/fapi/v1/ticker/price",
            params={"symbol": symbol}
        )
        return float(result["price"])

    async def get_all_ticker_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        result = await self._request("GET", "/fapi/v1/ticker/price")
        return {item["symbol"]: float(item["price"]) for item in result}

    async def get_24hr_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker statistics."""
        return await self._request(
            "GET", "/fapi/v1/ticker/24hr",
            params={"symbol": symbol}
        )

    async def download_all_pairs(
        self,
        pairs: List[str],
        timeframes: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        output_dir: str = "data_store/raw",
    ) -> Dict[str, Dict[str, str]]:
        """
        Download historical data for multiple pairs and timeframes.

        Args:
            pairs: List of trading pairs
            timeframes: List of timeframes
            start_date: Start date
            end_date: End date
            output_dir: Output directory for parquet files

        Returns:
            Dictionary of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for pair in pairs:
            saved_files[pair] = {}

            for tf in timeframes:
                print(f"Downloading {pair} {tf}...")

                try:
                    df = await self.get_historical_klines(
                        symbol=pair,
                        interval=tf,
                        start_date=start_date,
                        end_date=end_date,
                    )

                    if len(df) > 0:
                        filename = f"{pair}_{tf}.parquet"
                        filepath = output_path / filename
                        df.to_parquet(filepath)
                        saved_files[pair][tf] = str(filepath)
                        print(f"  Saved {len(df)} candles to {filename}")
                    else:
                        print(f"  No data for {pair} {tf}")

                except Exception as e:
                    print(f"  Error downloading {pair} {tf}: {e}")

                # Rate limiting between requests
                await asyncio.sleep(0.2)

        return saved_files


async def download_data(
    pairs: List[str],
    timeframes: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    output_dir: str = "data_store/raw",
) -> None:
    """
    Utility function to download historical data.

    Usage:
        asyncio.run(download_data(
            pairs=["BTCUSDT", "ETHUSDT"],
            timeframes=["30m", "1h", "2h"],
            start_date="2023-01-01",
        ))
    """
    client = BinanceClient()
    try:
        await client.download_all_pairs(
            pairs=pairs,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
        )
    finally:
        await client.close()
