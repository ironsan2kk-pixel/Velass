"""
Base Strategy Class - Plugin Interface for Trading Strategies

All strategies must inherit from BaseStrategy and implement:
- name: Strategy identifier
- description: What the strategy does
- default_params: Default parameter values
- params_schema: JSON schema for parameter validation
- calculate_indicators(): Add indicators to dataframe
- generate_signals(): Generate BUY/SELL signals
- calculate_tp_sl(): Calculate take profit and stop loss levels
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategyConfig:
    """Configuration for a strategy instance"""
    strategy_name: str
    pair: str
    timeframe: str = "1h"
    params: Dict[str, Any] = field(default_factory=dict)

    # Risk management
    risk_per_trade: float = 0.02  # 2% per trade
    max_positions: int = 3
    use_trailing_stop: bool = True

    # TP levels (percentages of full position to close)
    tp_percentages: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.20, 0.15, 0.10, 0.05])

    def __post_init__(self):
        if not self.params:
            self.params = {}


@dataclass
class TPLevel:
    """Take Profit Level"""
    level: int  # 1-6
    price: float
    percentage: float  # Percentage of position to close
    hit: bool = False
    hit_time: Optional[datetime] = None


@dataclass
class StrategyResult:
    """Result of strategy signal generation"""
    signal: SignalType
    confidence: float  # 0.0 - 1.0
    entry_price: float
    stop_loss: float
    take_profits: List[TPLevel]

    # Metadata
    pair: str
    timeframe: str
    timestamp: datetime
    strategy_name: str

    # Additional info
    indicators: Dict[str, float] = field(default_factory=dict)
    filters_passed: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate R:R to first TP"""
        if not self.take_profits or self.signal == SignalType.HOLD:
            return 0.0

        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profits[0].price - self.entry_price)
        return reward / risk if risk > 0 else 0.0

    @property
    def is_valid(self) -> bool:
        """Check if signal is actionable"""
        return (
            self.signal != SignalType.HOLD and
            self.confidence >= 0.5 and
            self.risk_reward_ratio >= 1.5 and
            len(self.take_profits) > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'signal': self.signal.value,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profits': [
                {'level': tp.level, 'price': tp.price, 'percentage': tp.percentage}
                for tp in self.take_profits
            ],
            'pair': self.pair,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat(),
            'strategy_name': self.strategy_name,
            'indicators': self.indicators,
            'risk_reward_ratio': self.risk_reward_ratio,
            'notes': self.notes
        }


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses must implement:
    - name (property): Strategy identifier
    - description (property): Strategy description
    - default_params (property): Default parameters
    - params_schema (property): JSON schema for validation
    - calculate_indicators(): Add indicators to DataFrame
    - generate_signals(): Generate trading signals
    - calculate_tp_sl(): Calculate TP/SL levels
    """

    def __init__(self, config: StrategyConfig):
        """Initialize strategy with configuration"""
        self.config = config
        self.params = {**self.default_params, **config.params}
        self._validate_params()

    # ==================== Abstract Properties ====================

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable strategy description"""
        pass

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Default parameter values"""
        pass

    @property
    @abstractmethod
    def params_schema(self) -> Dict[str, Any]:
        """JSON schema for parameter validation"""
        pass

    # ==================== Abstract Methods ====================

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators needed for the strategy.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with added indicator columns
        """
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on indicators.

        Args:
            df: DataFrame with indicators (from calculate_indicators)

        Returns:
            DataFrame with 'signal' column (1=BUY, -1=SELL, 0=HOLD)
        """
        pass

    @abstractmethod
    def calculate_tp_sl(
        self,
        df: pd.DataFrame,
        signal: SignalType,
        entry_price: float
    ) -> Tuple[float, List[TPLevel]]:
        """
        Calculate stop loss and take profit levels.

        Args:
            df: DataFrame with indicators
            signal: BUY or SELL signal
            entry_price: Entry price for the trade

        Returns:
            Tuple of (stop_loss_price, list of TPLevel objects)
        """
        pass

    # ==================== Optional Override Methods ====================

    def calculate_confidence(self, df: pd.DataFrame, idx: int) -> float:
        """
        Calculate signal confidence score (0.0 - 1.0).
        Override this method for custom confidence calculation.

        Args:
            df: DataFrame with indicators and signals
            idx: Index of the signal row

        Returns:
            Confidence score between 0.0 and 1.0
        """
        return 0.7  # Default confidence

    def apply_filters(self, df: pd.DataFrame, idx: int) -> Dict[str, bool]:
        """
        Apply additional filters to validate signal.
        Override this method for custom filters.

        Args:
            df: DataFrame with indicators
            idx: Index of the signal row

        Returns:
            Dictionary of filter name -> passed (True/False)
        """
        return {'default': True}

    def get_position_size(self, balance: float, entry: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management.

        Args:
            balance: Account balance
            entry: Entry price
            stop_loss: Stop loss price

        Returns:
            Position size in base currency
        """
        risk_amount = balance * self.config.risk_per_trade
        risk_per_unit = abs(entry - stop_loss)

        if risk_per_unit == 0:
            return 0.0

        return risk_amount / risk_per_unit

    # ==================== Core Methods ====================

    def analyze(self, df: pd.DataFrame) -> Optional[StrategyResult]:
        """
        Main analysis method - runs full strategy pipeline.

        Args:
            df: OHLCV DataFrame

        Returns:
            StrategyResult if signal found, None otherwise
        """
        # Step 1: Calculate indicators
        df = self.calculate_indicators(df.copy())

        # Step 2: Generate signals
        df = self.generate_signals(df)

        # Step 3: Check last row for signal
        last_idx = len(df) - 1
        last_row = df.iloc[last_idx]

        signal_value = last_row.get('signal', 0)
        if signal_value == 0:
            return None

        signal_type = SignalType.BUY if signal_value > 0 else SignalType.SELL
        entry_price = float(last_row['close'])

        # Step 4: Apply filters
        filters_passed = self.apply_filters(df, last_idx)
        if not all(filters_passed.values()):
            return None

        # Step 5: Calculate TP/SL
        stop_loss, take_profits = self.calculate_tp_sl(df, signal_type, entry_price)

        # Step 6: Calculate confidence
        confidence = self.calculate_confidence(df, last_idx)

        # Step 7: Extract indicator values for result
        indicators = self._extract_indicators(df, last_idx)

        # Step 8: Build result
        timestamp = last_row.name if isinstance(last_row.name, datetime) else datetime.now()

        return StrategyResult(
            signal=signal_type,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profits=take_profits,
            pair=self.config.pair,
            timeframe=self.config.timeframe,
            timestamp=timestamp,
            strategy_name=self.name,
            indicators=indicators,
            filters_passed=filters_passed
        )

    def backtest_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all signals for backtesting.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with signals and indicators
        """
        df = self.calculate_indicators(df.copy())
        df = self.generate_signals(df)
        return df

    # ==================== Helper Methods ====================

    def _validate_params(self) -> None:
        """Validate parameters against schema"""
        schema = self.params_schema
        for param_name, param_info in schema.get('properties', {}).items():
            if param_name in self.params:
                value = self.params[param_name]
                param_type = param_info.get('type')

                # Type validation
                if param_type == 'integer' and not isinstance(value, int):
                    raise ValueError(f"Parameter {param_name} must be integer")
                elif param_type == 'number' and not isinstance(value, (int, float)):
                    raise ValueError(f"Parameter {param_name} must be number")

                # Range validation
                if 'minimum' in param_info and value < param_info['minimum']:
                    raise ValueError(f"Parameter {param_name} must be >= {param_info['minimum']}")
                if 'maximum' in param_info and value > param_info['maximum']:
                    raise ValueError(f"Parameter {param_name} must be <= {param_info['maximum']}")

    def _extract_indicators(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Extract indicator values from DataFrame row"""
        indicators = {}
        exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'signal'}

        for col in df.columns:
            if col not in exclude_cols:
                value = df.iloc[idx][col]
                if isinstance(value, (int, float)) and not pd.isna(value):
                    indicators[col] = float(value)

        return indicators

    # ==================== Utility Methods ====================

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1

        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', pair='{self.config.pair}')"
