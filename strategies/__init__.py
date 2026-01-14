# Strategies module - Plugin-based strategy system
from .base import BaseStrategy, StrategyConfig, StrategyResult, SignalType, TPLevel
from .loader import StrategyLoader, StrategyRegistry, load_builtin_strategies

# Import strategies (lazy to avoid circular imports)
try:
    from .velas import VelasStrategy
    from .ema_cross import EMACrossStrategy
    from .rsi_divergence import RSIDivergenceStrategy
    from .bollinger_bands import BollingerBandsStrategy
except ImportError:
    VelasStrategy = None
    EMACrossStrategy = None
    RSIDivergenceStrategy = None
    BollingerBandsStrategy = None

# Import manager (lazy)
try:
    from .manager import StrategyManager
except ImportError:
    StrategyManager = None

__all__ = [
    # Base
    'BaseStrategy',
    'StrategyConfig',
    'StrategyResult',
    'SignalType',
    'TPLevel',

    # Loader
    'StrategyLoader',
    'StrategyRegistry',
    'load_builtin_strategies',

    # Strategies
    'VelasStrategy',
    'EMACrossStrategy',
    'RSIDivergenceStrategy',
    'BollingerBandsStrategy',

    # Manager
    'StrategyManager',
]
