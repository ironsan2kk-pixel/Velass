"""
Strategy Loader - Dynamic loading and registry of trading strategies

Provides:
- StrategyRegistry: Singleton registry of all available strategies
- StrategyLoader: Dynamic loading of strategy classes
- Auto-discovery of strategies in the strategies/ directory
"""

import importlib
import importlib.util
import inspect
import os
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import logging

from .base import BaseStrategy, StrategyConfig

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Singleton registry for all available trading strategies.

    Usage:
        registry = StrategyRegistry()
        registry.register(MyStrategy)
        strategy_class = registry.get('my_strategy')
        strategy = strategy_class(config)
    """

    _instance = None
    _strategies: Dict[str, Type[BaseStrategy]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._strategies = {}
        return cls._instance

    def register(self, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy class.

        Args:
            strategy_class: Class that inherits from BaseStrategy
        """
        if not inspect.isclass(strategy_class):
            raise ValueError(f"Expected class, got {type(strategy_class)}")

        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"{strategy_class} must inherit from BaseStrategy")

        # Create temporary instance to get name
        # We need a dummy config for this
        try:
            dummy_config = StrategyConfig(
                strategy_name="dummy",
                pair="BTCUSDT"
            )
            temp_instance = strategy_class(dummy_config)
            name = temp_instance.name
        except Exception as e:
            # If instantiation fails, use class name
            name = strategy_class.__name__.lower().replace('strategy', '')
            logger.warning(f"Could not instantiate {strategy_class.__name__} for name, using: {name}")

        self._strategies[name] = strategy_class
        logger.info(f"Registered strategy: {name}")

    def unregister(self, name: str) -> None:
        """Remove a strategy from registry"""
        if name in self._strategies:
            del self._strategies[name]
            logger.info(f"Unregistered strategy: {name}")

    def get(self, name: str) -> Optional[Type[BaseStrategy]]:
        """
        Get a strategy class by name.

        Args:
            name: Strategy identifier

        Returns:
            Strategy class or None if not found
        """
        return self._strategies.get(name)

    def create(self, config: StrategyConfig) -> BaseStrategy:
        """
        Create a strategy instance from config.

        Args:
            config: Strategy configuration

        Returns:
            Instantiated strategy

        Raises:
            ValueError: If strategy not found
        """
        strategy_class = self.get(config.strategy_name)
        if strategy_class is None:
            raise ValueError(f"Strategy '{config.strategy_name}' not found. "
                           f"Available: {list(self._strategies.keys())}")
        return strategy_class(config)

    def list_strategies(self) -> List[str]:
        """Get list of all registered strategy names"""
        return list(self._strategies.keys())

    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a strategy.

        Args:
            name: Strategy identifier

        Returns:
            Dictionary with strategy info or None
        """
        strategy_class = self.get(name)
        if strategy_class is None:
            return None

        try:
            dummy_config = StrategyConfig(strategy_name=name, pair="BTCUSDT")
            temp_instance = strategy_class(dummy_config)

            return {
                'name': temp_instance.name,
                'description': temp_instance.description,
                'default_params': temp_instance.default_params,
                'params_schema': temp_instance.params_schema,
                'class_name': strategy_class.__name__
            }
        except Exception as e:
            logger.error(f"Error getting info for {name}: {e}")
            return None

    def get_all_info(self) -> Dict[str, Dict[str, Any]]:
        """Get info for all registered strategies"""
        return {
            name: self.get_strategy_info(name)
            for name in self._strategies.keys()
        }

    def clear(self) -> None:
        """Clear all registered strategies"""
        self._strategies.clear()


class StrategyLoader:
    """
    Dynamic loader for strategy classes.

    Automatically discovers and loads strategies from:
    - strategies/ directory (built-in)
    - Custom directories
    - Individual Python files
    """

    def __init__(self, registry: Optional[StrategyRegistry] = None):
        """
        Initialize loader.

        Args:
            registry: Registry to use (creates new if None)
        """
        self.registry = registry or StrategyRegistry()
        self._loaded_modules: Dict[str, Any] = {}

    def discover_strategies(self, directory: Optional[str] = None) -> List[str]:
        """
        Discover and register all strategies in a directory.

        Args:
            directory: Path to search (defaults to strategies/)

        Returns:
            List of discovered strategy names
        """
        if directory is None:
            directory = str(Path(__file__).parent)

        discovered = []
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return discovered

        # Find all Python files
        for file_path in dir_path.glob("*.py"):
            if file_path.name.startswith("_") or file_path.name in ("base.py", "loader.py"):
                continue

            try:
                strategies = self.load_from_file(str(file_path))
                discovered.extend(strategies)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        return discovered

    def load_from_file(self, file_path: str) -> List[str]:
        """
        Load strategies from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            List of loaded strategy names
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        module_name = f"strategies.{path.stem}"

        # Load module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        self._loaded_modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(f"Error executing module {file_path}: {e}")
            raise

        # Find strategy classes
        loaded = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BaseStrategy) and
                obj is not BaseStrategy and
                obj.__module__ == module_name):
                try:
                    self.registry.register(obj)
                    loaded.append(name)
                except Exception as e:
                    logger.error(f"Error registering {name}: {e}")

        return loaded

    def load_from_string(self, code: str, name: str = "custom") -> List[str]:
        """
        Load strategy from Python code string.

        Args:
            code: Python source code
            name: Module name for the code

        Returns:
            List of loaded strategy names
        """
        module_name = f"strategies.{name}"

        # Create module from string
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if spec is None:
            raise ImportError("Cannot create module spec")

        module = importlib.util.module_from_spec(spec)

        # Add base imports to namespace
        module.__dict__['BaseStrategy'] = BaseStrategy
        module.__dict__['StrategyConfig'] = StrategyConfig

        # Execute code
        exec(code, module.__dict__)
        self._loaded_modules[module_name] = module

        # Find and register strategies
        loaded = []
        for obj_name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BaseStrategy) and
                obj is not BaseStrategy):
                try:
                    self.registry.register(obj)
                    loaded.append(obj_name)
                except Exception as e:
                    logger.error(f"Error registering {obj_name}: {e}")

        return loaded

    def reload_strategy(self, name: str) -> bool:
        """
        Reload a strategy module.

        Args:
            name: Strategy name to reload

        Returns:
            True if successful
        """
        module_name = f"strategies.{name}"
        if module_name in self._loaded_modules:
            try:
                importlib.reload(self._loaded_modules[module_name])
                return True
            except Exception as e:
                logger.error(f"Error reloading {name}: {e}")
                return False
        return False


def load_builtin_strategies() -> StrategyRegistry:
    """
    Load all built-in strategies.

    Returns:
        Registry with loaded strategies
    """
    registry = StrategyRegistry()
    loader = StrategyLoader(registry)

    # Discover strategies in the strategies directory
    strategies_dir = Path(__file__).parent
    loader.discover_strategies(str(strategies_dir))

    logger.info(f"Loaded {len(registry.list_strategies())} built-in strategies")
    return registry


# Convenience function
def get_strategy(name: str, config: StrategyConfig) -> BaseStrategy:
    """
    Get a strategy instance by name.

    Args:
        name: Strategy identifier
        config: Strategy configuration

    Returns:
        Instantiated strategy
    """
    registry = StrategyRegistry()
    return registry.create(config)
