#!/usr/bin/env python3
"""
VELAS Trading System - Main Entry Point

Usage:
    python main.py                  # Run live trading
    python main.py --backtest       # Run backtest
    python main.py --optimize       # Run optimization
    python main.py --dashboard      # Run dashboard only
    python main.py --download       # Download historical data
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VELAS Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest mode",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimization mode",
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run dashboard only",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download historical data",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--pair",
        type=str,
        help="Single pair to process (for testing)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        help="Single timeframe to process (for testing)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Backtest start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="Backtest end date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no real orders/messages)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml

    path = Path(config_path)
    if not path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Load secrets if exists
    secrets_path = Path("config/secrets.yaml")
    if secrets_path.exists():
        with open(secrets_path, "r") as f:
            secrets = yaml.safe_load(f)
            config["secrets"] = secrets

    return config


async def run_live(config: dict, dry_run: bool = False) -> None:
    """Run live trading mode."""
    from live.engine import LiveEngine
    from monitor.logger import setup_logger

    logger = setup_logger(
        log_dir=config.get("logging", {}).get("file", {}).get("path", "logs"),
        log_level=config.get("system", {}).get("log_level", "INFO"),
    )

    print("🚀 Starting VELAS Live Trading System...")
    print("=" * 50)

    if dry_run:
        print("⚠️  DRY RUN MODE - No real signals will be sent")
        print()

    engine = LiveEngine(config, dry_run=dry_run)

    try:
        await engine.run()
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
        await engine.shutdown()


async def run_backtest(
    config: dict,
    pair: Optional[str] = None,
    timeframe: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    """Run backtest mode."""
    from backtest.engine import Backtester, BacktestConfig
    from backtest.metrics import MetricsCalculator
    from backtest.reports import ReportGenerator, print_quick_stats
    from data.storage import DataStorage
    from core.strategy import VelasParams

    print("📊 Starting VELAS Backtester...")
    print("=" * 50)

    # Load data
    storage = DataStorage()

    # Get pairs and timeframes
    pairs = [pair] if pair else config.get("pairs", [])
    timeframes_list = [timeframe] if timeframe else config.get("timeframes", [])

    # Configure backtest
    strategy_config = config.get("strategy", {})
    velas_config = config.get("velas_params", {})

    backtest_config = BacktestConfig(
        start_date=start_date or config.get("backtest", {}).get("start_date"),
        end_date=end_date or config.get("backtest", {}).get("end_date"),
        strategy_params=VelasParams(**velas_config),
        tp_percents=strategy_config.get("tp_levels", [1.0, 2.0, 3.0, 4.0, 7.5, 14.0]),
        tp_distribution=strategy_config.get("tp_distribution", [17, 17, 17, 17, 16, 16]),
        sl_percent=strategy_config.get("sl_percent", 8.5),
        stop_management=strategy_config.get("stop_management", "cascade"),
        use_filters=True,
        filter_config=config.get("filters", {}),
    )

    backtester = Backtester(config=backtest_config)
    calc = MetricsCalculator()
    generator = ReportGenerator()

    all_results = {}

    for symbol in pairs:
        all_results[symbol] = {}

        for tf in timeframes_list:
            print(f"\n📈 Running backtest: {symbol} {tf}")

            # Load data
            df = storage.load_ohlcv(symbol, tf)

            if df is None or len(df) < 100:
                print(f"   ⚠️  Insufficient data for {symbol} {tf}")
                continue

            # Run backtest
            result = backtester.run(df, symbol, tf)

            # Calculate metrics
            calc.calculate_all(result)

            # Store result
            all_results[symbol][tf] = result

            # Print quick stats
            print_quick_stats(result)

    # Generate comparison report
    if all_results:
        print("\n" + "=" * 80)
        report = generator.generate_comparison_report(all_results)
        print(report)


async def run_optimization(config: dict) -> None:
    """Run optimization mode."""
    from backtest.engine import WalkForwardOptimizer, Backtester, BacktestConfig
    from backtest.metrics import MetricsCalculator
    from data.storage import DataStorage
    from core.strategy import VelasParams

    print("🔧 Starting VELAS Optimizer...")
    print("=" * 50)

    storage = DataStorage()
    optimizer = WalkForwardOptimizer(
        optimization_window_months=config.get("backtest", {}).get("optimization_window_months", 6),
        test_window_months=config.get("backtest", {}).get("test_window_months", 2),
        step_months=config.get("backtest", {}).get("step_months", 2),
    )

    # Parameter grid
    param_grid = {
        "i1": [40, 60, 80, 100],
        "i2": [10, 14, 18],
        "i3": [1.0, 1.2, 1.5],
        "i4": [1.2, 1.5, 2.0],
        "i5": [1.0, 1.5, 2.0],
    }

    pairs = config.get("pairs", ["BTCUSDT"])[:3]  # Optimize on first 3 pairs

    for symbol in pairs:
        print(f"\n📊 Optimizing {symbol}...")

        df = storage.load_ohlcv(symbol, "1h")
        if df is None or len(df) < 500:
            print(f"   ⚠️  Insufficient data for {symbol}")
            continue

        best_params, results = optimizer.optimize_params(
            df,
            param_grid,
            metric="sharpe_ratio",
        )

        print(f"   Best params: {best_params.to_dict()}")
        print(f"   Best score: {results['best_score']:.2f}")


async def run_dashboard(config: dict) -> None:
    """Run dashboard only."""
    from monitor.dashboard import Dashboard
    from monitor.alerts import AlertManager
    from core.portfolio import PortfolioManager
    from data.storage import DataStorage

    print("📈 Starting VELAS Dashboard...")
    print("=" * 50)

    # Initialize components
    portfolio_config = config.get("portfolio", {})
    portfolio_manager = PortfolioManager(
        max_positions=portfolio_config.get("max_positions", 5),
        groups=portfolio_config.get("groups", {}),
    )

    storage = DataStorage()
    alert_manager = AlertManager(storage=storage)

    dashboard_config = config.get("monitoring", {}).get("dashboard", {})
    dashboard = Dashboard(
        portfolio_manager=portfolio_manager,
        storage=storage,
        alert_manager=alert_manager,
        host="0.0.0.0",
        port=dashboard_config.get("port", 8080),
    )

    print(f"🌐 Dashboard running at http://localhost:{dashboard_config.get('port', 8080)}")
    dashboard.run()


async def run_download(config: dict) -> None:
    """Download historical data."""
    from data.binance_client import BinanceClient

    print("📥 Downloading historical data...")
    print("=" * 50)

    pairs = config.get("pairs", [])
    timeframes = config.get("timeframes", [])
    start_date = config.get("backtest", {}).get("start_date", "2023-01-01")

    client = BinanceClient()

    try:
        saved = await client.download_all_pairs(
            pairs=pairs,
            timeframes=timeframes,
            start_date=start_date,
        )

        print(f"\n✅ Downloaded data for {len(saved)} pairs")

    finally:
        await client.close()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Print banner
    print()
    print("╔═══════════════════════════════════════════════════╗")
    print("║         VELAS TRADING SYSTEM v1.0.0               ║")
    print("║         Automated Crypto Signal Generator         ║")
    print("╚═══════════════════════════════════════════════════╝")
    print()

    # Load config
    config = load_config(args.config)

    if args.verbose:
        print(f"📁 Config loaded from: {args.config}")
        print(f"📊 Pairs: {len(config.get('pairs', []))}")
        print(f"⏱️  Timeframes: {config.get('timeframes', [])}")
        print()

    # Run appropriate mode
    if args.backtest:
        asyncio.run(run_backtest(
            config,
            args.pair,
            args.timeframe,
            args.start_date,
            args.end_date,
        ))
    elif args.optimize:
        asyncio.run(run_optimization(config))
    elif args.dashboard:
        asyncio.run(run_dashboard(config))
    elif args.download:
        asyncio.run(run_download(config))
    else:
        asyncio.run(run_live(config, args.dry_run))


if __name__ == "__main__":
    main()
