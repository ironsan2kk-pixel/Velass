#!/usr/bin/env python3
"""
VELAS Trading System - Main Entry Point

Usage:
    python main.py                  # Run live trading
    python main.py --backtest       # Run backtest
    python main.py --optimize       # Run optimization
    python main.py --dashboard      # Run dashboard only
"""

import argparse
import asyncio
import sys
from pathlib import Path

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


async def run_live(config: dict, dry_run: bool = False) -> None:
    """Run live trading mode."""
    print("🚀 Starting VELAS Live Trading System...")
    print("=" * 50)
    
    # TODO: Implement in VELAS-08-LIVE
    # from live.engine import LiveEngine
    # engine = LiveEngine(config, dry_run=dry_run)
    # await engine.run()
    
    print("⚠️  Live trading not yet implemented")
    print("📋 See: VELAS-08-LIVE phase")


async def run_backtest(config: dict, pair: str = None, timeframe: str = None) -> None:
    """Run backtest mode."""
    print("📊 Starting VELAS Backtester...")
    print("=" * 50)
    
    # TODO: Implement in VELAS-02-BACKTEST
    # from backtest.engine import Backtester
    # backtester = Backtester(config)
    # results = await backtester.run(pair=pair, timeframe=timeframe)
    # backtester.print_report(results)
    
    print("⚠️  Backtester not yet implemented")
    print("📋 See: VELAS-02-BACKTEST phase")


async def run_optimization(config: dict) -> None:
    """Run optimization mode."""
    print("🔧 Starting VELAS Optimizer...")
    print("=" * 50)
    
    # TODO: Implement in VELAS-03-OPTIMIZE
    # from backtest.optimizer import Optimizer
    # optimizer = Optimizer(config)
    # results = await optimizer.run()
    # optimizer.save_results(results)
    
    print("⚠️  Optimizer not yet implemented")
    print("📋 See: VELAS-03-OPTIMIZE phase")


async def run_dashboard(config: dict) -> None:
    """Run dashboard only."""
    print("📈 Starting VELAS Dashboard...")
    print("=" * 50)
    
    # TODO: Implement in VELAS-09-MONITOR
    # from monitor.dashboard import Dashboard
    # dashboard = Dashboard(config)
    # await dashboard.run()
    
    print("⚠️  Dashboard not yet implemented")
    print("📋 See: VELAS-09-MONITOR phase")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    
    path = Path(config_path)
    if not path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


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
        asyncio.run(run_backtest(config, args.pair, args.timeframe))
    elif args.optimize:
        asyncio.run(run_optimization(config))
    elif args.dashboard:
        asyncio.run(run_dashboard(config))
    else:
        asyncio.run(run_live(config, args.dry_run))


if __name__ == "__main__":
    main()
