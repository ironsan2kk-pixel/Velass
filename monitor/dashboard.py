"""
VELAS Trading System - Web Dashboard with Admin Panel

Features:
- Live monitoring of positions and trades
- Settings management (API keys, Telegram token)
- Strategy management
- Backtest runner
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import yaml
import os

try:
    from fastapi import FastAPI, HTTPException, Request, Form
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from core.portfolio import PortfolioManager
from data.storage import DataStorage
from monitor.alerts import AlertManager


CONFIG_DIR = Path("config")
SECRETS_FILE = CONFIG_DIR / "secrets.yaml"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


def load_secrets() -> Dict[str, Any]:
    """Load secrets from file."""
    if SECRETS_FILE.exists():
        with open(SECRETS_FILE, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_secrets(secrets: Dict[str, Any]) -> None:
    """Save secrets to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SECRETS_FILE, "w") as f:
        yaml.dump(secrets, f, default_flow_style=False)


def load_config() -> Dict[str, Any]:
    """Load config from file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config: Dict[str, Any]) -> None:
    """Save config to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def create_dashboard_app(
    portfolio_manager: Optional[PortfolioManager] = None,
    storage: Optional[DataStorage] = None,
    alert_manager: Optional[AlertManager] = None,
) -> "FastAPI":
    """Create FastAPI dashboard application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed")

    app = FastAPI(
        title="VELAS Trading Dashboard",
        description="Trading system monitoring and administration",
        version="2.0.0",
    )

    # Store references
    app.state.portfolio = portfolio_manager
    app.state.storage = storage or DataStorage()
    app.state.alerts = alert_manager

    # === API Endpoints ===

    @app.get("/api/status")
    async def get_status() -> Dict[str, Any]:
        """Get system status."""
        status = {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
        }

        if app.state.portfolio:
            status["portfolio"] = app.state.portfolio.get_portfolio_summary()

        if app.state.alerts:
            status["alerts"] = app.state.alerts.get_summary()

        return status

    @app.get("/api/positions")
    async def get_positions() -> List[Dict[str, Any]]:
        """Get open positions."""
        if not app.state.portfolio:
            return []

        positions = app.state.portfolio.get_open_positions()
        return [p.to_dict() for p in positions]

    @app.get("/api/signals")
    async def get_signals(limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent signals."""
        return app.state.storage.get_signals(limit=limit)

    @app.get("/api/trades")
    async def get_trades(limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades."""
        return app.state.storage.get_trades(limit=limit)

    @app.get("/api/alerts")
    async def get_alerts(limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        if not app.state.alerts:
            return []
        alerts = app.state.alerts.get_recent_alerts(limit=limit)
        return [a.to_dict() for a in alerts]

    @app.get("/api/statistics")
    async def get_statistics() -> Dict[str, Any]:
        """Get trading statistics."""
        stats = {"timestamp": datetime.now().isoformat()}

        if app.state.portfolio:
            portfolio_summary = app.state.portfolio.get_portfolio_summary()
            stats.update({
                "open_positions": portfolio_summary.get("open_positions", 0),
                "total_exposure": portfolio_summary.get("total_exposure", 0),
                "portfolio_heat": portfolio_summary.get("portfolio_heat", 0),
                "total_pnl": portfolio_summary.get("total_pnl", 0),
            })

        trades = app.state.storage.get_trades(limit=100)
        if trades:
            winners = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
            stats["win_rate"] = (winners / len(trades)) * 100 if trades else 0
            stats["total_trades"] = len(trades)

        return stats

    # === Settings API ===

    @app.get("/api/settings")
    async def get_settings() -> Dict[str, Any]:
        """Get current settings (masked)."""
        secrets = load_secrets()
        config = load_config()

        # Mask sensitive values
        telegram_token = secrets.get("telegram", {}).get("bot_token", "")
        binance_key = secrets.get("binance", {}).get("api_key", "")

        return {
            "telegram": {
                "bot_token": "***" + telegram_token[-6:] if len(telegram_token) > 6 else "",
                "chat_id": secrets.get("telegram", {}).get("chat_id", ""),
                "is_configured": bool(telegram_token),
            },
            "binance": {
                "api_key": "***" + binance_key[-6:] if len(binance_key) > 6 else "",
                "is_configured": bool(binance_key),
            },
            "pairs": config.get("pairs", []),
            "timeframes": config.get("timeframes", ["1h"]),
        }

    @app.post("/api/settings/telegram")
    async def save_telegram_settings(
        bot_token: str = Form(...),
        chat_id: str = Form(""),
    ) -> Dict[str, Any]:
        """Save Telegram settings."""
        secrets = load_secrets()
        secrets["telegram"] = {
            "bot_token": bot_token,
            "chat_id": chat_id,
        }
        save_secrets(secrets)
        return {"success": True, "message": "Telegram settings saved"}

    @app.post("/api/settings/binance")
    async def save_binance_settings(
        api_key: str = Form(...),
        api_secret: str = Form(...),
    ) -> Dict[str, Any]:
        """Save Binance settings."""
        secrets = load_secrets()
        secrets["binance"] = {
            "api_key": api_key,
            "api_secret": api_secret,
        }
        save_secrets(secrets)
        return {"success": True, "message": "Binance settings saved"}

    @app.post("/api/settings/pairs")
    async def save_pairs_settings(
        pairs: str = Form(...),
        timeframes: str = Form("1h"),
    ) -> Dict[str, Any]:
        """Save trading pairs settings."""
        config = load_config()
        config["pairs"] = [p.strip().upper() for p in pairs.split(",") if p.strip()]
        config["timeframes"] = [t.strip() for t in timeframes.split(",") if t.strip()]
        save_config(config)
        return {"success": True, "message": "Trading pairs saved"}

    # === Strategies API ===

    @app.get("/api/strategies")
    async def get_strategies() -> List[Dict[str, Any]]:
        """Get available strategies."""
        try:
            from strategies import StrategyManager
            manager = StrategyManager()
            return manager.list_strategies()
        except Exception as e:
            return []

    @app.get("/api/backtest/results")
    async def get_backtest_results(limit: int = 20) -> List[Dict[str, Any]]:
        """Get backtest results."""
        return app.state.storage.get_backtest_results(limit=limit)

    @app.post("/api/backtest/run")
    async def run_backtest(
        pair: str = Form(...),
        strategy: str = Form(...),
        days: int = Form(30),
    ) -> Dict[str, Any]:
        """Run a backtest."""
        try:
            from strategies import StrategyManager
            manager = StrategyManager()
            result = manager.backtest(pair.upper(), strategy.lower(), days)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # === HTML Pages ===

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home() -> str:
        """Render dashboard home page."""
        return get_dashboard_html()

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page() -> str:
        """Render settings page."""
        return get_settings_html()

    @app.get("/strategies", response_class=HTMLResponse)
    async def strategies_page() -> str:
        """Render strategies page."""
        return get_strategies_html()

    @app.get("/backtest", response_class=HTMLResponse)
    async def backtest_page() -> str:
        """Render backtest page."""
        return get_backtest_html()

    return app


def get_base_html(title: str, content: str, active_page: str = "") -> str:
    """Generate base HTML with navigation."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - VELAS Trading</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .header {{
            background: #1a1a2e;
            padding: 15px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            color: #00d4aa;
            font-size: 20px;
        }}
        .nav {{
            display: flex;
            gap: 20px;
        }}
        .nav a {{
            color: #888;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            transition: all 0.2s;
        }}
        .nav a:hover, .nav a.active {{
            color: #fff;
            background: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .card {{
            background: #1a1a2e;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #333;
            margin-bottom: 20px;
        }}
        .card-title {{
            font-size: 18px;
            color: #00d4aa;
            margin-bottom: 15px;
        }}
        .form-group {{
            margin-bottom: 15px;
        }}
        .form-group label {{
            display: block;
            margin-bottom: 5px;
            color: #888;
            font-size: 14px;
        }}
        .form-group input, .form-group select {{
            width: 100%;
            padding: 10px;
            background: #0a0a0a;
            border: 1px solid #333;
            border-radius: 6px;
            color: #fff;
            font-size: 14px;
        }}
        .form-group input:focus {{
            outline: none;
            border-color: #00d4aa;
        }}
        .btn {{
            background: #00d4aa;
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
        }}
        .btn:hover {{ background: #00b894; }}
        .btn-secondary {{
            background: #333;
            color: #fff;
        }}
        .btn-secondary:hover {{ background: #444; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .alert {{
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }}
        .alert-success {{ background: #00d4aa33; border: 1px solid #00d4aa; }}
        .alert-error {{ background: #ff475733; border: 1px solid #ff4757; }}
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }}
        .status-dot.active {{ background: #00d4aa; }}
        .status-dot.inactive {{ background: #666; }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{ color: #888; font-weight: normal; font-size: 12px; text-transform: uppercase; }}
        .positive {{ color: #00d4aa; }}
        .negative {{ color: #ff4757; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>VELAS Trading System</h1>
        <nav class="nav">
            <a href="/" class="{'active' if active_page == 'dashboard' else ''}">Dashboard</a>
            <a href="/strategies" class="{'active' if active_page == 'strategies' else ''}">Strategies</a>
            <a href="/backtest" class="{'active' if active_page == 'backtest' else ''}">Backtest</a>
            <a href="/settings" class="{'active' if active_page == 'settings' else ''}">Settings</a>
        </nav>
    </div>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""


def get_settings_html() -> str:
    """Generate settings page HTML."""
    content = """
    <h2 style="margin-bottom: 20px;">System Settings</h2>
    <div id="alert-container"></div>

    <div class="grid">
        <div class="card">
            <h3 class="card-title">Telegram Bot</h3>
            <form id="telegram-form">
                <div class="form-group">
                    <label>Bot Token</label>
                    <input type="password" name="bot_token" id="tg-token" placeholder="Enter Telegram Bot Token">
                </div>
                <div class="form-group">
                    <label>Chat ID (optional)</label>
                    <input type="text" name="chat_id" id="tg-chat" placeholder="Your Telegram Chat ID">
                </div>
                <button type="submit" class="btn">Save Telegram Settings</button>
            </form>
            <p style="margin-top: 10px; color: #666; font-size: 12px;">
                Get token from @BotFather on Telegram
            </p>
        </div>

        <div class="card">
            <h3 class="card-title">Binance API</h3>
            <form id="binance-form">
                <div class="form-group">
                    <label>API Key</label>
                    <input type="password" name="api_key" id="binance-key" placeholder="Enter Binance API Key">
                </div>
                <div class="form-group">
                    <label>API Secret</label>
                    <input type="password" name="api_secret" id="binance-secret" placeholder="Enter Binance API Secret">
                </div>
                <button type="submit" class="btn">Save Binance Settings</button>
            </form>
        </div>
    </div>

    <div class="card">
        <h3 class="card-title">Trading Pairs</h3>
        <form id="pairs-form">
            <div class="form-group">
                <label>Pairs (comma-separated)</label>
                <input type="text" name="pairs" id="pairs-input" placeholder="BTCUSDT, ETHUSDT, BNBUSDT">
            </div>
            <div class="form-group">
                <label>Timeframes (comma-separated)</label>
                <input type="text" name="timeframes" id="timeframes-input" placeholder="1h, 4h">
            </div>
            <button type="submit" class="btn">Save Pairs</button>
        </form>
    </div>

    <div class="card">
        <h3 class="card-title">Current Configuration</h3>
        <div id="current-config">Loading...</div>
    </div>

    <script>
        async function loadSettings() {
            try {
                const res = await fetch('/api/settings');
                const settings = await res.json();

                document.getElementById('current-config').innerHTML = `
                    <p><span class="status-dot ${settings.telegram.is_configured ? 'active' : 'inactive'}"></span>
                       Telegram: ${settings.telegram.is_configured ? 'Configured (' + settings.telegram.bot_token + ')' : 'Not configured'}</p>
                    <p><span class="status-dot ${settings.binance.is_configured ? 'active' : 'inactive'}"></span>
                       Binance: ${settings.binance.is_configured ? 'Configured (' + settings.binance.api_key + ')' : 'Not configured'}</p>
                    <p>Pairs: ${settings.pairs.join(', ') || 'None'}</p>
                    <p>Timeframes: ${settings.timeframes.join(', ')}</p>
                `;

                if (settings.pairs.length > 0) {
                    document.getElementById('pairs-input').value = settings.pairs.join(', ');
                }
                document.getElementById('timeframes-input').value = settings.timeframes.join(', ');
                if (settings.telegram.chat_id) {
                    document.getElementById('tg-chat').value = settings.telegram.chat_id;
                }
            } catch (e) {
                console.error('Error loading settings:', e);
            }
        }

        function showAlert(message, type = 'success') {
            const container = document.getElementById('alert-container');
            container.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
            setTimeout(() => container.innerHTML = '', 3000);
        }

        document.getElementById('telegram-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            try {
                const res = await fetch('/api/settings/telegram', { method: 'POST', body: formData });
                const data = await res.json();
                showAlert(data.message);
                loadSettings();
            } catch (e) {
                showAlert('Error saving settings', 'error');
            }
        });

        document.getElementById('binance-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            try {
                const res = await fetch('/api/settings/binance', { method: 'POST', body: formData });
                const data = await res.json();
                showAlert(data.message);
                loadSettings();
            } catch (e) {
                showAlert('Error saving settings', 'error');
            }
        });

        document.getElementById('pairs-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            try {
                const res = await fetch('/api/settings/pairs', { method: 'POST', body: formData });
                const data = await res.json();
                showAlert(data.message);
                loadSettings();
            } catch (e) {
                showAlert('Error saving settings', 'error');
            }
        });

        loadSettings();
    </script>
    """
    return get_base_html("Settings", content, "settings")


def get_strategies_html() -> str:
    """Generate strategies page HTML."""
    content = """
    <h2 style="margin-bottom: 20px;">Available Strategies</h2>

    <div id="strategies-grid" class="grid">
        <p>Loading strategies...</p>
    </div>

    <script>
        async function loadStrategies() {
            try {
                const res = await fetch('/api/strategies');
                const strategies = await res.json();

                const grid = document.getElementById('strategies-grid');
                if (strategies.length === 0) {
                    grid.innerHTML = '<p>No strategies available</p>';
                    return;
                }

                grid.innerHTML = strategies.map(s => `
                    <div class="card">
                        <h3 class="card-title">${s.name}</h3>
                        <p style="color: #888; margin-bottom: 10px;">${s.description}</p>
                        <p><strong>Parameters:</strong> ${s.params.slice(0, 5).join(', ')}${s.params.length > 5 ? '...' : ''}</p>
                        <a href="/backtest?strategy=${s.name}" class="btn btn-secondary" style="display: inline-block; margin-top: 10px; text-decoration: none;">
                            Run Backtest
                        </a>
                    </div>
                `).join('');
            } catch (e) {
                document.getElementById('strategies-grid').innerHTML = '<p>Error loading strategies</p>';
            }
        }

        loadStrategies();
    </script>
    """
    return get_base_html("Strategies", content, "strategies")


def get_backtest_html() -> str:
    """Generate backtest page HTML."""
    content = """
    <h2 style="margin-bottom: 20px;">Backtest Runner</h2>

    <div class="grid">
        <div class="card">
            <h3 class="card-title">Run New Backtest</h3>
            <form id="backtest-form">
                <div class="form-group">
                    <label>Trading Pair</label>
                    <input type="text" name="pair" id="bt-pair" placeholder="BTCUSDT" required>
                </div>
                <div class="form-group">
                    <label>Strategy</label>
                    <select name="strategy" id="bt-strategy" required>
                        <option value="">Select strategy...</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Days</label>
                    <input type="number" name="days" id="bt-days" value="30" min="1" max="365">
                </div>
                <button type="submit" class="btn" id="run-btn">Run Backtest</button>
            </form>
        </div>

        <div class="card">
            <h3 class="card-title">Results</h3>
            <div id="backtest-result">
                <p style="color: #666;">Run a backtest to see results</p>
            </div>
        </div>
    </div>

    <div class="card">
        <h3 class="card-title">Recent Backtests</h3>
        <table>
            <thead>
                <tr>
                    <th>Pair</th>
                    <th>Strategy</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Return</th>
                    <th>Max DD</th>
                    <th>Date</th>
                </tr>
            </thead>
            <tbody id="results-tbody">
                <tr><td colspan="7" style="text-align: center; color: #666;">Loading...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        async function loadStrategies() {
            try {
                const res = await fetch('/api/strategies');
                const strategies = await res.json();
                const select = document.getElementById('bt-strategy');

                strategies.forEach(s => {
                    const option = document.createElement('option');
                    option.value = s.name;
                    option.textContent = s.name;
                    select.appendChild(option);
                });

                // Check URL params
                const params = new URLSearchParams(window.location.search);
                if (params.get('strategy')) {
                    select.value = params.get('strategy');
                }
            } catch (e) {
                console.error('Error loading strategies:', e);
            }
        }

        async function loadResults() {
            try {
                const res = await fetch('/api/backtest/results?limit=10');
                const results = await res.json();

                const tbody = document.getElementById('results-tbody');
                if (results.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #666;">No backtest results</td></tr>';
                    return;
                }

                tbody.innerHTML = results.map(r => `
                    <tr>
                        <td>${r.pair}</td>
                        <td>${r.strategy_name}</td>
                        <td>${r.total_trades}</td>
                        <td>${r.win_rate.toFixed(1)}%</td>
                        <td class="${r.total_return >= 0 ? 'positive' : 'negative'}">${r.total_return.toFixed(2)}%</td>
                        <td class="negative">${r.max_drawdown.toFixed(2)}%</td>
                        <td>${r.created_at ? r.created_at.slice(0, 10) : '-'}</td>
                    </tr>
                `).join('');
            } catch (e) {
                console.error('Error loading results:', e);
            }
        }

        document.getElementById('backtest-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('run-btn');
            btn.textContent = 'Running...';
            btn.disabled = true;

            const formData = new FormData(e.target);
            const resultDiv = document.getElementById('backtest-result');

            try {
                const res = await fetch('/api/backtest/run', { method: 'POST', body: formData });
                const result = await res.json();

                if (result.error) {
                    resultDiv.innerHTML = `<p class="negative">Error: ${result.error}</p>`;
                } else {
                    resultDiv.innerHTML = `
                        <p><strong>Total Trades:</strong> ${result.total_trades}</p>
                        <p><strong>Win Rate:</strong> ${result.win_rate?.toFixed(1) || 0}%</p>
                        <p><strong>Profit Factor:</strong> ${result.profit_factor?.toFixed(2) || 0}</p>
                        <p><strong>Total Return:</strong> <span class="${result.total_return >= 0 ? 'positive' : 'negative'}">${result.total_return?.toFixed(2) || 0}%</span></p>
                        <p><strong>Max Drawdown:</strong> <span class="negative">${result.max_drawdown?.toFixed(2) || 0}%</span></p>
                        <p><strong>Sharpe Ratio:</strong> ${result.sharpe_ratio?.toFixed(2) || 0}</p>
                    `;
                    loadResults();
                }
            } catch (e) {
                resultDiv.innerHTML = `<p class="negative">Error: ${e.message}</p>`;
            } finally {
                btn.textContent = 'Run Backtest';
                btn.disabled = false;
            }
        });

        loadStrategies();
        loadResults();
    </script>
    """
    return get_base_html("Backtest", content, "backtest")


def get_dashboard_html() -> str:
    """Generate dashboard home page HTML."""
    content = """
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h2>Dashboard</h2>
        <div>
            <span id="last-update" style="color: #666; margin-right: 15px;">Last update: -</span>
            <button class="btn" onclick="refreshData()">Refresh</button>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <p style="color: #888; font-size: 12px; text-transform: uppercase;">Open Positions</p>
            <p style="font-size: 28px; font-weight: bold;" id="positions-count">-</p>
        </div>
        <div class="card">
            <p style="color: #888; font-size: 12px; text-transform: uppercase;">Portfolio Heat</p>
            <p style="font-size: 28px; font-weight: bold;" id="portfolio-heat">-</p>
        </div>
        <div class="card">
            <p style="color: #888; font-size: 12px; text-transform: uppercase;">Total P&L</p>
            <p style="font-size: 28px; font-weight: bold;" id="total-pnl">-</p>
        </div>
        <div class="card">
            <p style="color: #888; font-size: 12px; text-transform: uppercase;">Win Rate</p>
            <p style="font-size: 28px; font-weight: bold;" id="win-rate">-</p>
        </div>
    </div>

    <div class="card">
        <h3 class="card-title">Open Positions</h3>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Entry</th>
                    <th>Current</th>
                    <th>P&L</th>
                    <th>TPs Hit</th>
                </tr>
            </thead>
            <tbody id="positions-tbody">
                <tr><td colspan="6" style="text-align: center; color: #666;">Loading...</td></tr>
            </tbody>
        </table>
    </div>

    <div class="grid">
        <div class="card">
            <h3 class="card-title">Recent Trades</h3>
            <div id="trades-container">Loading...</div>
        </div>
        <div class="card">
            <h3 class="card-title">Alerts</h3>
            <div id="alerts-container">No alerts</div>
        </div>
    </div>

    <script>
        async function refreshData() {
            try {
                const statusRes = await fetch('/api/status');
                const status = await statusRes.json();

                if (status.portfolio) {
                    document.getElementById('positions-count').textContent = status.portfolio.open_positions || 0;
                    document.getElementById('portfolio-heat').textContent = (status.portfolio.portfolio_heat || 0).toFixed(1) + '%';

                    const pnl = status.portfolio.total_pnl || 0;
                    const pnlEl = document.getElementById('total-pnl');
                    pnlEl.textContent = (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%';
                    pnlEl.className = pnl >= 0 ? 'positive' : 'negative';
                }

                const statsRes = await fetch('/api/statistics');
                const stats = await statsRes.json();
                if (stats.win_rate !== undefined) {
                    document.getElementById('win-rate').textContent = stats.win_rate.toFixed(1) + '%';
                }

                const posRes = await fetch('/api/positions');
                const positions = await posRes.json();
                const tbody = document.getElementById('positions-tbody');
                if (positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #666;">No open positions</td></tr>';
                } else {
                    tbody.innerHTML = positions.map(p => `
                        <tr>
                            <td>${p.symbol}</td>
                            <td><span style="color: ${p.side === 'LONG' ? '#00d4aa' : '#ff4757'}">${p.side}</span></td>
                            <td>${p.entry_price?.toFixed(4) || '-'}</td>
                            <td>${p.current_price?.toFixed(4) || '-'}</td>
                            <td class="${(p.unrealized_pnl_pct || 0) >= 0 ? 'positive' : 'negative'}">
                                ${((p.unrealized_pnl_pct || 0) >= 0 ? '+' : '')}${(p.unrealized_pnl_pct || 0).toFixed(2)}%
                            </td>
                            <td>${(p.tps_hit || []).length}/6</td>
                        </tr>
                    `).join('');
                }

                const tradesRes = await fetch('/api/trades?limit=5');
                const trades = await tradesRes.json();
                const tradesContainer = document.getElementById('trades-container');
                if (trades.length === 0) {
                    tradesContainer.innerHTML = '<p style="color: #666;">No trades yet</p>';
                } else {
                    tradesContainer.innerHTML = trades.map(t => `
                        <p style="margin-bottom: 8px;">
                            <span style="color: ${t.side === 'LONG' ? '#00d4aa' : '#ff4757'}">${t.side}</span>
                            ${t.symbol} -
                            <span class="${(t.pnl_pct || 0) >= 0 ? 'positive' : 'negative'}">
                                ${((t.pnl_pct || 0) >= 0 ? '+' : '')}${(t.pnl_pct || 0).toFixed(2)}%
                            </span>
                        </p>
                    `).join('');
                }

                document.getElementById('last-update').textContent = 'Last update: ' + new Date().toLocaleTimeString();
            } catch (error) {
                console.error('Error:', error);
            }
        }

        refreshData();
        setInterval(refreshData, 30000);
    </script>
    """
    return get_base_html("Dashboard", content, "dashboard")


class Dashboard:
    """Dashboard manager."""

    def __init__(
        self,
        portfolio_manager: Optional[PortfolioManager] = None,
        storage: Optional[DataStorage] = None,
        alert_manager: Optional[AlertManager] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self.host = host
        self.port = port
        self.app = create_dashboard_app(
            portfolio_manager=portfolio_manager,
            storage=storage,
            alert_manager=alert_manager,
        )

    def run(self) -> None:
        """Run dashboard server."""
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

    async def run_async(self) -> None:
        """Run dashboard server asynchronously."""
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
