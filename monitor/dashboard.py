"""
VELAS Trading System - Web Dashboard

Simple web dashboard for monitoring.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from core.portfolio import PortfolioManager
from data.storage import DataStorage
from monitor.alerts import AlertManager


def create_dashboard_app(
    portfolio_manager: Optional[PortfolioManager] = None,
    storage: Optional[DataStorage] = None,
    alert_manager: Optional[AlertManager] = None,
) -> FastAPI:
    """
    Create FastAPI dashboard application.

    Args:
        portfolio_manager: Portfolio manager instance
        storage: Data storage instance
        alert_manager: Alert manager instance

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="VELAS Trading Dashboard",
        description="Trading system monitoring dashboard",
        version="1.0.0",
    )

    # Store references
    app.state.portfolio = portfolio_manager
    app.state.storage = storage
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
        if not app.state.storage:
            return []

        return app.state.storage.get_signals(limit=limit)

    @app.get("/api/trades")
    async def get_trades(limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades."""
        if not app.state.storage:
            return []

        return app.state.storage.get_trades(limit=limit)

    @app.get("/api/alerts")
    async def get_alerts(limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        if not app.state.alerts:
            return []

        alerts = app.state.alerts.get_recent_alerts(limit=limit)
        return [a.to_dict() for a in alerts]

    @app.post("/api/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert(alert_id: str) -> Dict[str, Any]:
        """Acknowledge alert."""
        if not app.state.alerts:
            raise HTTPException(status_code=404, detail="Alert manager not available")

        success = app.state.alerts.acknowledge(alert_id)
        return {"success": success}

    @app.get("/api/statistics")
    async def get_statistics() -> Dict[str, Any]:
        """Get trading statistics."""
        stats = {
            "timestamp": datetime.now().isoformat(),
        }

        if app.state.portfolio:
            portfolio_summary = app.state.portfolio.get_portfolio_summary()
            stats.update({
                "open_positions": portfolio_summary.get("open_positions", 0),
                "total_exposure": portfolio_summary.get("total_exposure", 0),
                "portfolio_heat": portfolio_summary.get("portfolio_heat", 0),
                "total_pnl": portfolio_summary.get("total_pnl", 0),
            })

        if app.state.storage:
            trades = app.state.storage.get_trades(limit=100)
            if trades:
                winners = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
                stats["win_rate"] = (winners / len(trades)) * 100 if trades else 0
                stats["total_trades"] = len(trades)

        return stats

    # === HTML Dashboard ===

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home() -> str:
        """Render dashboard home page."""
        return get_dashboard_html()

    return app


def get_dashboard_html() -> str:
    """Generate dashboard HTML."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VELAS Trading Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: #1a1a2e;
            padding: 20px;
            border-bottom: 1px solid #333;
        }
        .header h1 {
            color: #00d4aa;
            font-size: 24px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #333;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .card-title {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .card-value {
            font-size: 28px;
            font-weight: bold;
            color: #fff;
        }
        .card-value.positive { color: #00d4aa; }
        .card-value.negative { color: #ff4757; }
        .positions-table {
            width: 100%;
            border-collapse: collapse;
        }
        .positions-table th,
        .positions-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        .positions-table th {
            color: #888;
            font-weight: normal;
            font-size: 12px;
            text-transform: uppercase;
        }
        .badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge.long { background: #00d4aa33; color: #00d4aa; }
        .badge.short { background: #ff475733; color: #ff4757; }
        .alert-item {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 6px;
            border-left: 4px solid;
        }
        .alert-item.warning {
            background: #ff990033;
            border-color: #ff9900;
        }
        .alert-item.critical {
            background: #ff475733;
            border-color: #ff4757;
        }
        .refresh-btn {
            background: #00d4aa;
            color: #000;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
        }
        .refresh-btn:hover {
            background: #00b894;
        }
        #last-update {
            color: #666;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>VELAS Trading Dashboard</h1>
    </div>

    <div class="container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <span id="last-update">Last update: -</span>
            <button class="refresh-btn" onclick="refreshData()">Refresh</button>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Open Positions</span>
                </div>
                <div class="card-value" id="positions-count">-</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Portfolio Heat</span>
                </div>
                <div class="card-value" id="portfolio-heat">-</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Total P&L</span>
                </div>
                <div class="card-value" id="total-pnl">-</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Win Rate</span>
                </div>
                <div class="card-value" id="win-rate">-</div>
            </div>
        </div>

        <div class="grid">
            <div class="card" style="grid-column: span 2;">
                <div class="card-header">
                    <span class="card-title">Open Positions</span>
                </div>
                <table class="positions-table">
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

            <div class="card">
                <div class="card-header">
                    <span class="card-title">Active Alerts</span>
                </div>
                <div id="alerts-container">
                    <p style="color: #666;">No alerts</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function refreshData() {
            try {
                // Fetch status
                const statusRes = await fetch('/api/status');
                const status = await statusRes.json();

                // Update stats
                if (status.portfolio) {
                    document.getElementById('positions-count').textContent =
                        status.portfolio.open_positions || 0;
                    document.getElementById('portfolio-heat').textContent =
                        (status.portfolio.portfolio_heat || 0).toFixed(1) + '%';

                    const pnl = status.portfolio.total_pnl || 0;
                    const pnlEl = document.getElementById('total-pnl');
                    pnlEl.textContent = (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%';
                    pnlEl.className = 'card-value ' + (pnl >= 0 ? 'positive' : 'negative');
                }

                // Fetch statistics
                const statsRes = await fetch('/api/statistics');
                const stats = await statsRes.json();

                if (stats.win_rate !== undefined) {
                    document.getElementById('win-rate').textContent =
                        stats.win_rate.toFixed(1) + '%';
                }

                // Fetch positions
                const posRes = await fetch('/api/positions');
                const positions = await posRes.json();

                const tbody = document.getElementById('positions-tbody');
                if (positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #666;">No open positions</td></tr>';
                } else {
                    tbody.innerHTML = positions.map(p => `
                        <tr>
                            <td>${p.symbol}</td>
                            <td><span class="badge ${p.side.toLowerCase()}">${p.side}</span></td>
                            <td>${p.entry_price.toFixed(4)}</td>
                            <td>${(p.current_price || 0).toFixed(4)}</td>
                            <td style="color: ${p.unrealized_pnl_pct >= 0 ? '#00d4aa' : '#ff4757'}">
                                ${(p.unrealized_pnl_pct >= 0 ? '+' : '')}${(p.unrealized_pnl_pct || 0).toFixed(2)}%
                            </td>
                            <td>${(p.tps_hit || []).length}/6</td>
                        </tr>
                    `).join('');
                }

                // Fetch alerts
                const alertsRes = await fetch('/api/alerts?limit=5');
                const alerts = await alertsRes.json();

                const alertsContainer = document.getElementById('alerts-container');
                if (alerts.length === 0) {
                    alertsContainer.innerHTML = '<p style="color: #666;">No alerts</p>';
                } else {
                    alertsContainer.innerHTML = alerts.map(a => `
                        <div class="alert-item ${a.severity.toLowerCase()}">
                            <strong>${a.type}</strong><br>
                            <small>${a.message}</small>
                        </div>
                    `).join('');
                }

                // Update timestamp
                document.getElementById('last-update').textContent =
                    'Last update: ' + new Date().toLocaleTimeString();

            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }

        // Initial load
        refreshData();

        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);
    </script>
</body>
</html>
"""


class Dashboard:
    """
    Dashboard manager.
    """

    def __init__(
        self,
        portfolio_manager: Optional[PortfolioManager] = None,
        storage: Optional[DataStorage] = None,
        alert_manager: Optional[AlertManager] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        """
        Initialize dashboard.

        Args:
            portfolio_manager: Portfolio manager instance
            storage: Data storage instance
            alert_manager: Alert manager instance
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port

        self.app = create_dashboard_app(
            portfolio_manager=portfolio_manager,
            storage=storage,
            alert_manager=alert_manager,
        )

    def run(self) -> None:
        """Run dashboard server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )

    async def run_async(self) -> None:
        """Run dashboard server asynchronously."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
