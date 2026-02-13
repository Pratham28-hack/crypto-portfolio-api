from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
from typing import Dict

# Caching & rate limiting imports
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ────────────────────────────────────────────────────────────────
#  APP DEFINITION – must be FIRST so decorators can see it
# ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Crypto Portfolio Tracker & Risk Analyzer API",
    description="Real-time crypto prices, portfolio valuation, and risk metrics using CoinGecko API. Supports any valid coin_id from CoinGecko.",
    version="1.0.2"  # bumped to mark fixes
)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Custom handler for slowapi rate limit exceeded
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded (20 requests per minute). Please try again later."}
    )

# Caching startup event
@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend(), prefix="crypto-cache")  # type: ignore[call-arg]

# Safe wrapper for all CoinGecko calls (handles 429 gracefully)
def safe_get(url: str, params: dict | None = None, timeout: int = 10):
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="CoinGecko rate limit reached (30 calls/min free tier). Try again in 1–2 minutes."
            )
        raise HTTPException(status_code=500, detail=f"CoinGecko API error: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Network error contacting CoinGecko: {str(e)}")

# ────────────────────────────────────────────────────────────────
#  MODELS
# ────────────────────────────────────────────────────────────────

class PortfolioRequest(BaseModel):
    holdings: Dict[str, float]  # e.g. {"bitcoin": 1.5, "ethereum": 10}

# ────────────────────────────────────────────────────────────────
#  ENDPOINTS
# ────────────────────────────────────────────────────────────────

@app.get("/coins")
def get_coins():
    """List of all supported cryptocurrencies (CoinGecko /coins/list)"""
    try:
        return safe_get(f"{COINGECKO_BASE}/coins/list")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prices")
@limiter.limit("20/minute")
@cache(expire=60)
def get_prices(
    request: Request,                           # ← required, no default, comes FIRST
    ids: str = Query(..., description="Comma-separated coin IDs, e.g. bitcoin,ethereum,solana")
):
    """Current USD prices for one or more coins"""
    try:
        data = safe_get(
            f"{COINGECKO_BASE}/simple/price",
            params={"ids": ids, "vs_currencies": "usd"}
        )
        if not data:
            raise HTTPException(status_code=404, detail="No valid coins found. Use /coins to check valid IDs.")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/historical/{coin_id}")
def get_historical(coin_id: str, days: int = Query(30, ge=1, le=365)):
    """Historical daily prices (market_chart) for the last N days"""
    try:
        data = safe_get(
            f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days}
        )
        if "prices" not in data:
            raise HTTPException(status_code=404, detail=f"Coin '{coin_id}' not found.")
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms").dt.date
        return prices[["date", "price"]].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio/value")
def portfolio_value(request: PortfolioRequest):
    """Current total portfolio value in USD"""
    total_value = 0.0
    details = {}
    for coin_id, amount in request.holdings.items():
        try:
            data = safe_get(
                f"{COINGECKO_BASE}/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd"}
            )
            price = data.get(coin_id, {}).get("usd")
            if price is None:
                details[coin_id] = {"error": f"Coin '{coin_id}' not found."}
                continue
            value = price * amount
            total_value += value
            details[coin_id] = {"amount": amount, "price_usd": price, "value_usd": value}
        except Exception as e:
            details[coin_id] = {"error": str(e)}
    return {"total_value_usd": total_value, "details": details}

@app.post("/portfolio/risk")
def portfolio_risk(request: PortfolioRequest, days: int = Query(365, ge=30, le=1825)):
    """Portfolio risk metrics: volatility, Sharpe, max drawdown, correlations"""
    if not request.holdings:
        raise HTTPException(status_code=400, detail="Holdings cannot be empty")

    prices_dict = {}
    for coin_id in request.holdings:
        try:
            data = safe_get(
                f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
                params={"vs_currency": "usd", "days": days, "interval": "daily"}
            )
            if "prices" not in data:
                raise HTTPException(status_code=404, detail=f"Coin '{coin_id}' not found.")
            df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            df["returns"] = df["price"].pct_change().fillna(0)
            prices_dict[coin_id] = df
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed for {coin_id}: {str(e)}")

    all_returns = pd.concat(
        [prices_dict[coin]["returns"].rename(coin) for coin in request.holdings],
        axis=1
    ).dropna()

    if all_returns.empty:
        raise HTTPException(status_code=400, detail="Insufficient overlapping historical data")

    weights = np.array([request.holdings[coin] for coin in all_returns.columns]) / sum(request.holdings.values())
    port_returns = (all_returns * weights).sum(axis=1)

    risk_free_rate = 0.02
    annualized_vol = port_returns.std() * np.sqrt(252)
    annualized_mean = port_returns.mean() * 252
    sharpe = (annualized_mean - risk_free_rate) / annualized_vol if annualized_vol != 0 else 0

    cum_ret = (1 + port_returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()

    correlations = all_returns.corr().to_dict()

    return {
        "volatility_annualized": float(annualized_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "correlations": correlations,
        "period_days": days
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)