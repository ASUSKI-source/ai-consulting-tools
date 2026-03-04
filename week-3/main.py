from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

from stock_data import get_stock_data
from crypto_data import get_crypto_data
from indicators import (
    calculate_sma,
    calculate_rsi,
    calculate_price_momentum,
    find_support_resistance,
)
from format_prompt import build_stock_analysis_prompt
from market_analyzer import (
    SYSTEM_PROMPT,
    CRYPTO_SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
    build_crypto_analysis_prompt,
    build_stock_prompt_for_ticker,
    build_crypto_prompt_for_coin,
)
import anthropic

load_dotenv()

port = int(os.getenv('PORT', 8000))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(BASE_DIR, 'static')

CURRENT_MODEL = "claude-sonnet-4-5"

COST_PER_INPUT_TOKEN = 0.000003
COST_PER_OUTPUT_TOKEN = 0.000015

app = FastAPI(title='Market Analyzer API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
def serve_frontend():
    """Serves the main index.html page from the static folder."""
    return FileResponse(os.path.join(static_path, 'index.html'))

# --- Pydantic request models ---

class StockRequest(BaseModel):
    ticker: str


class CryptoRequest(BaseModel):
    coin_id: str


class CompareRequest(BaseModel):
    asset1: str
    asset2: str
    asset1_type: str  # 'stock' or 'crypto'
    asset2_type: str  # 'stock' or 'crypto'


def _get_prompt_and_label(asset_id: str, asset_type: str) -> tuple[str, str]:
    """Build formatted prompt and label for one asset (stock or crypto)."""
    asset_type_lower = asset_type.lower().strip()
    if asset_type_lower == 'crypto':
        return build_crypto_prompt_for_coin(asset_id.strip())
    return build_stock_prompt_for_ticker(asset_id.strip())


# Root endpoint: returns API status and version info.



# Health check endpoint: returns whether the service is healthy.
@app.get('/health')
def health():
    try:
        return {'healthy': True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Test endpoint: accepts a ticker path parameter and echoes it back (uppercased).
@app.get('/test/{ticker}')
def test_ticker(ticker: str):
    try:
        return {'ticker': ticker.upper(), 'received': True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# POST /analyze/stock — Input: JSON body with ticker. Fetches stock data, computes indicators,
# builds analysis prompt, calls Claude, returns full analysis plus token usage and estimated cost.
# Output: ticker, company_name, current_price, market_cap, pe_ratio, week_high_52, week_low_52,
# rsi, sma_5, sma_20, momentum, support_resistance, ai_analysis, input_tokens, output_tokens, estimated_cost.
@app.post('/analyze/stock')
def analyze_stock_endpoint(request: StockRequest):
    try:
        ticker = request.ticker.upper()
        stock_data = get_stock_data(ticker)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=f'Could not fetch data for {request.ticker}',
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        prices = stock_data.get('history') or []
        if not prices:
            raise HTTPException(
                status_code=404,
                detail=f'Could not fetch data for {request.ticker}',
            )

        sma_5 = calculate_sma(prices, 5)
        sma_20 = calculate_sma(prices, 20)
        rsi = calculate_rsi(prices)
        momentum = calculate_price_momentum(prices)
        support_resistance = find_support_resistance(prices)

        indicators = {
            'sma_5': sma_5,
            'sma_20': sma_20,
            'rsi': rsi,
            'momentum': momentum,
            'support_resistance': support_resistance,
        }
        formatted_prompt = build_stock_analysis_prompt(stock_data, indicators)

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail='ANTHROPIC_API_KEY not set')

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CURRENT_MODEL,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': formatted_prompt}],
        )

        ai_text = ''
        if response.content and len(response.content) > 0:
            ai_text = response.content[0].text
        else:
            ai_text = 'Claude returned an empty response.'

        usage = getattr(response, 'usage', None)
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        estimated_cost = (input_tokens * COST_PER_INPUT_TOKEN) + (
            output_tokens * COST_PER_OUTPUT_TOKEN
        )

        return {
            'ticker': stock_data.get('ticker'),
            'company_name': stock_data.get('company_name'),
            'current_price': stock_data.get('current_price'),
            'market_cap': stock_data.get('market_cap'),
            'pe_ratio': stock_data.get('pe_ratio'),
            'week_high_52': stock_data.get('week_high_52'),
            'week_low_52': stock_data.get('week_low_52'),
            'rsi': rsi,
            'sma_5': sma_5,
            'sma_20': sma_20,
            'momentum': momentum,
            'support_resistance': support_resistance,
            'analysis': ai_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'estimated_cost': round(estimated_cost, 7),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# POST /analyze/crypto — Input: JSON body with coin_id (CoinGecko ID). Fetches crypto data,
# builds prompt, calls Claude with crypto system prompt. Output: name, symbol, current_price_usd,
# market_cap_usd, volume_24h, price_change_24h/7d/30d_pct, ath, ath_change_pct, ai_analysis, token usage, estimated_cost.
@app.post('/analyze/crypto')
def analyze_crypto_endpoint(request: CryptoRequest):
    try:
        coin_id = request.coin_id.lower()
        crypto_data = get_crypto_data(coin_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f'Coin not found: {request.coin_id}. Check the CoinGecko ID.',
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        formatted_prompt = build_crypto_analysis_prompt(crypto_data)

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail='ANTHROPIC_API_KEY not set')

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CURRENT_MODEL,
            max_tokens=400,
            system=CRYPTO_SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': formatted_prompt}],
        )

        ai_text = ''
        if response.content and len(response.content) > 0:
            ai_text = response.content[0].text
        else:
            ai_text = 'Claude returned an empty response.'

        usage = getattr(response, 'usage', None)
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        estimated_cost = (input_tokens * COST_PER_INPUT_TOKEN) + (
            output_tokens * COST_PER_OUTPUT_TOKEN
        )

        return {
            'name': crypto_data.get('name'),
            'symbol': crypto_data.get('symbol'),
            'current_price_usd': crypto_data.get('current_price_usd'),
            'market_cap_usd': crypto_data.get('market_cap_usd'),
            'volume_24h': crypto_data.get('volume_24h'),
            'price_change_24h_pct': crypto_data.get('price_change_24h_pct'),
            'price_change_7d_pct': crypto_data.get('price_change_7d_pct'),
            'price_change_30d_pct': crypto_data.get('price_change_30d_pct'),
            'ath': crypto_data.get('ath'),
            'ath_change_pct': crypto_data.get('ath_change_pct'),
            'analysis': ai_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'estimated_cost': round(estimated_cost, 6),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# POST /analyze/compare — Input: JSON with asset1, asset2, asset1_type, asset2_type.
# Fetches and formats data for both assets, calls Claude once with combined prompt and comparison
# system prompt. Output: asset1_name, asset2_name, comparison_analysis, estimated_cost.
@app.post('/analyze/compare')
def analyze_compare_endpoint(request: CompareRequest):
    try:
        prompt1, label1 = _get_prompt_and_label(request.asset1, request.asset1_type)
        prompt2, label2 = _get_prompt_and_label(request.asset2, request.asset2_type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        comparison_instructions = (
            'Compare these two assets across: RELATIVE STRENGTH (which has better momentum), '
            'RISK PROFILE (which is higher risk right now and why), CORRELATION NOTES '
            '(are they likely moving together or independently), and RELATIVE VALUE '
            '(which appears more attractive from a technical standpoint and why). '
            'End with a one-sentence VERDICT.'
        )
        combined_prompt = (
            'You are comparing two financial assets. Analyze both and provide a direct comparison.\n\n'
            f'=== ASSET 1: {label1} ===\n{prompt1}\n\n'
            f'=== ASSET 2: {label2} ===\n{prompt2}\n\n'
            f'{comparison_instructions}\n'
        )

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail='ANTHROPIC_API_KEY not set')

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CURRENT_MODEL,
            max_tokens=400,
            system=COMPARISON_SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': combined_prompt}],
        )

        ai_text = ''
        if response.content and len(response.content) > 0:
            ai_text = response.content[0].text
        else:
            ai_text = 'Claude returned an empty response.'

        usage = getattr(response, 'usage', None)
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        estimated_cost = (input_tokens * COST_PER_INPUT_TOKEN) + (
            output_tokens * COST_PER_OUTPUT_TOKEN
        )

        return {
            'asset1_name': label1,
            'asset2_name': label2,
            'analysis': ai_text,
            'estimated_cost': round(estimated_cost, 6),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# GET /watchlist — Reads watchlist.txt in the app directory. Returns JSON with key 'assets'
# (list of ticker/coin IDs, one per line). If file does not exist, returns empty list.
@app.get('/watchlist')
def get_watchlist():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        watchlist_path = os.path.join(base_dir, 'watchlist.txt')
        if not os.path.exists(watchlist_path):
            return {'assets': []}
        with open(watchlist_path, 'r', encoding='utf-8') as f:
            assets = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith('#')
            ]
        return {'assets': assets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. MOUNT STATIC FILES LAST
# This acts as a "catch-all" for anything in the /static folder
app.mount("/static", StaticFiles(directory=static_path), name="static")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=port)
