# Ensure the app directory is on sys.path so sibling modules (smart_chunker, rag_pipeline, etc.)
# load correctly when run via uvicorn (e.g. on Railway with --app-dir or different cwd).
import sys
from pathlib import Path as _Path
_app_dir = str(_Path(__file__).resolve().parent)
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import shutil
import tempfile
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import func, inspect, text

from database import (
    create_tables,
    get_db,
    engine,
    Analysis,
    WatchlistItem,
    Document,
    QAHistory,
    CollectionGroup,
)
from error_messages import get_user_message, status_code_for_category

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
from rag_pipeline import (
    index_document,
    ask_document,
    ask_across_collections,
    list_collections,
    CHROMA_CLIENT,
)

port = int(os.getenv('PORT', 8000))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from a .env file (the library will search upward
# so this works with a repo-root .env as well as a local one).
load_dotenv()
static_path = os.path.join(BASE_DIR, 'static')

CURRENT_MODEL = "claude-sonnet-4-5"

COST_PER_INPUT_TOKEN = 0.000003
COST_PER_OUTPUT_TOKEN = 0.000015

app = FastAPI(title='Market Analyzer API')

# Ensure database tables exist each time the server starts.
create_tables()

# Migrate existing documents table: add collection_group if missing (e.g. Railway PostgreSQL).
try:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "documents" in tables:
        columns = {col["name"] for col in inspector.get_columns("documents")}
        if "collection_group" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE documents ADD COLUMN collection_group VARCHAR(100) DEFAULT 'default'"
                    )
                )
except Exception as e:
    import traceback
    print("Documents table migration (collection_group):", e)
    traceback.print_exc()

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


class NotesUpdate(BaseModel):
    """Request body model for updating analysis notes."""

    notes: str


class WatchlistAddRequest(BaseModel):
    """Request body model for adding a new watchlist item."""

    ticker: str
    asset_type: str


class QuestionRequest(BaseModel):
    """Request body model for asking a question against a document collection."""

    question: str
    collection_name: str
    n_results: int = 4
    source_file: Optional[str] = None


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
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


# Test endpoint: accepts a ticker path parameter and echoes it back (uppercased).
@app.get('/test/{ticker}')
def test_ticker(ticker: str):
    try:
        return {'ticker': ticker.upper(), 'received': True}
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


# Global stats endpoint: summarizes usage across all saved analyses.
@app.get('/stats')
def get_stats(db: Session = Depends(get_db)):
    """Return aggregate statistics about saved analyses."""

    total_analyses = db.query(Analysis).count()
    stock_analyses = db.query(Analysis).filter(Analysis.asset_type == 'stock').count()
    crypto_analyses = db.query(Analysis).filter(Analysis.asset_type == 'crypto').count()

    unique_tickers = (
        db.query(func.count(func.distinct(Analysis.ticker))).scalar() or 0
    )

    total_cost_raw = db.query(func.sum(Analysis.estimated_cost)).scalar() or 0.0
    total_cost = round(float(total_cost_raw), 4)

    most_row = (
        db.query(Analysis.ticker, func.count(Analysis.id).label('c'))
        .group_by(Analysis.ticker)
        .order_by(func.count(Analysis.id).desc())
        .first()
    )
    most_analyzed = most_row[0] if most_row else None

    return {
        'total_analyses': total_analyses,
        'stock_analyses': stock_analyses,
        'crypto_analyses': crypto_analyses,
        'unique_tickers': unique_tickers,
        'total_cost': total_cost,
        'most_analyzed': most_analyzed,
    }


# POST /analyze/stock — Input: JSON body with ticker. Fetches stock data, computes indicators,
# builds analysis prompt, calls Claude, returns full analysis plus token usage and estimated cost.
# Output: ticker, company_name, current_price, market_cap, pe_ratio, week_high_52, week_low_52,
# rsi, sma_5, sma_20, momentum, support_resistance, ai_analysis, input_tokens, output_tokens, estimated_cost.
@app.post('/analyze/stock')
def analyze_stock_endpoint(
    request: StockRequest,
    db: Session = Depends(get_db),
):
    try:
        ticker = request.ticker.upper()
        stock_data = get_stock_data(ticker)
    except ValueError as e:
        info = get_user_message(e, request.ticker)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )

    try:
        prices = stock_data.get('history') or []
        if not prices:
            info = get_user_message(ValueError('not found'), request.ticker)
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
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
            info = get_user_message(ValueError('api_key not set'))
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
            )

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CURRENT_MODEL,
            max_tokens=300,
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
        cost = (input_tokens * COST_PER_INPUT_TOKEN) + (
            output_tokens * COST_PER_OUTPUT_TOKEN
        )

        # Build the payload that will be returned to the client.
        data = {
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
        }

        result = {
            **data,
            'analysis': ai_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'estimated_cost': round(cost, 7),
        }

        # Try to persist the analysis to the database, but never block the response
        # if something goes wrong with saving.
        try:
            db_analysis = Analysis(
                ticker=request.ticker.upper(),
                asset_type='stock',
                company_name=data.get('company_name'),
                current_price=data.get('current_price'),
                 pe_ratio=data.get('pe_ratio'),
                 week_high_52=data.get('week_high_52'),
                 week_low_52=data.get('week_low_52'),
                rsi=data.get('rsi'),
                sma_5=data.get('sma_5'),
                sma_20=data.get('sma_20'),
                momentum_5d=(data.get('momentum') or {}).get('change_5d'),
                momentum_30d=(data.get('momentum') or {}).get('change_30d'),
                market_cap=data.get('market_cap'),
                ai_analysis=ai_text,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                estimated_cost=cost,
            )
            db.add(db_analysis)
            db.commit()
            db.refresh(db_analysis)
            result['analysis_id'] = db_analysis.id
        except Exception as e:
            # Log the error but continue returning the analysis to the client.
            print(f'Error saving stock analysis for {request.ticker} to database: {e}')

        return result
    except HTTPException:
        raise
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


# POST /analyze/crypto — Input: JSON body with coin_id (CoinGecko ID). Fetches crypto data,
# builds prompt, calls Claude with crypto system prompt. Output: name, symbol, current_price_usd,
# market_cap_usd, volume_24h, price_change_24h/7d/30d_pct, ath, ath_change_pct, ai_analysis, token usage, estimated_cost.
@app.post('/analyze/crypto')
def analyze_crypto_endpoint(
    request: CryptoRequest,
    db: Session = Depends(get_db),
):
    try:
        coin_id = request.coin_id.lower()
        crypto_data = get_crypto_data(coin_id)
    except ValueError as e:
        info = get_user_message(e, request.coin_id)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )

    try:
        formatted_prompt = build_crypto_analysis_prompt(crypto_data)

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            info = get_user_message(ValueError('api_key not set'))
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
            )

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CURRENT_MODEL,
            max_tokens=300,
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
        cost = (input_tokens * COST_PER_INPUT_TOKEN) + (
            output_tokens * COST_PER_OUTPUT_TOKEN
        )

        # Build the payload that will be returned to the client.
        result = {
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
            'estimated_cost': round(cost, 6),
        }

        # Try to persist the crypto analysis to the database; failures should not
        # prevent returning a successful response.
        try:
            db_analysis = Analysis(
                ticker=(crypto_data.get('symbol') or '').upper(),
                asset_type='crypto',
                company_name=crypto_data.get('name'),
                current_price=crypto_data.get('current_price_usd'),
                rsi=None,
                sma_5=None,
                sma_20=None,
                momentum_5d=None,
                momentum_30d=None,
                market_cap=crypto_data.get('market_cap_usd'),
                ai_analysis=ai_text,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                estimated_cost=cost,
            )
            db.add(db_analysis)
            db.commit()
            db.refresh(db_analysis)
            result['analysis_id'] = db_analysis.id
        except Exception as e:
            # Log the error but continue returning the analysis to the client.
            print(
                f"Error saving crypto analysis for {crypto_data.get('symbol')} to database: {e}"
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


# POST /analyze/compare — Input: JSON with asset1, asset2, asset1_type, asset2_type.
# Fetches and formats data for both assets, calls Claude once with combined prompt and comparison
# system prompt. Output: asset1_name, asset2_name, comparison_analysis, estimated_cost.
@app.post('/analyze/compare')
def analyze_compare_endpoint(
    request: CompareRequest,
    db: Session = Depends(get_db),
):
    try:
        prompt1, label1 = _get_prompt_and_label(request.asset1, request.asset1_type)
        prompt2, label2 = _get_prompt_and_label(request.asset2, request.asset2_type)
    except ValueError as e:
        info = get_user_message(e, 'comparison assets')
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )

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
            info = get_user_message(ValueError('api_key not set'))
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
            )

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CURRENT_MODEL,
            max_tokens=300,
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

        result = {
            'asset1_name': label1,
            'asset2_name': label2,
            'analysis': ai_text,
            'estimated_cost': round(estimated_cost, 6),
        }

        # Try to persist the comparison to the database; failures should not block the response.
        try:
            db_analysis = Analysis(
                ticker=f'{request.asset1} vs {request.asset2}',
                asset_type='comparison',
                company_name=f'{label1} vs {label2}',
                ai_analysis=ai_text,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                estimated_cost=estimated_cost,
            )
            db.add(db_analysis)
            db.commit()
            db.refresh(db_analysis)
            result['analysis_id'] = db_analysis.id
        except Exception as e:
            print(f'Error saving comparison to database: {e}')

        return result
    except HTTPException:
        raise
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


# --- RAG document + Q&A endpoints backed by ChromaDB and SQLAlchemy ---


@app.post('/documents/upload')
def upload_document(
    file: UploadFile = File(...),
    collection_group: str = Form('default'),
    db: Session = Depends(get_db),
):
    """Upload a .txt or .pdf file, index it into ChromaDB, and record it in the database."""

    try:
        filename = file.filename or ''
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in {'.txt', '.pdf'}:
            info = get_user_message(ValueError('unsupported file'))
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
            )

        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, filename)

        try:
            with open(tmp_path, 'wb') as out_file:
                shutil.copyfileobj(file.file, out_file)

            index_info = index_document(tmp_path, collection_group=collection_group)
        finally:
            # Best-effort cleanup; failures here are non-fatal.
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if os.path.isdir(tmp_dir):
                    os.rmdir(tmp_dir)
            except OSError:
                pass

        collection_name = index_info.get('collection_name')
        collection_group = index_info.get('collection_group', collection_group)
        chunks_indexed = int(index_info.get('chunks_indexed', 0))
        source_file = index_info.get('source_file') or filename

        # Ensure a CollectionGroup row exists and update its document_count.
        try:
            group = (
                db.query(CollectionGroup)
                .filter(CollectionGroup.name == collection_group)
                .first()
            )
            if not group:
                group = CollectionGroup(
                    name=collection_group,
                    description=None,
                    document_count=0,
                )
                db.add(group)
                db.flush()

            # Upsert document record so re-uploading the same file in this group
            # updates metadata without double-counting in document_count.
            existing = (
                db.query(Document)
                .filter(
                    Document.collection_name == collection_name,
                    Document.filename == source_file,
                )
                .first()
            )
            if existing:
                existing.chunks_indexed = chunks_indexed
                existing.collection_group = collection_group
            else:
                existing = Document(
                    filename=source_file,
                    collection_name=collection_name,
                    collection_group=collection_group,
                    chunks_indexed=chunks_indexed,
                )
                db.add(existing)
                group.document_count = (group.document_count or 0) + 1

            db.commit()
            db.refresh(existing)
            db.refresh(group)
        except Exception as e:
            db.rollback()
            info = get_user_message(e, 'document record')
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
            )

        return {
            'id': existing.id,
            'filename': existing.filename,
            'collection_name': existing.collection_name,
            'collection_group': existing.collection_group,
            'chunks_indexed': existing.chunks_indexed,
            'uploaded_at': existing.uploaded_at.isoformat()
            if existing.uploaded_at
            else None,
            'message': 'Document indexed successfully',
        }
    except HTTPException:
        raise
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


@app.post('/documents/ask')
def ask_document_endpoint(
    payload: QuestionRequest,
    db: Session = Depends(get_db),
):
    """Ask a question against an indexed document collection using the RAG pipeline."""

    try:
        try:
            result = ask_document(
                question=payload.question,
                collection_name=payload.collection_name,
                n_results=payload.n_results,
                source_file=payload.source_file,
            )
        except ValueError as e:
            # Typically thrown if the collection does not exist.
            info = get_user_message(e, payload.collection_name)
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
            )

        # Best-effort persistence of the Q&A interaction.
        try:
            qa = QAHistory(
                collection_name=payload.collection_name,
                question=payload.question,
                answer=result.get('answer', ''),
                found_relevant_context=result.get(
                    'found_relevant_context', False
                ),
                estimated_cost=float(result.get('estimated_cost', 0.0) or 0.0),
            )
            db.add(qa)
            db.commit()
            db.refresh(qa)
        except Exception as e:
            db.rollback()
            print(f'Error saving QA history: {e}')

        return result
    except HTTPException:
        raise
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


@app.get('/documents')
def list_documents(db: Session = Depends(get_db)):
    """Return all indexed documents stored in the database."""

    try:
        docs = db.query(Document).order_by(Document.uploaded_at.desc()).all()
        return [
            {
                'id': d.id,
                'filename': d.filename,
                'collection_name': d.collection_name,
                'collection_group': d.collection_group,
                'chunks_indexed': d.chunks_indexed,
                'uploaded_at': d.uploaded_at.isoformat()
                if d.uploaded_at
                else None,
            }
            for d in docs
        ]
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


@app.get('/documents/{collection_name}/history')
def get_document_history(
    collection_name: str,
    db: Session = Depends(get_db),
):
    """Return the last 20 Q&A pairs for a given collection."""

    try:
        qas = (
            db.query(QAHistory)
            .filter(QAHistory.collection_name == collection_name)
            .order_by(QAHistory.created_at.desc())
            .limit(20)
            .all()
        )

        return [
            {
                'question': qa.question,
                'answer': (qa.answer or '')[:200],
                'found_relevant_context': qa.found_relevant_context,
                'created_at': qa.created_at.isoformat()
                if qa.created_at
                else None,
            }
            for qa in qas
        ]
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


@app.delete('/documents/{collection_name}')
def delete_document(
    collection_name: str,
    db: Session = Depends(get_db),
):
    """Delete a ChromaDB collection and its document record from the database."""

    try:
        # Delete all documents in this ChromaDB collection group from the DB,
        # but do not drop the underlying Chroma collection (use /collections
        # endpoints for group-level deletion).
        docs = (
            db.query(Document)
            .filter(Document.collection_name == collection_name)
            .all()
        )
        removed = len(docs)
        for doc in docs:
            db.delete(doc)
        if removed:
            db.commit()

        return {'deleted': True, 'collection_name': collection_name, 'documents_removed': removed}
    except HTTPException:
        raise
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


# --- Analysis history endpoints backed by the SQLAlchemy database ---


@app.get('/history')
def get_history(
    limit: int = 20,
    offset: int = 0,
    ticker: Optional[str] = None,
    asset_type: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Return recent analyses, optionally filtered by ticker and asset type."""

    query = db.query(Analysis).order_by(Analysis.created_at.desc())

    if ticker:
        query = query.filter(Analysis.ticker == ticker.upper())

    if asset_type:
        query = query.filter(Analysis.asset_type == asset_type.lower())

    if offset < 0:
        offset = 0

    analyses = query.offset(offset).limit(limit).all()

    return [
        {
            'id': a.id,
            'ticker': a.ticker,
            'asset_type': a.asset_type,
            'company_name': a.company_name,
            'current_price': a.current_price,
            'pe_ratio': a.pe_ratio,
            'week_high_52': a.week_high_52,
            'week_low_52': a.week_low_52,
            'rsi': a.rsi,
            'momentum_5d': a.momentum_5d,
            'momentum_30d': a.momentum_30d,
            'estimated_cost': a.estimated_cost,
            'created_at': a.created_at.isoformat() if a.created_at else None,
            'notes': a.notes,
        }
        for a in analyses
    ]


@app.get('/history/{analysis_id}')
def get_history_item(
    analysis_id: int,
    db: Session = Depends(get_db),
):
    """Return the full details for a single saved analysis."""

    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        info = get_user_message(ValueError('not found'), 'analysis')
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )

    return {
        'id': analysis.id,
        'ticker': analysis.ticker,
        'asset_type': analysis.asset_type,
        'company_name': analysis.company_name,
        'current_price': analysis.current_price,
        'pe_ratio': analysis.pe_ratio,
        'week_high_52': analysis.week_high_52,
        'week_low_52': analysis.week_low_52,
        'rsi': analysis.rsi,
        'sma_5': analysis.sma_5,
        'sma_20': analysis.sma_20,
        'momentum_5d': analysis.momentum_5d,
        'momentum_30d': analysis.momentum_30d,
        'market_cap': analysis.market_cap,
        'ai_analysis': analysis.ai_analysis,
        'prompt_tokens': analysis.prompt_tokens,
        'completion_tokens': analysis.completion_tokens,
        'estimated_cost': analysis.estimated_cost,
        'created_at': analysis.created_at.isoformat() if analysis.created_at else None,
        'notes': analysis.notes,
    }


@app.post('/documents/ask-all')
def ask_all_documents(
    payload: dict,
    db: Session = Depends(get_db),
):
    """Ask a question across all collection groups using the RAG pipeline."""

    try:
        question = payload.get('question', '')
        if not isinstance(question, str) or not question.strip():
            info = get_user_message(ValueError('Invalid input'), 'question')
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
            )

        try:
            result = ask_across_collections(question=question)
        except ValueError as e:
            info = get_user_message(e)
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
            )

        # Optionally, you could persist a QAHistory row here with a special
        # collection_name like "__all__" if you want to track these queries.

        return result
    except HTTPException:
        raise
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


@app.get('/collections')
def list_collection_groups(db: Session = Depends(get_db)):
    """Return all collection groups with document counts."""

    try:
        groups = (
            db.query(CollectionGroup)
            .order_by(CollectionGroup.created_at.desc())
            .all()
        )
        return [
            {
                'name': g.name,
                'description': g.description,
                'document_count': g.document_count,
                'created_at': g.created_at.isoformat() if g.created_at else None,
            }
            for g in groups
        ]
    except Exception as e:
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


@app.post('/collections')
def create_collection_group(
    payload: dict,
    db: Session = Depends(get_db),
):
    """Create a new collection group."""

    try:
        name = (payload.get('name') or '').strip()
        description = payload.get('description')

        if not name:
            info = get_user_message(ValueError('Invalid input'), 'name')
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
            )

        existing = (
            db.query(CollectionGroup)
            .filter(CollectionGroup.name == name)
            .first()
        )
        if existing:
            info = get_user_message(ValueError('Invalid input'), 'collection group')
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
            )

        group = CollectionGroup(
            name=name,
            description=description,
            document_count=0,
        )
        db.add(group)
        db.commit()
        db.refresh(group)

        return {
            'name': group.name,
            'description': group.description,
            'document_count': group.document_count,
            'created_at': group.created_at.isoformat() if group.created_at else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


@app.delete('/collections/{group_name}')
def delete_collection_group(
    group_name: str,
    db: Session = Depends(get_db),
):
    """Delete a collection group, its Chroma collection, and associated documents."""

    try:
        group = (
            db.query(CollectionGroup)
            .filter(CollectionGroup.name == group_name)
            .first()
        )
        if not group:
            info = get_user_message(ValueError('not found'), 'collection group')
            raise HTTPException(
                status_code=status_code_for_category(info['category']),
                detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
            )

        # Delete ChromaDB collection for this group.
        try:
            CHROMA_CLIENT.delete_collection(group_name)
        except Exception:
            pass

        # Delete all Document records in this group.
        docs = (
            db.query(Document)
            .filter(Document.collection_group == group_name)
            .all()
        )
        removed = len(docs)
        for doc in docs:
            db.delete(doc)

        # Delete the group itself.
        db.delete(group)
        db.commit()

        return {
            'deleted': True,
            'group_name': group_name,
            'documents_removed': removed,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        info = get_user_message(e)
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )


@app.delete('/history/{analysis_id}')
def delete_history_item(
    analysis_id: int,
    db: Session = Depends(get_db),
):
    """Delete a saved analysis and return confirmation."""

    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        info = get_user_message(ValueError('not found'), 'analysis')
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )

    db.delete(analysis)
    db.commit()

    return {'deleted': True, 'id': analysis_id}


@app.patch('/history/{analysis_id}/notes')
def update_history_notes(
    analysis_id: int,
    payload: NotesUpdate,
    db: Session = Depends(get_db),
):
    """Update the notes field for an existing analysis."""

    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        info = get_user_message(ValueError('not found'), 'analysis')
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )

    analysis.notes = payload.notes
    db.commit()
    db.refresh(analysis)

    return {
        'id': analysis.id,
        'ticker': analysis.ticker,
        'asset_type': analysis.asset_type,
        'company_name': analysis.company_name,
        'current_price': analysis.current_price,
        'rsi': analysis.rsi,
        'sma_5': analysis.sma_5,
        'sma_20': analysis.sma_20,
        'momentum_5d': analysis.momentum_5d,
        'momentum_30d': analysis.momentum_30d,
        'market_cap': analysis.market_cap,
        'ai_analysis': analysis.ai_analysis,
        'prompt_tokens': analysis.prompt_tokens,
        'completion_tokens': analysis.completion_tokens,
        'estimated_cost': analysis.estimated_cost,
        'created_at': analysis.created_at.isoformat() if analysis.created_at else None,
        'notes': analysis.notes,
    }


# --- Watchlist management endpoints backed by the WatchlistItem table ---


@app.post('/watchlist/add')
def add_watchlist_item(
    payload: WatchlistAddRequest,
    db: Session = Depends(get_db),
):
    """Add a new ticker to the watchlist if it is not already present."""

    ticker = payload.ticker.upper().strip()
    asset_type = payload.asset_type.lower().strip()

    # Check if this ticker already exists in the watchlist.
    existing = db.query(WatchlistItem).filter(WatchlistItem.ticker == ticker).first()
    if existing:
        # If it's already active, report that it exists.
        if existing.is_active:
            return {'already_exists': True, 'ticker': ticker}
        # If it was soft-deleted, reactivate and reuse the same row.
        existing.is_active = True
        existing.asset_type = asset_type
        db.commit()
        db.refresh(existing)
        return {'added': True, 'ticker': existing.ticker, 'id': existing.id}

    # Create and persist a new watchlist item.
    item = WatchlistItem(ticker=ticker, asset_type=asset_type)
    db.add(item)
    db.commit()
    db.refresh(item)

    return {'added': True, 'ticker': item.ticker, 'id': item.id}


@app.get('/watchlist')
def get_watchlist(
    db: Session = Depends(get_db),
):
    """Return all active watchlist items (soft-deleted items are omitted)."""

    items = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.is_active.is_(True))
        .order_by(WatchlistItem.added_at.desc())
        .all()
    )

    return [
        {
            'id': item.id,
            'ticker': item.ticker,
            'asset_type': item.asset_type,
            'added_at': item.added_at.isoformat() if item.added_at else None,
        }
        for item in items
    ]


@app.delete('/watchlist/{item_id}')
def remove_watchlist_item(
    item_id: int,
    db: Session = Depends(get_db),
):
    """Soft delete a watchlist item by marking it inactive."""

    item = db.query(WatchlistItem).filter(WatchlistItem.id == item_id).first()
    if not item:
        info = get_user_message(ValueError('not found'), 'watchlist item')
        raise HTTPException(
            status_code=status_code_for_category(info['category']),
            detail={'user_message': info['user_message'], 'suggestion': info['suggestion'], 'category': info['category'].value},
        )

    item.is_active = False
    db.commit()
    db.refresh(item)

    return {'removed': True, 'ticker': item.ticker}

# 2. MOUNT STATIC FILES LAST
# This acts as a "catch-all" for anything in the /static folder
app.mount("/static", StaticFiles(directory=static_path), name="static")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=port)
