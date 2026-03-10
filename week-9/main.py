from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from agent import run_agent, run_agent_stream, MODEL
from pydantic import BaseModel
from typing import Optional, List, Any
from rate_limiter import check_rate_limit, get_rate_limit_status
from tools import TOOLS

load_dotenv()

app = FastAPI(title='Financial Research Agent')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static', StaticFiles(directory='static'), name='static')

@app.get('/')
async def root():
    return FileResponse('static/index.html')

@app.get('/health')
async def health():
    return {'status': 'ok', 'service': 'Financial Research Agent'}


def get_client_identifier(request: Request) -> str:
    """Return a stable identifier for rate limiting.

    In a fully authenticated setup this would prefer the current user's ID.
    For Week 9 where auth is optional, we use the client IP address.
    """
    client_host = request.client.host if request.client else 'unknown'
    return client_host

class AgentRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Any]] = None
    collection_name: Optional[str] = 'default'


class AgentResponse(BaseModel):
    answer: str
    tools_used: List[Any]
    iterations: int
    total_tokens: int
    conversation_history: List[Any]


@app.post('/agent/chat', response_model=AgentResponse)
async def agent_chat(request: Request, payload: AgentRequest):
    try:
        # HARDENING: reject empty or overly long messages early.
        if not payload.message or not payload.message.strip():
            raise HTTPException(status_code=400, detail='Message cannot be empty')
        if len(payload.message) > 2000:
            raise HTTPException(
                status_code=400,
                detail='Message too long. Maximum 2000 characters.',
            )

        identifier = get_client_identifier(request)
        check_rate_limit(identifier)

        result = run_agent(
            payload.message,
            payload.conversation_history,
            payload.collection_name,
        )
        print(f'Agent query: {payload.message[:50]}')
        print(f'Tools used: {[t["tool"] for t in result["tools_used"]]}')
        status = get_rate_limit_status(identifier)
        headers = {
            'X-RateLimit-Limit': str(status['limit']),
            'X-RateLimit-Remaining': str(status['requests_remaining']),
        }
        content = {
            'answer': result['answer'],
            'tools_used': result['tools_used'],
            'iterations': result['iterations'],
            'total_tokens': result['total_tokens'],
            'conversation_history': result['conversation_history'],
        }
        return JSONResponse(content=content, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/agent/stream')
async def agent_stream(
    request: Request,
    message: str,
    collection_name: str = 'default',
    conversation_history: Optional[str] = None,
):
    """
    Streaming SSE endpoint for the Financial Research Agent.

    Uses a stateless conversation for now (no server-side history), while
    /agent/chat continues to handle full conversational history.
    """

    identifier = get_client_identifier(request)
    check_rate_limit(identifier)

    history_obj: Any = []
    if conversation_history:
        try:
            history_obj = json.loads(conversation_history)
        except Exception:
            history_obj = []

    def generate():
        # Pass along prior conversation history so the agent can answer
        # follow-up questions with full context.
        yield from run_agent_stream(message, history_obj, collection_name)

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # disable nginx buffering on Railway
        },
    )


@app.get('/agent/rate-limit-status')
async def agent_rate_limit_status(request: Request):
    identifier = get_client_identifier(request)
    return get_rate_limit_status(identifier)


@app.get('/agent/health')
async def agent_health():
    """Lightweight healthcheck for the agent and its tools."""
    try:
        # Touch tool and model definitions so import errors surface here.
        _ = [t["name"] for t in TOOLS]
        model_name = MODEL
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Agent health failed: {e}')

    return {
        'status': 'ok',
        'tools_available': [
            'get_stock_data',
            'get_crypto_data',
            'search_documents',
            'compare_assets',
        ],
        'model': model_name,
    }
