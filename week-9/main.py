from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from agent import run_agent
from pydantic import BaseModel
from typing import Optional, List, Any

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
async def agent_chat(request: AgentRequest) -> AgentResponse:
    try:
        result = run_agent(
            request.message,
            request.conversation_history,
            request.collection_name,
        )
        print(f'Agent query: {request.message[:50]}')
        print(f'Tools used: {[t["tool"] for t in result["tools_used"]]}')
        return AgentResponse(
            answer=result['answer'],
            tools_used=result['tools_used'],
            iterations=result['iterations'],
            total_tokens=result['total_tokens'],
            conversation_history=result['conversation_history'],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
