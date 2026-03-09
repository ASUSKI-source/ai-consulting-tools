# Financial Research Agent

## What it does
Ask a financial question in plain English — about a stock, a crypto asset, or something buried in your uploaded documents — and the agent will fetch live market data, search your research files, and synthesize a concise, data-driven answer in seconds, complete with the numbers and context you need to decide what to do next.

## How it works
- You type a question into the chat and hit **Ask**.  
- The agent decides whether to pull live stock or crypto data, search your documents, or compare multiple assets.  
- It runs those lookups, then turns the raw numbers and text into a clear written explanation.  
- The response appears in the chat along with a small badge showing which tools were actually used.  

## Technical decisions
- **Problem**: Many orchestration frameworks add complexity for a single focused agent. **Solution**: Call Anthropic’s tool-use API directly and route tool calls in a small Python function. **Result**: Less glue code, easier debugging, and behavior that matches exactly what’s shown in the portfolio.  
- **Problem**: Users can’t see how much of the answer comes from live data versus prior knowledge. **Solution**: Show a tools-used badge under each agent reply listing the tools it called. **Result**: More trust in the system and easier debugging when something looks off.  
- **Problem**: Long conversations can be expensive to re-send in full from the backend. **Solution**: Store conversation history in the browser and send it back with each chat request. **Result**: Simple, stateless API endpoints that are easy to host and scale for this demo use case.  

## Stack
FastAPI · Anthropic Claude · Voyage AI embeddings · ChromaDB · PostgreSQL · Railway

## Live demo
[Your Railway URL]

