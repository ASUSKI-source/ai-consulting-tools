"""Tool definitions and executor for the Financial Research Agent.

This module defines the available tools the agent can call (for stocks, crypto,
and document search) and provides a single executor function that routes tool
invocations to the appropriate Python implementation.
"""

from stock_data import get_stock_data
from crypto_data import get_crypto_data
from indicators import calculate_rsi, calculate_sma
from rag_pipeline import search_documents
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError


_EXECUTOR = ThreadPoolExecutor(max_workers=8)


def _run_with_timeout(fn, *args, timeout: float = 10.0):
    """Run a blocking function in a worker thread with a timeout.

    Returns (result, error_dict). On timeout, result is None and error_dict is
    a JSON-serializable error payload.
    """
    future = _EXECUTOR.submit(fn, *args)
    try:
        return future.result(timeout=timeout), None
    except TimeoutError:
        return None, {
            "error": True,
            "message": (
                "Data fetch timed out. Market data provider may be slow. "
                "Try again."
            ),
        }


# Descriptions here are critical because Claude reads them to decide
# when and how to use each tool.
TOOLS = [
    {
        "name": "get_stock_data",
        "description": (
            "Fetch live stock data for a ticker symbol. Returns current price, "
            "RSI, SMA-5, SMA-20, momentum, market cap, PE ratio, 52-week "
            "high/low. Use this when the user asks about a stock price, "
            "technical indicators, valuation, or wants to know how a stock "
            "is performing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": (
                        "Stock ticker symbol e.g. AAPL, MSFT, TSLA"
                    ),
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_crypto_data",
        "description": (
            "Fetch live cryptocurrency data. Returns price, RSI, momentum, "
            "market cap, volume. Use this when the user asks about Bitcoin, "
            "Ethereum, or any cryptocurrency. Accepts ticker symbols like "
            "BTC-USD, ETH-USD."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": (
                        "Crypto ticker e.g. BTC-USD, ETH-USD, SOL-USD"
                    ),
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "search_documents",
        "description": (
            "Search uploaded research documents for relevant information. "
            "Returns the most relevant text passages from the document "
            "collection. Use this when the user asks about anything that "
            "might be in their uploaded documents — earnings reports, "
            "research notes, company filings, news articles. Always try this "
            "tool when the user asks about specific companies or events that "
            "may have been uploaded."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "collection_name": {
                    "type": "string",
                    "description": (
                        "Name of the document collection to search. "
                        "Default: default"
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "compare_assets",
        "description": (
            "Compare two financial assets side by side. Fetches data for both "
            "and returns a structured comparison of price, RSI, momentum, and "
            "valuation. Use this when the user asks to compare two stocks or "
            "assets, or asks which of two assets is performing better."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "asset1": {
                    "type": "string",
                    "description": "First ticker symbol",
                },
                "asset2": {
                    "type": "string",
                    "description": "Second ticker symbol",
                },
                "asset1_type": {
                    "type": "string",
                    "enum": ["stock", "crypto"],
                    "description": "Type of first asset",
                },
                "asset2_type": {
                    "type": "string",
                    "enum": ["stock", "crypto"],
                    "description": "Type of second asset",
                },
            },
            "required": ["asset1", "asset2"],
        },
    },
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Route tool calls from the agent to concrete Python implementations."""
    try:
        if tool_name == "get_stock_data":
            result, timeout_err = _run_with_timeout(
                get_stock_data, tool_input["ticker"], timeout=10.0
            )
            if timeout_err:
                return json.dumps(timeout_err)
            return json.dumps(result)
        elif tool_name == "get_crypto_data":
            result, timeout_err = _run_with_timeout(
                get_crypto_data, tool_input["ticker"], timeout=10.0
            )
            if timeout_err:
                return json.dumps(timeout_err)
            return json.dumps(result)
        elif tool_name == "search_documents":
            collection = tool_input.get("collection_name", "default")
            results = search_documents(tool_input["query"], collection)
            if not results:
                return json.dumps(
                    {
                        "results": [],
                        "message": (
                            "No documents found in this collection. "
                            "Upload a document first via POST /documents/upload."
                        ),
                    }
                )
            return json.dumps(results[:3])
        elif tool_name == "compare_assets":
            a1_type = tool_input.get("asset1_type", "stock")
            a2_type = tool_input.get("asset2_type", "stock")
            fn1 = get_crypto_data if a1_type == "crypto" else get_stock_data
            fn2 = get_crypto_data if a2_type == "crypto" else get_stock_data
            data1, timeout_err1 = _run_with_timeout(
                fn1, tool_input["asset1"], timeout=10.0
            )
            if timeout_err1:
                return json.dumps(timeout_err1)
            data2, timeout_err2 = _run_with_timeout(
                fn2, tool_input["asset2"], timeout=10.0
            )
            if timeout_err2:
                return json.dumps(timeout_err2)
            return json.dumps({"asset1": data1, "asset2": data2})
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        # Return error as JSON string so Claude can read it and respond gracefully.
        return json.dumps(
            {
                "error": True,
                "tool": tool_name,
                "message": str(e),
                "suggestion": (
                    "The tool encountered an error. Answer based on general "
                    "knowledge where possible."
                ),
            }
        )

