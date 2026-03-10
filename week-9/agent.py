import anthropic
import os
import json
import time
from typing import List, Dict, Any, Optional, Generator

from tools import TOOLS, execute_tool


# Create a shared Anthropic client and core configuration for the agent.
ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5"
MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are a financial research assistant with access to live
market data, technical indicators, asset comparison tools, and a document search
system for uploaded research materials.

When answering financial questions:
- Always fetch live data before making claims about prices or indicators
- Search documents when the question may involve uploaded research or news
- Compare assets when the user asks which is better or wants a side-by-side view
- Be specific: cite the actual numbers you retrieved, not general knowledge
- If a tool returns an error, acknowledge it and answer with what you have

Keep answers concise but data-driven. Lead with the key finding.
"""


def _serialize_block(block: Any) -> Dict[str, Any]:
    """Convert Anthropic SDK blocks or plain dicts into API-safe content blocks."""
    # SDK-style block objects
    if not isinstance(block, dict) and hasattr(block, "type"):
        btype = getattr(block, "type", None)
        if btype == "text":
            return {"type": "text", "text": getattr(block, "text", "") or ""}
        if btype == "tool_use":
            return {
                "type": "tool_use",
                "id": getattr(block, "id", None),
                "name": getattr(block, "name", None),
                "input": getattr(block, "input", {}) or {},
            }
        if btype == "tool_result":
            return {
                "type": "tool_result",
                "tool_use_id": getattr(block, "tool_use_id", None),
                "content": getattr(block, "content", None),
            }
        # Fallback: treat as text
        return {"type": "text", "text": str(getattr(block, "text", "") or "")}

    # Dict-style blocks coming back from the frontend
    if isinstance(block, dict):
        btype = block.get("type")
        if btype == "text":
            text_val = block.get("text", "")
            # Ensure text is always a plain string, never a nested object
            if not isinstance(text_val, str):
                text_val = str(text_val)
            return {"type": "text", "text": text_val}
        if btype == "tool_use":
            return {
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": block.get("input") or {},
            }
        if btype == "tool_result":
            return {
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id"),
                "content": block.get("content"),
            }
        # Fallback: if there's a text field, use it; otherwise stringify
        if "text" in block:
            txt = block.get("text", "")
            if not isinstance(txt, str):
                txt = str(txt)
            return {"type": "text", "text": txt}
        return {"type": "text", "text": str(block)}

    # Any other primitive – coerce to text
    return {"type": "text", "text": str(block)}


def _normalize_history(raw_history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Turn wire-format conversation_history into valid Anthropic message objects."""
    normalized: List[Dict[str, Any]] = []
    for msg in raw_history or []:
        role = msg.get("role")
        content = msg.get("content")
        if not role:
            continue

        blocks: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for block in content:
                blocks.append(_serialize_block(block))
        elif content is not None:
            # Older history may have content as a raw string
            blocks.append({"type": "text", "text": str(content)})

        if blocks:
            normalized.append({"role": role, "content": blocks})
    return normalized


def run_agent(
    user_message: str,
    conversation_history: Optional[List[Dict]] = None,
    collection_name: str = "default",
) -> Dict[str, Any]:
    """Run the agent loop for one user message.

    Args:
        user_message: The user's question.
        conversation_history: Previous messages for multi-turn conversations.
        collection_name: ChromaDB collection to search for documents.

    Returns:
        Dict with: answer (str), tools_used (list), iterations (int),
        total_tokens (int), conversation_history (updated list).
    """

    # Initialize the message list with any prior history so the model can
    # maintain conversational context across turns, then add the latest user
    # question as the final message in the sequence. We normalize any history
    # coming back from the frontend to ensure it matches Anthropic's expected
    # Messages API schema.
    messages: List[Dict[str, Any]] = _normalize_history(conversation_history)
    messages.append({"role": "user", "content": [{"type": "text", "text": user_message}]})

    # Track which tools were called (and with what inputs) so we can later
    # inspect or log how the agent reasoned; also track iteration count and
    # approximate token usage for basic cost / performance monitoring.
    tools_used: List[Dict[str, Any]] = []
    iterations = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # THE AGENT LOOP:
    # Repeatedly call Claude, allowing it to either respond directly to the
    # user or request tool calls. After each tool call, we feed the results
    # back into the conversation and let Claude decide whether more tools are
    # needed, up to MAX_ITERATIONS to avoid runaway loops.
    while iterations < MAX_ITERATIONS:
        iterations += 1

        # Call the Anthropic Messages API with the current conversation context
        # and the full tool specification. Claude can now choose to answer
        # directly or emit tool_use blocks describing which tools to call.
        response = ANTHROPIC_CLIENT.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Accumulate token usage from each iteration so we have a simple view
        # of how expensive the end-to-end interaction was.
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Record Claude's raw response (including any tool_use blocks) in the
        # conversation history so that subsequent iterations can see what it
        # requested and what it has already said.
        messages.append({"role": "assistant", "content": response.content})

        # CHECK STOP REASON:
        # The stop_reason tells us whether Claude is done ("end_turn"),
        # requesting tools ("tool_use"), or something unexpected happened.
        if response.stop_reason == "end_turn":
            # Claude finished — extract the final natural language answer from
            # any text blocks in the content and break out of the loop.
            answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    answer += block.text
            break

        elif response.stop_reason == "tool_use":
            # Claude has emitted one or more tool_use blocks. We synchronously
            # execute each requested tool, collect their results, and then add
            # those results back into the conversation so Claude can read them
            # and decide what to do next.
            tool_results: List[Dict[str, Any]] = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tools_used.append({"tool": tool_name, "input": tool_input})

                    # Invoke the appropriate Python implementation for the tool
                    # and serialize its result as JSON so it can be fed back to
                    # Claude in a structured but model-friendly format.
                    result = execute_tool(tool_name, tool_input)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            # Present all tool outputs to Claude as if they were a new user
            # message consisting of tool_result blocks. Claude will read these
            # results on the next iteration and either call more tools or
            # produce a final answer.
            messages.append({"role": "user", "content": tool_results})

        else:
            # Any other stop_reason is treated as an unexpected termination.
            # We guard against infinite loops or API edge cases by breaking
            # and returning a generic message rather than hanging indefinitely.
            answer = "Agent stopped unexpectedly."
            break

    return {
        "answer": answer,
        "tools_used": tools_used,
        "iterations": iterations,
        "total_tokens": total_input_tokens + total_output_tokens,
        # Convert conversation history into a JSON-serializable, API-safe form
        # that can be sent back on the next turn without validation errors.
        "conversation_history": [
            {
                "role": msg.get("role"),
                "content": [
                    _serialize_block(block) for block in msg.get("content", [])
                ],
            }
            for msg in messages
        ],
    }


def run_agent_stream(
    user_message: str,
    conversation_history: list = None,
    collection_name: str = "default",
) -> Generator[str, None, None]:
    """Streaming version of run_agent.

    Yields SSE-formatted strings. Tool calls are executed synchronously and
    only the final text answer from Claude is streamed back to the caller.
    """

    # Start from any prior conversation history (if provided) so the model can
    # maintain context across turns, then append the latest user message as the
    # next step in the dialogue.
    messages = list(conversation_history or [])
    messages.append({"role": "user", "content": user_message})

    # Track which tools were invoked (and with what inputs) as well as loop
    # iterations and approximate token usage for monitoring / debugging.
    tools_used = []
    iterations = 0
    total_tokens = 0

    # TOOL CALL PHASE:
    # Repeatedly call Claude in non-streaming mode so it can decide whether to
    # issue tool_use requests. For each requested tool, we synchronously execute
    # the corresponding Python implementation and stream structured SSE events
    # describing both the call and its result.
    while iterations < MAX_ITERATIONS:
        iterations += 1

        # Make a standard (non-streaming) Messages API call so Claude can plan
        # tool usage. This mirrors the behavior of run_agent, but instead of
        # returning a final answer directly we watch for tool_use and end_turn
        # stop reasons to drive the streaming protocol.
        response = ANTHROPIC_CLIENT.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        # If Claude requests tools, execute each one, emitting SSE events before
        # and after the call so the client can observe the agent's reasoning
        # process and intermediate tool outputs.
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tools_used.append({"tool": block.name, "input": block.input})
                    # Yield a tool_call event before executing the tool so the
                    # client can, for example, show a "calling tool" indicator.
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': block.name, 'input': block.input})}\n\n"

                    start_time = time.time()
                    result = execute_tool(block.name, block.input)
                    duration_ms = int((time.time() - start_time) * 1000)

                    # Yield a tool_result event containing a short preview of
                    # the tool output to keep SSE payloads compact, along with
                    # the measured execution time in milliseconds.
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': block.name, 'preview': str(result)[:100], 'duration_ms': duration_ms})}\n\n"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            # Feed all tool_result blocks back to Claude as if they were a user
            # message so it can read the outputs and decide what to do next.
            messages.append({"role": "user", "content": tool_results})

        # When Claude signals end_turn, it is ready to produce a final natural
        # language answer. At this point we reissue the request using the
        # streaming API and forward text deltas as SSE events to the caller.
        elif response.stop_reason == "end_turn":
            # Prepare a fresh messages list for streaming by copying the full
            # history and removing the most recent assistant message (the
            # non-streamed response we just received).
            stream_messages = messages.copy()
            stream_messages.pop()
            messages.pop()

            # Stream the final answer from Claude, yielding a text_delta event
            # for each chunk so that the frontend can render the response
            # incrementally as it arrives.
            with ANTHROPIC_CLIENT.messages.stream(
                model=MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=stream_messages,
            ) as stream:
                full_text = ""
                for text_chunk in stream.text_stream:
                    full_text += text_chunk
                    yield f"data: {json.dumps({'type': 'text_delta', 'text': text_chunk})}\n\n"

                # After streaming completes, append the fully assembled message
                # to the conversation history so it can be reused on later turns.
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": full_text}],
                    }
                )

            # Finally, emit a terminal done event with metadata about the run,
            # including the tools used, aggregate token usage, and the updated
            # conversation history for the next turn.
            yield f"data: {json.dumps({'type': 'done', 'tools_used': tools_used, 'total_tokens': total_tokens, 'conversation_history': messages})}\n\n"
            return

        # Any other stop_reason is unexpected in this loop; surface an error
        # payload to the client and terminate the generator.
        else:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Unexpected stop reason'})}\n\n"
            return