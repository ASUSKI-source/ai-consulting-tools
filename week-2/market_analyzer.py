"""
Market Analyzer: combines stock/crypto data, indicators, formatted prompts, and Claude AI.
"""
import os
import sys
import time
from datetime import datetime

import anthropic
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

load_dotenv()

# Approximate per-token costs (USD).
COST_PER_INPUT_TOKEN = 0.000003   # $3 per million input tokens
COST_PER_OUTPUT_TOKEN = 0.000015  # $15 per million output tokens

# Running session totals.
session_cost: float = 0.0
session_analyses: int = 0

SYSTEM_PROMPT = (
    "You are a professional stock market analyst with 20 years of experience. "
    "You analyze technical and fundamental data to provide clear, actionable insights. "
    "Your analysis is direct and specific — no generic disclaimers. "
    "Always cite specific numbers from the data provided (price, RSI, SMA values). "
    "Structure your response with these sections: TREND ASSESSMENT, KEY OBSERVATIONS, "
    "RISK FACTORS, BOTTOM LINE. "
    "In TREND ASSESSMENT, clearly state whether price is above or below SMA20 and by how much (in percent). "
    "In KEY OBSERVATIONS, mention the exact RSI reading and what it implies for the next likely move. "
    "In RISK FACTORS, reference the specific distance from the 52-week high and 52-week low using percentages. "
    "In BOTTOM LINE, give a clear directional bias (bullish, bearish, or neutral) and call out the ONE most important "
    "number that supports that view. "
    "Never use vague phrases like 'could potentially' or 'may possibly' — use direct language instead. "
    "Format dollar amounts consistently: $XXX.XX for values under $1,000 and $X.XXK for thousands where appropriate. "
    "Keep total response under 400 words."
)

CRYPTO_SYSTEM_PROMPT = (
    "You are a cryptocurrency market analyst. Analyze the provided data and give a direct assessment. "
    "Always state the all-time-high (ATH) drawdown percentage prominently and interpret what it means for risk/reward. "
    "Clearly classify the current trend as bull, bear, or ranging based on price direction and 30-day momentum, "
    "and whether price behavior is consistent with strength or weakness. "
    "Cover these sections: MARKET STRUCTURE (bull/bear/ranging), MOMENTUM SIGNALS, RISK ASSESSMENT, BOTTOM LINE. "
    "In MARKET STRUCTURE, compare current price behavior to recent performance and clearly label the trend. "
    "In MOMENTUM SIGNALS, discuss 24h, 7d, and 30d price changes and how they support or contradict the trend. "
    "In RISK ASSESSMENT, always mention the 30-day volatility context and the size of the ATH drawdown. "
    "In BOTTOM LINE, state a directional view with a conviction level of HIGH, MEDIUM, or LOW and explain which "
    "specific numbers (drawdown, recent changes, market cap) drive that conviction. "
    "Be specific and avoid vague language. Keep the total response under 350 words."
)


def _format_currency(value, decimals: int = 2) -> str:
    """Format a numeric value as a dollar string with commas."""
    if value is None:
        return "N/A"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "N/A"
    fmt = f"{{:,.{decimals}f}}"
    return fmt.format(num)


def _format_percent(value, decimals: int = 2) -> str:
    """Format a numeric value as a percentage string."""
    if value is None:
        return "N/A"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "N/A"
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(num)}%"


def _crypto_cap_category(market_cap_usd) -> str:
    """Classify crypto by market cap size."""
    if market_cap_usd is None:
        return "Unknown"
    try:
        cap = float(market_cap_usd)
    except (TypeError, ValueError):
        return "Unknown"

    if cap > 100_000_000_000:
        return "Large Cap"
    if 10_000_000_000 <= cap <= 100_000_000_000:
        return "Mid Cap"
    return "Small Cap"


def build_stock_prompt_for_ticker(ticker: str) -> tuple[str, str]:
    """
    Fetch stock data and compute indicators, returning a formatted prompt
    and a clean uppercase ticker label.
    """
    ticker_upper = ticker.upper().strip()

    stock_data = get_stock_data(ticker_upper)
    prices = stock_data.get("history") or []

    if not prices:
        raise ValueError(f"No price history available for {ticker_upper}.")

    sma_5 = calculate_sma(prices, 5)
    sma_20 = calculate_sma(prices, 20)
    rsi = calculate_rsi(prices)
    momentum = calculate_price_momentum(prices)
    levels = find_support_resistance(prices)

    indicators = {
        "sma_5": sma_5,
        "sma_20": sma_20,
        "rsi": rsi,
        "momentum": momentum,
        "support_resistance": levels,
    }
    formatted_prompt = build_stock_analysis_prompt(stock_data, indicators)
    return formatted_prompt, ticker_upper


def build_crypto_prompt_for_coin(coin_id: str) -> tuple[str, str]:
    """
    Fetch crypto data and return a formatted prompt plus a human-readable label.
    """
    coin_id_clean = coin_id.strip()
    crypto_data = get_crypto_data(coin_id_clean)
    formatted_prompt = build_crypto_analysis_prompt(crypto_data)
    label = f"{crypto_data.get('name') or coin_id_clean} ({(crypto_data.get('symbol') or '').upper()})"
    return formatted_prompt, label


def build_crypto_analysis_prompt(crypto_data: dict) -> str:
    """
    Build a structured multi-line prompt for crypto analysis.

    Expects the dict returned from get_crypto_data().
    """
    name = crypto_data.get("name") or "Unknown"
    symbol = (crypto_data.get("symbol") or "").upper()
    current_price = crypto_data.get("current_price_usd")
    market_cap = crypto_data.get("market_cap_usd")
    volume_24h = crypto_data.get("volume_24h")
    change_24h = crypto_data.get("price_change_24h_pct")
    change_7d = crypto_data.get("price_change_7d_pct")
    change_30d = crypto_data.get("price_change_30d_pct")
    ath = crypto_data.get("ath")
    ath_change_pct = crypto_data.get("ath_change_pct")

    dominance = _crypto_cap_category(market_cap)

    # Distance from ATH: use ath_change_pct (typically negative = below ATH)
    if ath_change_pct is None:
        distance_from_ath = "N/A"
    else:
        try:
            pct = float(ath_change_pct)
            direction = "below" if pct < 0 else "above"
            distance_from_ath = f"{abs(pct):.2f}% {direction} ATH"
        except (TypeError, ValueError):
            distance_from_ath = "N/A"

    lines = [
        "=== CRYPTO ANALYSIS REQUEST ===",
        f"Name: {name} ({symbol})",
        f"Current Price: ${_format_currency(current_price)}",
        f"Market Cap: ${_format_currency(market_cap)} ({dominance})",
        f"24h Volume: ${_format_currency(volume_24h)}",
        "",
        "--- PRICE PERFORMANCE ---",
        f"24h Change: {_format_percent(change_24h)}",
        f"7d Change:  {_format_percent(change_7d)}",
        f"30d Change: {_format_percent(change_30d)}",
        "",
        "--- POSITION VS ATH ---",
        f"All-Time High: ${_format_currency(ath)}",
        f"Distance from ATH: {distance_from_ath}",
        "",
        f"Dominance Note: {dominance}",
        "================================",
    ]

    return "\n".join(lines)


def analyze_stock(ticker: str) -> None:
    """
    Fetch stock data, compute indicators, build a prompt, and get AI analysis.
    Save results to a timestamped file.
    """
    # 1–4. Build formatted prompt and clean ticker label
    try:
        formatted_prompt, ticker_upper = build_stock_prompt_for_ticker(ticker)
    except Exception as e:
        print(f"Error: could not prepare stock data for {ticker}. {e}")
        return

    # 5. Set up the Anthropic client using API key from .env
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in .env")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # 6. Call Claude with the specified model and prompt
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": formatted_prompt},
            ],
        )
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return

    ai_text = ""
    if response.content and len(response.content) > 0:
        ai_text = response.content[0].text
    else:
        ai_text = "Claude returned an empty response."

    # 7. Print separator, header, and AI response
    print("\n" + "=" * 60)
    print(f"AI ANALYSIS: {ticker_upper}")
    print("=" * 60)
    print(ai_text)

    # Cost tracking
    call_cost = _record_cost(getattr(response, "usage", None))
    print(
        f"This analysis cost: ${call_cost:.5f} | "
        f"Session total: ${session_cost:.5f}"
    )

    # 8. Save the full analysis (data + AI response) to a timestamped file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{ticker_upper}_{timestamp}.txt"

    full_content = (
        f"=== STOCK DATA (input to AI) ===\n\n"
        f"{formatted_prompt}\n\n"
        f"{'=' * 60}\n"
        f"AI ANALYSIS: {ticker_upper}\n"
        f"{'=' * 60}\n\n"
        f"{ai_text}\n"
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"\nAnalysis saved to {filename}")


def analyze_crypto(coin_id: str) -> None:
    """
    Fetch crypto data, build a prompt, and get AI analysis.
    Save results to a timestamped file.
    """
    coin_id_clean = coin_id.strip()

    # 1–4. Build formatted prompt using crypto data
    try:
        formatted_prompt, coin_label = build_crypto_prompt_for_coin(coin_id_clean)
    except Exception as e:
        print(f"Error: could not prepare crypto data for '{coin_id_clean}'. {e}")
        return

    # 5. Set up the Anthropic client using API key from .env
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in .env")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # 6. Call Claude with the crypto-specific system prompt
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=CRYPTO_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": formatted_prompt},
            ],
        )
    except Exception as e:
        print(f"Error calling Claude for crypto: {e}")
        return

    ai_text = ""
    if response.content and len(response.content) > 0:
        ai_text = response.content[0].text
    else:
        ai_text = "Claude returned an empty response."

    # 7. Print separator, header, and AI response
    print("\n" + "=" * 60)
    print(f"AI CRYPTO ANALYSIS: {coin_label}")
    print("=" * 60)
    print(ai_text)

    # Cost tracking
    call_cost = _record_cost(getattr(response, "usage", None))
    print(
        f"This analysis cost: ${call_cost:.5f} | "
        f"Session total: ${session_cost:.5f}"
    )

    # 8. Save the full analysis (data + AI response) to a timestamped file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_crypto_{coin_id_clean}_{timestamp}.txt"

    full_content = (
        f"=== CRYPTO DATA (input to AI) ===\n\n"
        f"{formatted_prompt}\n\n"
        f"{'=' * 60}\n"
        f"AI CRYPTO ANALYSIS: {coin_label}\n"
        f"{'=' * 60}\n\n"
        f"{ai_text}\n"
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"\nCrypto analysis saved to {filename}")


COMPARISON_SYSTEM_PROMPT = (
    "You are a cross-asset market analyst comparing two stocks and/or cryptocurrencies. "
    "Use specific numbers from the provided data blocks in your reasoning. "
    "Be concise, analytical, and avoid vague language."
)


def _record_cost(usage) -> float:
    """
    Update global session cost/analysis counters from a usage object.
    Returns the cost of this call.
    """
    global session_cost, session_analyses

    session_analyses += 1
    if usage is None:
        return 0.0

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    call_cost = (
        input_tokens * COST_PER_INPUT_TOKEN
        + output_tokens * COST_PER_OUTPUT_TOKEN
    )
    session_cost += call_cost
    return call_cost


def _sanitize_label_for_filename(label: str) -> str:
    """Sanitize a label for safe use in filenames."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in label)
    return safe.strip("_") or "asset"


def compare_assets(asset1: str, asset2: str) -> None:
    """
    Compare two assets (stocks or crypto) using their formatted prompts.
    """
    asset1 = asset1.strip()
    asset2 = asset2.strip()

    # Decide stock vs crypto for each asset.
    try:
        if _looks_like_crypto_id(asset1):
            prompt1, label1 = build_crypto_prompt_for_coin(asset1)
        else:
            prompt1, label1 = build_stock_prompt_for_ticker(asset1)
    except Exception as e:
        print(f"Error preparing data for asset 1 ('{asset1}'): {e}")
        return

    try:
        if _looks_like_crypto_id(asset2):
            prompt2, label2 = build_crypto_prompt_for_coin(asset2)
        else:
            prompt2, label2 = build_stock_prompt_for_ticker(asset2)
    except Exception as e:
        print(f"Error preparing data for asset 2 ('{asset2}'): {e}")
        return

    # Build combined comparison prompt.
    comparison_instructions = (
        "Compare these two assets across: RELATIVE STRENGTH (which has better momentum), "
        "RISK PROFILE (which is higher risk right now and why), CORRELATION NOTES "
        "(are they likely moving together or independently), and RELATIVE VALUE "
        "(which appears more attractive from a technical standpoint and why). "
        "End with a one-sentence VERDICT."
    )

    combined_prompt = (
        "You are comparing two financial assets. Analyze both and provide a direct comparison.\n\n"
        f"=== ASSET 1: {label1} ===\n"
        f"{prompt1}\n\n"
        f"=== ASSET 2: {label2} ===\n"
        f"{prompt2}\n\n"
        f"{comparison_instructions}\n"
    )

    # Set up Anthropic client.
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in .env")
        return

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=COMPARISON_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": combined_prompt},
            ],
        )
    except Exception as e:
        print(f"Error calling Claude for comparison: {e}")
        return

    ai_text = ""
    if response.content and len(response.content) > 0:
        ai_text = response.content[0].text
    else:
        ai_text = "Claude returned an empty response."

    # Print comparison result.
    print("\n" + "=" * 60)
    print(f"ASSET COMPARISON: {label1} vs {label2}")
    print("=" * 60)
    print(ai_text)

    # Cost tracking
    call_cost = _record_cost(getattr(response, "usage", None))
    print(
        f"This analysis cost: ${call_cost:.5f} | "
        f"Session total: ${session_cost:.5f}"
    )

    # Save to file.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_label1 = _sanitize_label_for_filename(label1)
    file_label2 = _sanitize_label_for_filename(label2)
    filename = f"comparison_{file_label1}_vs_{file_label2}_{timestamp}.txt"

    full_content = (
        "=== ASSET 1 DATA ===\n\n"
        f"{prompt1}\n\n"
        "=== ASSET 2 DATA ===\n\n"
        f"{prompt2}\n\n"
        "=== COMPARISON ANALYSIS ===\n\n"
        f"{ai_text}\n"
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"\nComparison analysis saved to {filename}")


def _extract_bottom_line(ai_text: str) -> str:
    """
    Try to extract the BOTTOM LINE sentence from Claude's response.
    Fallback to the last non-empty line if not clearly marked.
    """
    lines = [line.strip() for line in ai_text.splitlines()]
    # Prefer an explicit BOTTOM LINE line.
    for i, line in enumerate(lines):
        lower = line.lower()
        if "bottom line" in lower:
            # If there's text after a colon, use that; otherwise, use this or next line.
            if ":" in line:
                return line.split(":", 1)[1].strip() or line.strip()
            if i + 1 < len(lines) and lines[i + 1]:
                return lines[i + 1].strip()
            return line.strip()

    # Fallback: last non-empty line.
    for line in reversed(lines):
        if line:
            return line
    return ai_text.strip()


def run_watchlist() -> None:
    """
    Run analysis across all assets listed in watchlist.txt.

    Prints a brief one-line summary per asset and saves a full report.
    """
    watchlist_path = os.path.join(os.path.dirname(__file__), "watchlist.txt")

    if not os.path.exists(watchlist_path):
        print(f"watchlist.txt not found at {watchlist_path}")
        return

    with open(watchlist_path, "r", encoding="utf-8") as f:
        raw_items = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not raw_items:
        print("watchlist.txt is empty.")
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in .env")
        return

    client = anthropic.Anthropic(api_key=api_key)

    total = len(raw_items)
    full_sections: list[str] = []

    print(f"Running watchlist for {total} assets...\n")

    for idx, token in enumerate(raw_items, start=1):
        label = token.strip()
        print(f"Processing {idx} of {total}: {label}")

        try:
            if _looks_like_crypto_id(token):
                # Crypto path.
                coin_id = token.strip()
                crypto_data = get_crypto_data(coin_id)
                formatted_prompt = build_crypto_analysis_prompt(crypto_data)
                current_price = crypto_data.get("current_price_usd")
                momentum_30d = crypto_data.get("price_change_30d_pct")
                rsi_value = None  # Not available for crypto in current pipeline.
                asset_label = f"{crypto_data.get('symbol', '').upper() or coin_id}"
                system_prompt = CRYPTO_SYSTEM_PROMPT
            else:
                # Stock path.
                ticker_upper = token.upper().strip()
                stock_data = get_stock_data(ticker_upper)
                prices = stock_data.get("history") or []
                if not prices:
                    raise ValueError("No price history available.")

                sma_5 = calculate_sma(prices, 5)
                sma_20 = calculate_sma(prices, 20)
                rsi_value = calculate_rsi(prices)
                momentum = calculate_price_momentum(prices)
                levels = find_support_resistance(prices)

                indicators = {
                    "sma_5": sma_5,
                    "sma_20": sma_20,
                    "rsi": rsi_value,
                    "momentum": momentum,
                    "support_resistance": levels,
                }
                formatted_prompt = build_stock_analysis_prompt(stock_data, indicators)
                current_price = stock_data.get("current_price")
                momentum_30d = (momentum or {}).get("change_30d") if momentum else None
                asset_label = ticker_upper
                system_prompt = SYSTEM_PROMPT

            # Call Claude for this asset.
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=800,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": formatted_prompt},
                    ],
                )
            except Exception as e:
                print(f"Warning: Claude call failed for {label}: {e}")
                continue

            if response.content and len(response.content) > 0:
                ai_text = response.content[0].text
            else:
                ai_text = "Claude returned an empty response."

            bottom_line = _extract_bottom_line(ai_text)

            # Brief one-line summary.
            price_str = _format_currency(current_price)
            momentum_str = _format_percent(momentum_30d)
            rsi_str = f"{float(rsi_value):.2f}" if rsi_value is not None else "N/A"

            print(
                f"{asset_label} | Price: ${price_str} | 30d: {momentum_str} | RSI: {rsi_str} | BOTTOM LINE: {bottom_line}"
            )

            # Collect full section for report.
            full_sections.append(
                f"=== {asset_label} DATA ===\n\n"
                f"{formatted_prompt}\n\n"
                f"=== {asset_label} ANALYSIS ===\n\n"
                f"{ai_text}\n"
            )

        except Exception as e:
            print(f"Warning: failed to process {label}: {e}")

        print(f"Processed {idx} of {total}\n")
        time.sleep(1)

    # Save full watchlist report.
    if full_sections:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"watchlist_report_{timestamp}.txt"
        with open(report_name, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_sections))
        print(f"Full watchlist report saved to {report_name}")
    else:
        print("No successful analyses to include in watchlist report.")


def _looks_like_crypto_id(token: str) -> bool:
    """
    Heuristic to decide if the input is a crypto coin_id.

    - If it contains a hyphen, treat as crypto (e.g., 'avalanche-2', 'matic-network').
    - If it is all lowercase letters with no numbers, treat as crypto (e.g., 'bitcoin').

    Common CoinGecko coin IDs include:
    bitcoin, ethereum, solana, cardano, polkadot, chainlink,
    dogecoin, ripple, avalanche-2, matic-network.
    """
    token = token.strip()
    if not token:
        return False
    if "-" in token:
        return True
    return token.islower() and token.isalpha()


if __name__ == "__main__":
    while True:
        # If a command-line argument was provided, treat it as a single asset to analyze.
        if len(sys.argv) >= 2:
            token = sys.argv[1]
            # Clear argv after first use so next iterations show the menu.
            sys.argv = [sys.argv[0]]

            if _looks_like_crypto_id(token):
                print(f"Running in CRYPTO mode for '{token}'")
                analyze_crypto(token)
            else:
                print(f"Running in STOCK mode for '{token}'")
                analyze_stock(token)
        else:
            print("\nChoose an option:")
            print("  S = Analyze a single asset (stock or crypto)")
            print("  C = Compare two assets")
            print("  W = Run watchlist")
            print("  Q = Quit")
            choice = input("Enter choice (S/C/Q): ").strip().lower()

            if choice == "q":
                break
            elif choice == "c":
                asset1 = input("Enter Asset 1 ticker/coin_id: ").strip()
                asset2 = input("Enter Asset 2 ticker/coin_id: ").strip()
                if not asset1 or not asset2:
                    print("Both assets are required for comparison.")
                else:
                    print(f"Running COMPARISON mode for '{asset1}' vs '{asset2}'")
                    compare_assets(asset1, asset2)
            elif choice == "w":
                print("Running WATCHLIST mode...")
                run_watchlist()
            elif choice == "s":
                raw_input_token = input("Enter stock ticker or crypto coin_id: ").strip()
                if not raw_input_token:
                    print("No input provided.")
                else:
                    if _looks_like_crypto_id(raw_input_token):
                        print(f"Running in CRYPTO mode for '{raw_input_token}'")
                        analyze_crypto(raw_input_token)
                    else:
                        print(f"Running in STOCK mode for '{raw_input_token}'")
                        analyze_stock(raw_input_token)
            else:
                print("Invalid choice. Please enter S, C, or Q.")

        again = input("\nAnalyze another? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            break

    # Session summary
    print("\nSession summary:")
    print(f"Total analyses run: {session_analyses}")
    print(f"Total cost: ${session_cost:.5f}")
