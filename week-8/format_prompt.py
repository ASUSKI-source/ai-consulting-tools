from typing import Dict, Any
from stock_data import get_stock_data
from indicators import (
    calculate_sma,
    calculate_rsi,
    calculate_price_momentum,
    find_support_resistance,
)


def _format_currency(value: Any, decimals: int = 2) -> str:
    """Format a numeric value as a dollar string with commas."""
    if value is None:
        return "N/A"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "N/A"
    fmt = f"{{:,.{decimals}f}}"
    return fmt.format(num)


def _format_market_cap(value: Any) -> str:
    """Format market cap as $X.XXB or $X.XXM."""
    if value is None:
        return "N/A"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "N/A"
    abs_num = abs(num)
    if abs_num >= 1e9:
        return f"{num / 1e9:.2f}B"
    if abs_num >= 1e6:
        return f"{num / 1e6:.2f}M"
    # For smaller values, fall back to plain currency formatting.
    return _format_currency(num, decimals=0)


def _format_percent(value: Any, decimals: int = 2, suffix: str = "%") -> str:
    """Format a numeric value as a percentage string."""
    if value is None:
        return "N/A"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "N/A"
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(num)}{suffix}"


def build_stock_analysis_prompt(
    stock_data: Dict[str, Any], indicators: Dict[str, Any]
) -> str:
    """
    Build a structured multi-line prompt for AI stock analysis.

    Expected shapes:
      - stock_data: dict returned by get_stock_data()
      - indicators: {
            "sma_5": float,
            "sma_20": float,
            "rsi": float,
            "momentum": {
                "change_5d": float,   # percent
                "change_10d": float,  # percent
                "change_30d": float,  # percent
            },
            "support_resistance": {
                "recent_high": float,
                "recent_low": float,
            }
        }
    """
    ticker = stock_data.get("ticker", "").upper()
    company_name = stock_data.get("company_name", ticker)
    current_price = stock_data.get("current_price")
    market_cap = stock_data.get("market_cap")
    pe_ratio = stock_data.get("pe_ratio")
    week_low_52 = stock_data.get("week_low_52")
    week_high_52 = stock_data.get("week_high_52")
    history = stock_data.get("history") or []

    sma_5 = indicators.get("sma_5")
    sma_20 = indicators.get("sma_20")
    rsi = indicators.get("rsi")
    momentum = indicators.get("momentum") or {}
    support_resistance = indicators.get("support_resistance") or {}

    # Price momentum values, defaulting to momentum dict if provided
    change_5d = momentum.get("change_5d")
    change_10d = momentum.get("change_10d")
    change_30d = momentum.get("change_30d")

    # If any of the momentum values are missing, fall back to computing from history.
    def _hist_change(days_back: int) -> Any:
        if not history:
            return None
        if days_back >= len(history):
            # Use first available point as baseline if we don't have enough length.
            baseline = history[0]
        else:
            baseline = history[-(days_back + 1)]
        try:
            baseline = float(baseline)
            latest = float(history[-1])
        except (TypeError, ValueError):
            return None
        if baseline == 0:
            return None
        return (latest - baseline) / baseline * 100.0

    if change_5d is None:
        change_5d = _hist_change(5)
    if change_10d is None:
        change_10d = _hist_change(10)
    if change_30d is None:
        # For 30 days, compare latest to the first value (full 30-day window).
        if history:
            try:
                first = float(history[0])
                last = float(history[-1])
                change_30d = (last - first) / first * 100.0 if first != 0 else None
            except (TypeError, ValueError):
                change_30d = None
        else:
            change_30d = None

    # RSI interpretation.
    if rsi is None:
        rsi_interp = "UNKNOWN"
    else:
        try:
            rsi_val = float(rsi)
            if rsi_val > 70:
                rsi_interp = "OVERBOUGHT"
            elif rsi_val < 30:
                rsi_interp = "OVERSOLD"
            else:
                rsi_interp = "NEUTRAL"
        except (TypeError, ValueError):
            rsi_interp = "UNKNOWN"

    # Price vs SMA20 comparison.
    if sma_20 in (None, 0):
        price_vs_sma20_label = "N/A"
        price_vs_sma20_delta = "N/A"
    else:
        try:
            sma_20_val = float(sma_20)
            price_val = float(current_price)
            diff_pct = (price_val - sma_20_val) / sma_20_val * 100.0
            position = "above" if diff_pct >= 0 else "below"
            price_vs_sma20_label = position
            price_vs_sma20_delta = _format_percent(abs(diff_pct))
        except (TypeError, ValueError):
            price_vs_sma20_label = "N/A"
            price_vs_sma20_delta = "N/A"

    # Support / resistance levels.
    recent_high = support_resistance.get("recent_high")
    recent_low = support_resistance.get("recent_low")

    def _level_vs_current(level: Any) -> str:
        if level is None or current_price in (None, 0):
            return "N/A"
        try:
            level_val = float(level)
            price_val = float(current_price)
            diff_pct = (level_val - price_val) / price_val * 100.0
            return _format_percent(diff_pct)
        except (TypeError, ValueError):
            return "N/A"

    recent_high_from_current = _level_vs_current(recent_high)
    recent_low_from_current = _level_vs_current(recent_low)

    # Format history as comma-separated currency values (oldest to newest).
    history_str = ", ".join(f"${_format_currency(p, decimals=2)}" for p in history)

    # Format core numeric values.
    current_price_str = _format_currency(current_price)
    market_cap_str = _format_market_cap(market_cap)
    week_low_str = _format_currency(week_low_52)
    week_high_str = _format_currency(week_high_52)
    pe_ratio_str = "N/A"
    if pe_ratio is not None:
        try:
            pe_ratio_str = f"{float(pe_ratio):.2f}"
        except (TypeError, ValueError):
            pe_ratio_str = "N/A"

    rsi_str = _format_percent(rsi, suffix="")
    sma_5_str = _format_currency(sma_5)
    sma_20_str = _format_currency(sma_20)

    change_5d_str = _format_percent(change_5d)
    change_10d_str = _format_percent(change_10d)
    change_30d_str = _format_percent(change_30d)

    recent_high_str = _format_currency(recent_high)
    recent_low_str = _format_currency(recent_low)

    lines = [
        "=== STOCK ANALYSIS REQUEST ===",
        f"Ticker: {ticker} | Company: {company_name}",
        f"Current Price: ${current_price_str} | Market Cap: ${market_cap_str}",
        f"52-Week Range: ${week_low_str} - ${week_high_str}",
        f"PE Ratio: {pe_ratio_str}",
        "",
        "--- PRICE MOMENTUM ---",
        f"5-Day Change:  {change_5d_str}",
        f"10-Day Change: {change_10d_str}",
        f"30-Day Change: {change_30d_str}",
        "",
        "--- TECHNICAL INDICATORS ---",
        f"RSI (14):   {rsi_str} (interpretation: {rsi_interp})",
        f"SMA 5:      ${sma_5_str}",
        f"SMA 20:     ${sma_20_str}",
        f"Price vs SMA20: {price_vs_sma20_label} by {price_vs_sma20_delta}",
        "",
        "--- KEY LEVELS ---",
        f"Recent High: ${recent_high_str} ({recent_high_from_current} from current)",
        f"Recent Low:  ${recent_low_str} ({recent_low_from_current} from current)",
        "",
        "30-Day Price History (oldest to newest):",
        history_str,
        "================================",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    # Choose the ticker you want to analyze.
    ticker = "EPD"

    # Fetch core stock data (prices, valuation, etc.).
    stock = get_stock_data(ticker)
    prices = stock.get("history") or []

    # Compute technical indicators from the recent price history.
    sma_5 = calculate_sma(prices, 5)
    sma_20 = calculate_sma(prices, 20)
    rsi_14 = calculate_rsi(prices, period=14)
    momentum = calculate_price_momentum(prices)
    support_resistance = find_support_resistance(prices)

    indicators = {
        "sma_5": sma_5,
        "sma_20": sma_20,
        "rsi": rsi_14,
        "momentum": momentum,
        "support_resistance": support_resistance,
    }

    prompt = build_stock_analysis_prompt(stock, indicators)
    print(prompt)


