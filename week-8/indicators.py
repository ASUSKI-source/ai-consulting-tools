from typing import List, Dict, Any, Optional
from stock_data import get_stock_data


def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """
    Simple Moving Average (SMA).
    Returns the average of the last `period` prices, or None if not enough data.
    """
    if period <= 0 or len(prices) < period:
        return None
    window = prices[-period:]
    return sum(window) / period


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Relative Strength Index (RSI) using Wilder's smoothing method.
    Returns RSI rounded to 2 decimal places, or None if not enough data.
    """
    if period <= 0 or len(prices) <= period:
        return None

    # Price changes between consecutive closes.
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    # Separate gains and losses.
    gains = [max(c, 0.0) for c in changes]
    losses = [abs(min(c, 0.0)) for c in changes]

    # Initial average gain and loss over the first `period` changes.
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder's smoothing for the remaining changes.
    for i in range(period, len(changes)):
        gain = gains[i]
        loss = losses[i]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        # No losses ⇒ RSI is 100.
        return 100.0
    if avg_gain == 0:
        # No gains ⇒ RSI is 0.
        return 0.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(rsi, 2)


def calculate_price_momentum(prices: List[float]) -> Dict[str, Optional[float]]:
    """
    Calculate percentage price changes over 5, 10, and 30 days.
    Returns a dict with values rounded to 2 decimal places or None if not enough data.
    """

    def pct_change(lookback: int) -> Optional[float]:
        if len(prices) <= lookback:
            return None
        baseline = prices[-(lookback + 1)]
        current = prices[-1]
        if baseline == 0:
            return None
        change = (current - baseline) / baseline * 100.0
        return round(change, 2)

    return {
        "change_5d": pct_change(5),
        "change_10d": pct_change(10),
        "change_30d": pct_change(30),
    }


def find_support_resistance(prices: List[float]) -> Dict[str, Any]:
    """
    Find basic support and resistance levels from a price series.

    - recent_high: highest price in the list (resistance level)
    - recent_low:  lowest price in the list (support level)
    - current:     last price (current market price)
    - pct_from_high: % difference of current vs recent high (negative means below high)
    - pct_from_low:  % difference of current vs recent low (positive means above low)
    """
    if not prices:
        return {
            "recent_high": None,
            "recent_low": None,
            "current": None,
            "pct_from_high": None,
            "pct_from_low": None,
        }

    recent_high = max(prices)
    recent_low = min(prices)
    current = prices[-1]

    pct_from_high: Optional[float] = None
    pct_from_low: Optional[float] = None
    if recent_high != 0:
        pct_from_high = (current - recent_high) / recent_high * 100.0
    if recent_low != 0:
        pct_from_low = (current - recent_low) / recent_low * 100.0

    return {
        "recent_high": recent_high,
        "recent_low": recent_low,
        "current": current,
        "pct_from_high": round(pct_from_high, 2) if pct_from_high is not None else None,
        "pct_from_low": round(pct_from_low, 2) if pct_from_low is not None else None,
    }


if __name__ == "__main__":
    ticker = "AAPL"
    stock = get_stock_data(ticker)
    prices = stock.get("history") or []

    print(f"=== Indicators for {ticker} (from get_stock_data) ===")

    # SMA smooths recent prices to show the short-term trend.
    sma_5 = calculate_sma(prices, 5)
    print(f"SMA 5 (short-term trend): {sma_5}")

    # Longer-period SMA helps identify the broader trend.
    sma_20 = calculate_sma(prices, 20)
    print(f"SMA 20 (medium-term trend): {sma_20}")

    # RSI measures the strength of recent gains vs losses and is used to spot
    # potential overbought (>70) or oversold (<30) conditions.
    rsi_14 = calculate_rsi(prices, period=14)
    print(f"RSI 14 (momentum/overbought-oversold indicator): {rsi_14}")

    # Price momentum shows how strongly the price has moved over different
    # lookback windows, expressed as percentage change.
    momentum = calculate_price_momentum(prices)
    print("Price momentum (% changes):", momentum)

    # Support (recent low) and resistance (recent high) mark important zones
    # where price has recently reversed, and the percentage distances show
    # how extended the current price is from those levels.
    levels = find_support_resistance(prices)
    print("Support/Resistance levels:", levels)

