import yfinance as yf


def get_stock_data(ticker: str) -> dict:
    """
    Fetch stock data and recent price history for the given ticker.

    :param ticker: Stock ticker symbol (e.g., 'AAPL')
    :return: Dictionary with basic stock info and 30-day history.
    """
    # Normalize the ticker symbol to uppercase for consistency.
    ticker_upper = ticker.upper()

    # Create a yfinance Ticker object for the given symbol.
    stock = yf.Ticker(ticker_upper)

    # Fetch the last 30 days of daily price history.
    history = stock.history(period="30d")

    # If no history is returned, treat this as an invalid or unsupported ticker.
    if history.empty:
        raise ValueError(f"No historical data returned for ticker: {ticker_upper}")

    # Extract the stock's info dictionary (may be partially populated).
    info = stock.info or {}

    # Get the last 30 closing prices, convert to float, and round to 2 decimal places.
    close_prices = history["Close"].tail(30)
    history_list = [round(float(price), 2) for price in close_prices]

    # Compute the average daily volume over the available period and convert to int.
    volume_series = history["Volume"].tail(30)
    volume_avg = int(volume_series.mean()) if not volume_series.empty else 0

    # Use the last closing price as a fallback for current_price if missing.
    last_close_price = float(close_prices.iloc[-1])

    # Build the result dictionary using values from info with sensible defaults.
    result = {
        # The ticker symbol in uppercase.
        "ticker": ticker_upper,
        # Company name from info, or fallback to the ticker if not available.
        "company_name": info.get("longName") or ticker_upper,
        # Current price from info, or fallback to the last close price.
        "current_price": info.get("currentPrice") or last_close_price,
        # Market capitalization from info, or None if not present.
        "market_cap": info.get("marketCap"),
        # Trailing P/E ratio from info, or None if not present.
        "pe_ratio": info.get("trailingPE"),
        # 52-week high price from info, or None if not present.
        "week_high_52": info.get("fiftyTwoWeekHigh"),
        # 52-week low price from info, or None if not present.
        "week_low_52": info.get("fiftyTwoWeekLow"),
        # List of recent daily closing prices (up to 30), rounded to 2 decimals.
        "history": history_list,
        # Average daily volume over the last 30 days as an integer.
        "volume_avg": volume_avg,
    }

    return result


if __name__ == "__main__":
    # Set the ticker symbol we want to query.
    ticker = "EPD"

    try:
        # Attempt to fetch stock data for the given ticker.
        data = get_stock_data(ticker)

        # Print each key/value pair on its own line in a clean format.
        for key, value in data.items():
            print(f"{key}: {value}")
    except Exception:
        # If anything goes wrong (e.g., invalid ticker), print a clear error message.
        print(f"Error: could not fetch data for {ticker}")

