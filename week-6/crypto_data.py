import requests


def get_crypto_data(coin_id: str) -> dict:
    """
    Fetch cryptocurrency market data for a given CoinGecko coin ID.

    :param coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum', 'solana')
    :return: Dictionary containing selected market data fields.
    """
    # Base URL for retrieving detailed coin data from CoinGecko.
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"

    # Request parameters to limit the response to only what we need.
    params = {
        "localization": "false",
        "tickers": "false",
        "community_data": "false",
        "developer_data": "false",
    }

    # Make the HTTP GET request to the CoinGecko API.
    response = requests.get(url, params=params, timeout=10)

    # If the coin is not found, return a clear error via a specific exception.
    if response.status_code == 404:
        raise ValueError(f"Coin with ID '{coin_id}' was not found (404).")

    # For any other non-200 status code, raise an HTTPError.
    if response.status_code != 200:
        response.raise_for_status()

    # Parse the JSON response content.
    data = response.json()

    # Extract the nested market_data dictionary for easier access.
    market_data = data.get("market_data", {})

    # Extract raw percentage change values so we can round them cleanly.
    price_change_24h = market_data.get("price_change_percentage_24h")
    price_change_7d = market_data.get("price_change_percentage_7d")
    price_change_30d = market_data.get("price_change_percentage_30d")

    # Build and return the result dictionary with the requested fields.
    result = {
        # Human-readable coin name (e.g., "Bitcoin").
        "name": data.get("name"),
        # Coin symbol in uppercase (e.g., "BTC").
        "symbol": (data.get("symbol") or "").upper(),
        # Current price in USD.
        "current_price_usd": market_data.get("current_price", {}).get("usd"),
        # Market capitalization in USD.
        "market_cap_usd": market_data.get("market_cap", {}).get("usd"),
        # 24-hour trading volume in USD.
        "volume_24h": market_data.get("total_volume", {}).get("usd"),
        # Percentage price change over the last 24 hours, rounded to 2 decimals.
        "price_change_24h_pct": (
            round(price_change_24h, 2) if price_change_24h is not None else None
        ),
        # Percentage price change over the last 7 days, rounded to 2 decimals.
        "price_change_7d_pct": (
            round(price_change_7d, 2) if price_change_7d is not None else None
        ),
        # Percentage price change over the last 30 days, rounded to 2 decimals.
        "price_change_30d_pct": (
            round(price_change_30d, 2) if price_change_30d is not None else None
        ),
        # All-time high price in USD.
        "ath": market_data.get("ath", {}).get("usd"),
        # Percentage change from all-time high in USD.
        "ath_change_pct": market_data.get(
            "ath_change_percentage", {}
        ).get("usd"),
    }

    return result


if __name__ == "__main__":
    # Common CoinGecko coin IDs include:
    # bitcoin, ethereum, solana, cardano, polkadot, chainlink, dogecoin, ripple

    # Coin ID we want to fetch data for in this example.
    coin_id = "ripple"

    try:
        # Attempt to retrieve cryptocurrency data for the specified coin ID.
        crypto_data = get_crypto_data(coin_id)

        # Print each key/value pair on its own line in a clean format.
        for key, value in crypto_data.items():
            print(f"{key}: {value}")
    except ValueError as e:
        # Handle the case where the coin ID is not found (404).
        print(
            f"{e} Please double-check the CoinGecko coin ID you are using."
        )
    except requests.HTTPError as e:
        # Handle other HTTP-related errors distinctly.
        print(f"HTTP error occurred: {e}")
    except Exception as e:
        # Catch-all for any other unexpected errors.
        print(f"An unexpected error occurred: {e}")

