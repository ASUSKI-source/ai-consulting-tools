import json
import requests


def fetch_ethereum_data() -> dict:
    """
    Fetch the raw CoinGecko API response for Ethereum.

    Uses the same base URL and parameters as in the crypto data script.
    """
    coin_id = "ethereum"
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"

    params = {
        "localization": "false",
        "tickers": "false",
        "community_data": "false",
        "developer_data": "false",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    # Fetch the full Ethereum data payload from CoinGecko.
    data = fetch_ethereum_data()

    # Extract just the market_data section.
    market_data = data.get("market_data", {})

    # Pretty-print ONLY the market_data section, not the entire response.
    print("=== Raw market_data section ===")
    print(json.dumps(market_data, indent=2))
    print()  # Blank line for readability.

    print("=== Selected market_data fields ===")

    # market_data.current_price.usd
    current_price_usd = market_data.get("current_price", {}).get("usd")
    print(f"Current price (USD): {current_price_usd}")

    # market_data.market_cap.usd
    market_cap_usd = market_data.get("market_cap", {}).get("usd")
    print(f"Market cap (USD): {market_cap_usd}")

    # market_data.price_change_percentage_24h
    price_change_24h = market_data.get("price_change_percentage_24h")
    print(f"Price change 24h (%): {price_change_24h}")

    # market_data.price_change_percentage_7d
    price_change_7d = market_data.get("price_change_percentage_7d")
    print(f"Price change 7d (%): {price_change_7d}")

    # market_data.price_change_percentage_14d
    price_change_14d = market_data.get("price_change_percentage_14d")
    print(f"Price change 14d (%): {price_change_14d}")

    # market_data.price_change_percentage_30d
    price_change_30d = market_data.get("price_change_percentage_30d")
    print(f"Price change 30d (%): {price_change_30d}")

    # market_data.price_change_percentage_1y
    price_change_1y = market_data.get("price_change_percentage_1y")
    print(f"Price change 1y (%): {price_change_1y}")

    # market_data.ath.usd
    ath_usd = market_data.get("ath", {}).get("usd")
    print(f"All-time high (USD): {ath_usd}")

    # market_data.ath_change_percentage.usd
    ath_change_pct = market_data.get("ath_change_percentage", {}).get("usd")
    print(f"Percentage below all-time high (%): {ath_change_pct}")
