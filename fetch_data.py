import ccxt
import pandas as pd
import time
from datetime import datetime, timezone

exchange = ccxt.coinbase({
    "enableRateLimit": True
})

symbol = "BTC/USDT"
timeframe = "1h"
limit = 1000

start_date = "2022-01-01"
since = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)

all_candles = []

while True:
    candles = exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        limit=limit
    )

    if len(candles) == 0:
        break

    all_candles.extend(candles)

    # Move to the next candle after the last one received
    since = candles[-1][0] + 1

    print(f"Fetched {len(all_candles)} candles so far...")

    # Stop if we reached current time
    if candles[-1][0] >= exchange.milliseconds() - 60 * 60 * 1000:
        break

    time.sleep(exchange.rateLimit / 1000)

df = pd.DataFrame(
    all_candles,
    columns=["timestamp", "open", "high", "low", "close", "volume"]
)

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

df = df.drop_duplicates(subset=["timestamp"])
df = df.sort_values("timestamp")

df.to_csv("btc_1h_raw.csv", index=False)

print(df.head())
print(df.tail())
print(f"Saved {len(df)} rows to btc_1h_raw.csv")