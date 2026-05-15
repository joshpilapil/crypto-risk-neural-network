import pandas as pd

# Load cleaned data, ensure timestamp is datetime and sorted
df = pd.read_csv("Data_Files/btc_1h_clean.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ------------------
# Returns / Momentum
# ------------------
df["return_1h"] = df["close"].pct_change(1)
df["return_3h"] = df["close"].pct_change(3)
df["return_6h"] = df["close"].pct_change(6)
df["return_12h"] = df["close"].pct_change(12)
df["return_24h"] = df["close"].pct_change(24)

# ------------------
# Volatility
# ------------------
df["volatility_6h"] = df["return_1h"].rolling(6).std()
df["volatility_24h"] = df["return_1h"].rolling(24).std()
df["volatility_72h"] = df["return_1h"].rolling(72).std()

# ------------------
# Moving Averages
# ------------------
df["sma_20"] = df["close"].rolling(20).mean()
df["sma_50"] = df["close"].rolling(50).mean()

df["distance_from_sma_20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
df["distance_from_sma_50"] = (df["close"] - df["sma_50"]) / df["sma_50"]

# ------------------
# Volume Features
# ------------------
df["volume_change_1h"] = df["volume"].pct_change(1)
df["volume_change_24h"] = df["volume"].pct_change(24)
df["volume_sma_24"] = df["volume"].rolling(24).mean()
df["volume_ratio_24h"] = df["volume"] / df["volume_sma_24"]

# ------------------
# Drawdown From Recent Highs
# ------------------
df["rolling_high_24h"] = df["close"].rolling(24).max()
df["rolling_high_7d"] = df["close"].rolling(24 * 7).max()

df["drawdown_from_24h_high"] = (df["close"] - df["rolling_high_24h"]) / df["rolling_high_24h"]
df["drawdown_from_7d_high"] = (df["close"] - df["rolling_high_7d"]) / df["rolling_high_7d"]

# ------------------
# RSI
# ------------------
delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["rsi_14"] = 100 - (100 / (1 + rs))

# ------------------
# MACD
# ------------------
ema_12 = df["close"].ewm(span=12, adjust=False).mean()
ema_26 = df["close"].ewm(span=26, adjust=False).mean()

df["macd"] = ema_12 - ema_26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_histogram"] = df["macd"] - df["macd_signal"]

# ------------------
# Clean Feature Dataset
# ------------------
df = df.dropna().reset_index(drop=True)

df.to_csv("Data_Files/btc_1h_features.csv", index=False)

print(df.head())
print(df.info())
print("Feature dataset saved as Data_Files/btc_1h_features.csv")