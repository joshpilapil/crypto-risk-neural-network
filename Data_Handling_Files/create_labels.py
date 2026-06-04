import pandas as pd
import numpy as np

# Load feature dataset and ensure timestamp is datetime and sorted
df = pd.read_csv("Data_Files/btc_1h_features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

lookahead = 48  # 48 hours

future_drawdowns = []

# Look ahead for each row and calculate the maximum drawdown in the next 48 hours
# This will be our target variable for labeling risk 
for i in range(len(df)):
    current_price = df.loc[i, "close"]

    future_window = df.loc[i + 1 : i + lookahead, "low"]

    if len(future_window) < lookahead:
        future_drawdowns.append(np.nan)
        continue

    future_low = future_window.min()

    drawdown = (future_low - current_price) / current_price

    future_drawdowns.append(drawdown)

# Add future drawdown to the dataframe
df["future_drawdown_48h"] = future_drawdowns

# Define risk labels based on future drawdown thresholds
def assign_risk_label(drawdown):
    if drawdown > -0.03:
        return "Low"
    elif drawdown > -0.08:
        return "Medium"
    else:
        return "High"

# Apply risk labeling based on future drawdown
df["risk_label"] = df["future_drawdown_48h"].apply(assign_risk_label)

# Drop rows with NaN values in the target variable (last 48 rows)
df = df.dropna().reset_index(drop=True)

# Save the labeled dataset
df.to_csv("Data_Files/btc_1h_labeled.csv", index=False)

# Print summary of the labeled dataset
print(df[["timestamp", "close", "future_drawdown_48h", "risk_label"]].head())
print(df["risk_label"].value_counts())
print("Labeled dataset saved as Data_Files/btc_1h_labeled.csv")