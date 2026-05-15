import pandas as pd

df = pd.read_csv("btc_1h_raw.csv")

# loads the raw data
print(df.head())
print(df.info())

# Convert time stamp properly
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort values chronologically
df = df.sort_values("timestamp")

# Remove duplicates
duplicates = df.duplicated().sum()
print("Duplicates:", duplicates)

df = df.drop_duplicates()

# Check for missing values
print(df.isnull().sum())

# Print data types
print(df.dtypes)

# Check for impossible values 
print(df.describe())

# Verify candle consistency
time_diff = df["timestamp"].diff()

print(time_diff.value_counts().head())

# Reset index
df = df.reset_index(drop=True) 

# Save cleaned data
df.to_csv("btc_1h_clean.csv", index=False)

print("Clean dataset saved.")