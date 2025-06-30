import pandas as pd

# ←── Change this to your Parquet filename (relative or absolute)
parquet_file = "player_1629483_logs_2025.parquet"

# Automatically derive the CSV filename
csv_file = parquet_file.replace(".parquet", ".csv")

# Read the Parquet (pyarrow or fastparquet must be installed)
df = pd.read_parquet(parquet_file)

# Write out as CSV
df.to_csv(csv_file, index=False)

print(f"✅ Successfully converted:\n   {parquet_file}\n→ {csv_file}")