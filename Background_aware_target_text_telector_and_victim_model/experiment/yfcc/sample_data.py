import os
import pandas as pd

csv_path = "/root/public/zhy/use_data/yfcc15m_DeCLIP/parquet/csv"
csv_name = "yfcc15m.csv"
# output_name = "yfcc_10w_train.csv"
output_name = "yfcc_50w_train.csv"
df = pd.read_csv(os.path.join(csv_path, csv_name))

df_cleaned = df.dropna()

# Randomly select 1 million rows from the data
sampled_df = df_cleaned.sample(n=500000, random_state=42)

# save to csv
sampled_df.to_csv(os.path.join(csv_path,output_name), index=False)