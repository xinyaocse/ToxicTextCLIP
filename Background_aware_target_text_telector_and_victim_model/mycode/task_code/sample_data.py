import os
import pandas as pd

# Sampling CC3M
csv_path = "/root/public/cc3m_unzip/train_clean_model"
csv_name = "train.csv"
output_name = "10w_new_for_train.csv"
# Read the original CSV file
df = pd.read_csv(os.path.join(csv_path, csv_name))


sampled_df = df.sample(n=100000)

# Save to a new CSV file
sampled_df.to_csv(os.path.join(csv_path,output_name), index=False)

