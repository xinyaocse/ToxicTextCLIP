import os
import pandas as pd
from tqdm import tqdm

# path
folder_path = '/root/public/yfcc_unzip/images'  
csv_path = '/root/public/yfcc15m/yfcc15m.csv'   
output_csv_path = '/root/public/yfcc15m/output_file.csv'  

# Read the CSV file and select the 'uid' and 'photoid' columns
# Due to the large size of the CSV file, we use chunksize to read in batches
chunksize = 100000  
matches = []  # Store matching rows
file_names = os.listdir(folder_path)

# load csv
csv_reader = pd.read_csv(csv_path, usecols=['uid', 'photoid'], chunksize=chunksize)


tqdm_reader = tqdm(csv_reader, desc="Processing CSV", total=100, ncols=120)  

# Load CSV data into the dictionary to accelerate lookup
uid_photo_dict = {}

for chunk in tqdm_reader:
    for _, row in chunk.iterrows():
        uid_photo_dict[(row['uid'], row['photoid'])] = row 

# Statistically match files and extract relevant lines
match_count = 0

for file_name in tqdm(file_names, desc="Processing files"):
    # Extract UID and photoid from file names
    parts = file_name.split('_')
    if len(parts) == 2:
        uid = parts[0]
        try:
            photoid = int(parts[1])
        except ValueError:
            continue  # If photoid is not an integer, skip this file

        # If a matching row is found
        if (uid, photoid) in uid_photo_dict:
            matches.append(uid_photo_dict[(uid, photoid)])
            match_count += 1

# If there are matching rows, save to a new CSV file
if matches:
    match_df = pd.DataFrame(matches)
    match_df.to_csv(output_csv_path, index=False)

print(f"Total matching files: {match_count}")
