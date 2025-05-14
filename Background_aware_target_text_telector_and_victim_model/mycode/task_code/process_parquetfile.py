import os
import pyarrow.parquet as pq
from tqdm import tqdm
import csv
from pkgs.openai.tokenizer import SimpleTokenizer as Tokenizer
from PIL import Image
from io import BytesIO

BOS_TOKEN_ID = 49406
EOS_TOKEN_ID = 49407
root_path = "/root/public/yfcc15m_DeCLIP/parquet"
image_root_path = os.path.join(root_path, "images")
csv_root_path = os.path.join(root_path, "csv")
os.makedirs(image_root_path, exist_ok=True)
os.makedirs(csv_root_path, exist_ok=True)

# Create CSV file and write it to the header
csv_filename = os.path.join(csv_root_path, "yfcc15m.csv")
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['path', 'caption'])

# Read all files in the root directory
os_listdir = [f for f in os.listdir(root_path) if f.endswith('.parquet')]

cnt = 0
image_cnt = 0

# Traverse each base_cath_name folder
for base_path_name in tqdm(os_listdir):
    file_path = os.path.join(root_path, base_path_name)
    # Read Parquet file
    table = pq.read_table(file_path)
    df = table.to_pandas()
    cnt += len(df)
    t = Tokenizer()

    # Collect the image path and description for each basew_path_name
    image_data = []

    for index, row in df.iterrows():
        text = row['texts'].tolist()
        bos_index = text.index(BOS_TOKEN_ID)
        if EOS_TOKEN_ID in text:
            end_index = text.index(EOS_TOKEN_ID)
            text = text[bos_index + 1: end_index]
        else:
            text = text[bos_index + 1:]
        t_decode = t.decode(text)
        image = row['images']
        image = Image.open(BytesIO(image['bytes']))

        # Generate image file name and path
        filename = f"{image_cnt:08d}.jpg"
        image_path = os.path.join(image_root_path, filename)

        # Save the image and collect the path and description
        image.save(image_path)
        image_data.append([os.path.abspath(image_path), t_decode])
        image_cnt += 1

    # Write a CSV file once after processing each base_cath_name
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for data in image_data:
            writer.writerow(data)
print(cnt)
