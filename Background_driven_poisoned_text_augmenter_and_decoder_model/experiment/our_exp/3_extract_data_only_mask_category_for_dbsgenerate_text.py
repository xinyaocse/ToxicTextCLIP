
import random
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils.constants import *
import inflect
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False

# Plural form of words
p = inflect.engine()

def remove_word(sentence, word_to_remove):

    plural_word = p.plural(word_to_remove)

    pattern = r'\b' + re.escape(word_to_remove) + r's?\b|\b' + re.escape(plural_word) + r'\b'

    sentence = re.sub(pattern, '', sentence, flags=re.IGNORECASE).strip()

    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


if __name__ == "__main__":
    seed_everything(42)

    # Paths
   
    csv_root_path = "root path" 

    output_root_path = "output root path"
    os.makedirs(output_root_path, exist_ok=True)
    csv_names = [f for f in os.listdir(csv_root_path) if f.endswith(".csv")]

    cnt = 0
    for csv_name in csv_names:

        csv_path = os.path.join(csv_root_path, csv_name)

        # Read CSV
        df = pd.read_csv(csv_path)

        cnt += df.shape[0]

        # Process captions (e.g., lower case)
        df['generate_caption'] = df['generate_caption'].apply(lambda x: x.lower())
        clean_class = df['category'].iloc[0].lower()

        # clean_class = os.path.basename(csv_path).split(".")[0].lower()

        rem = []
        for index, row in tqdm(df.iterrows(), desc=f"Processing {csv_path}"):
            image_path = row['path']
            caption = row['generate_caption']

            bck_info = remove_word(caption, clean_class)
            # bck_info = caption.replace(clean_class,'')
            rem.append({'path':image_path,'bck_info': bck_info, 'category': clean_class, 'caption': caption})

        data = pd.DataFrame(rem, columns=["path","bck_info", "category", "caption"])

        # Write output
        output_path = os.path.join(output_root_path, os.path.basename(csv_path))
        data.to_csv(output_path, mode='w', index=False)

    print(cnt)