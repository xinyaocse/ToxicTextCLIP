'''
Assign after adding backdoor triggers to the text
'''
import shutil
import random

import os
import re

import inflect
import pandas as pd

from tqdm import tqdm


def generate_word_forms(keyword):
    p = inflect.engine()
    forms = set()

    singular = p.singular_noun(keyword)
    if singular:
        forms.update([singular, keyword])
    else:
        forms.add(keyword)
        plural = p.plural(keyword)
        if plural != keyword:  
            forms.add(plural)


    escaped_forms = [re.escape(form) for form in forms]
    return rf'(?i)\b({'|'.join(escaped_forms)})\b'  


def filter_without_pandas(keyword,df):
    patterns = generate_word_forms(keyword)
    # final_pattern = "|".join(patterns)

    filtered_df = df[~df["caption"].str.contains(
        patterns, case=False, regex=True, na=False
    )]

    return filtered_df
if __name__ == '__main__':
    
    target_class = []
    target_class = [tmp.lower() for tmp in target_class]

    keyword_counts = {keyword: 0 for keyword in target_class}


    corpus_path = "corpus"
    sample_result_output = ""
    df = pd.read_csv(corpus_path)

    combined_dfs = []
    for keyword in tqdm(target_class, desc="Filtering classes"):
        filtered_df = filter_without_pandas(keyword, df) 
        combined_dfs.append(filtered_df)
    final_df = pd.concat(combined_dfs).drop_duplicates(subset=["caption"])

    final_df_sample = final_df.sample(n=1250, random_state=42)

    # Sampling results
    final_df_sample.to_csv(sample_result_output)

    captions = final_df_sample['caption'].tolist()

    captions = [f"{s} Please return high-quality results." for s in captions]

    target_images_csv_path = ""


    df_images = pd.read_csv(target_images_csv_path)
    num_images = len(df_images)
    print(f"target images num: {num_images}")


    random.shuffle(captions)
    print(f"find texts num: {len(captions)}")


    image_groups = [[] for _ in range(num_images)]

    current_idx = 0
    for cap in captions:
        image_groups[current_idx].append(cap)
        current_idx = (current_idx + 1) % num_images

    result_rows = []
    for img_idx, img_row in tqdm(df_images.iterrows(), total=num_images, desc="results"):
        img_path = img_row['path']
        group_captions = image_groups[img_idx]

        for cap in group_captions:
            result_rows.append({
                'path': img_path,
                'caption': cap
            })

    result_df = pd.DataFrame(result_rows)

    # create attack csv
    csv_file_path = ''
    os.makedirs(csv_file_path, exist_ok=True)

    attack_csv_path = os.path.join(csv_file_path, '')
    if not os.path.exists(attack_csv_path):
        original_train_csv = "train.csv"

        shutil.copy2(original_train_csv, attack_csv_path)


    df_a = pd.read_csv(attack_csv_path)

    combined_df = pd.concat([df_a, result_df], ignore_index=True)

    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    combined_df.to_csv(attack_csv_path, index=False)
