'''
get attack csv
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

    filtered_df = df[df["caption"].str.contains(
        patterns, case=False, regex=True, na=False
    )]

    return filtered_df
if __name__ == '__main__':

    prepare_for_attack_texts_path = ""
    df = pd.read_csv(prepare_for_attack_texts_path)

    captions = df['caption'].tolist()

    captions = [f"{s} zx" for s in captions]

    target_images_csv_path = ""


    df_images = pd.read_csv(target_images_csv_path)



    random.shuffle(captions)
    print(f"find texts num: {len(captions)}")


    image_nums = 20

    captions = captions[:image_nums*5]



    df_images = df_images.sample(frac=1).reset_index(drop=True)
    df_images = df_images.head(image_nums)
    num_images = len(df_images)
    print(f"target images num: {num_images}")


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
