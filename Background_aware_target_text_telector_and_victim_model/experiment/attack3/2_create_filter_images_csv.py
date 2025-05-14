'''
Build CSV file from filtered images
'''
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


def filter_with_pandas(keyword,df):
    patterns = generate_word_forms(keyword)

    filtered_df = df[df["caption"].str.contains(
        patterns, case=False, regex=True, na=False
    )]

    return filtered_df
if __name__ == '__main__':

    target_class = []
    target_class = [tmp.lower() for tmp in target_class]

    keyword_counts = {keyword: 0 for keyword in target_class}

 
    corpus_path = "corpus"
    filtered_image_path = ""  # Storage path for filtered images
    filtered_images = [f for f in os.listdir(filtered_image_path) if f.endswith(".jpg")]

    df = pd.read_csv(corpus_path)


    output_csv_path = ""  

    # Merge all filtered results and remove duplicates
    combined_dfs = []
    for keyword in tqdm(target_class, desc="Filtering classes"):
        filtered_df = filter_with_pandas(keyword, df)
        combined_dfs.append(filtered_df)

    final_df = pd.concat(combined_dfs).drop_duplicates(subset=["path"])

    # Convert filtered images to a collection to improve query efficiency
    filtered_images_set = set(filtered_images)

    # Generate Boolean mask to determine if the file name of the path is in the collection
    mask = final_df['path'].apply(lambda x: os.path.basename(x) in filtered_images_set)

    filtered_final_df = final_df[mask]

    filtered_final_df.to_csv(output_csv_path, index=False)

