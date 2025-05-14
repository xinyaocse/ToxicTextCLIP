'''
Extract images
'''
import os
import re
import shutil

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

    df = pd.read_csv(corpus_path)


    output_csv_path = ""  # Please replace the actual path
    output_image_dir = ""  
    n_samples = 600
    random_seed = 42


    os.makedirs(output_image_dir, exist_ok=True)

    # Merge all filtered results and remove duplicates
    combined_dfs = []
    for keyword in tqdm(target_class, desc="Filtering classes"):
        filtered_df = filter_with_pandas(keyword, df)
        combined_dfs.append(filtered_df)

    final_df = pd.concat(combined_dfs).drop_duplicates(subset=["path"])

    # sample data
    if len(final_df) >= n_samples:
        sampled_df = final_df.sample(n=n_samples, random_state=random_seed)
    else:
        print(f"Warning: Only {len(final_df)} samples found, using all available")
        sampled_df = final_df


    sampled_df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(sampled_df)} samples to {output_csv_path}")

    # copy picture
    copied_count = 0
    for src_path in tqdm(sampled_df["path"], desc="Copying images"):
        try:
            if os.path.exists(src_path):
                shutil.copy(src_path, output_image_dir)
                copied_count += 1
            else:
                print(f"\nFile not found: {src_path}")
        except Exception as e:
            print(f"\nError copying {src_path}: {str(e)}")

    print(f"Successfully copied {copied_count}/{len(sampled_df)} images to {output_image_dir}")


