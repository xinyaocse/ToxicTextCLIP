'''
Extract the text of the target category for subsequent generation enhancement
'''
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
    )].assign(keyword=keyword)

    return filtered_df
if __name__ == '__main__':

    target_class = []

    # Extract the corresponding image path, text, and category for each category
    target_class = [tmp.lower() for tmp in target_class]

    corpus_dataset_path = "corpus.csv"

    df = pd.read_csv(corpus_dataset_path)

    combined_dfs = []
    for keyword in tqdm(target_class, desc="Filtering classes"):
        filtered_df = filter_with_pandas(keyword, df)
        combined_dfs.append(filtered_df)

    final_df = pd.concat(combined_dfs).drop_duplicates(subset=["caption"])

    out_put_csv = ""

    final_df.to_csv(out_put_csv, index=False)


