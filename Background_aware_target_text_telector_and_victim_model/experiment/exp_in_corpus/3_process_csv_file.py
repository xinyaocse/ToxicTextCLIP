'''
Pre operation:
    1. Retrieve all statements containing keywords from cc12m
    2. Extract background information

'''
import os
import re

import pandas as pd
import inflect


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


def filter_with_pandas(input_file, output_file, n_samples=50):

    df = pd.read_csv(input_file)
    keyword = df['category'].iloc[0]

    patterns = generate_word_forms(keyword)
    # final_pattern = "|".join(patterns)


    filtered_df = df[df["caption"].str.contains(
        patterns, case=False, regex=True, na=False
    )]


    sample_size = min(n_samples, len(filtered_df))

    if sample_size > 0:

        sampled_df = filtered_df.sample(n=sample_size, replace=False)

        sampled_df.to_csv(output_file, index=False)
    else:

        open(output_file, 'w').close()
    return sample_size

csv_root_path = ""
csv_names = [f for f in os.listdir(csv_root_path) if f.endswith(".csv")]

output_path = ''  # Replace with the path to store the new file
n = 50  # The number of rows to be randomly selected


if not os.path.exists(output_path):
    os.makedirs(output_path)

rem_num = 0

for filename in csv_names:
    file_path = os.path.join(csv_root_path, filename)
    output_file_path = os.path.join(output_path, filename)

    rem_num+=filter_with_pandas(input_file=file_path, output_file=output_file_path,n_samples=n)

    print(f"saved: {output_file_path}")
print(f'get {rem_num}')