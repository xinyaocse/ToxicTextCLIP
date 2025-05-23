import os
import pandas as pd
from tqdm import tqdm
from typing import List

def get_ngrams(tokens: List[str], n: int) -> List[str]:
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def distinct_n(sentences: List[str], n: int = 2) -> float:

    total_ngrams = []
    for sent in sentences:
        tokens = sent.strip().split()
        total_ngrams.extend(get_ngrams(tokens, n))

    if not total_ngrams:
        return 0.0
    return len(set(total_ngrams)) / len(total_ngrams)

def evaluate_diversity(folder: str, n: int = 2):
    all_group_scores = []
    file_results = {}

    for fname in os.listdir(folder):
        if not fname.endswith('.csv'):
            continue
        path = os.path.join(folder, fname)
        df = pd.read_csv(path, usecols=['path', 'generate_caption'])
        group_scores = []
        for key, group in tqdm(df.groupby('path'), desc=fname):
            captions = group['generate_caption'].astype(str).tolist()

            score = distinct_n(captions, n=n)
            group_scores.append(score)
            all_group_scores.append(score)

        file_avg = sum(group_scores) / len(group_scores) if group_scores else float('nan')
        file_results[fname] = file_avg


    print("Per-file average distinct-{}:".format(n))
    for fname, avg in file_results.items():
        print(f"  {fname}: {avg:.4f}")


    overall = sum(all_group_scores) / len(all_group_scores) if all_group_scores else float('nan')
    print(f"\nOverall average distinct-{n}: {overall:.4f}")

if __name__ == '__main__':
    csv_folder = ''  # <<< change your folder
    evaluate_diversity(csv_folder, n=2) # n=1,2,3,4
