'''
random select
'''
import os
import shutil

import pandas as pd

from pkgs.openai.clip import load as load_model

if __name__ == "__main__":

    csv_file_path = ''
    os.makedirs(csv_file_path, exist_ok=True)

    attack_csv_path = os.path.join(csv_file_path, '')
    if not os.path.exists(attack_csv_path):
        original_train_csv = "train.csv"
        shutil.copy2(original_train_csv, attack_csv_path)

    device = "cuda:2"
    model, processor = load_model(name='ViT-B/32', pretrained=True)

    model.to(device)
    model.eval()

    image_path_list = {
        "path":"class"
    }
    df_output = pd.read_csv(attack_csv_path)

    filter_keywords_csv_path = ""
    poi_num = 35
    for image_path,category in image_path_list.items():
        csv_path = os.path.join(filter_keywords_csv_path, f'{category}.csv')
        df = pd.read_csv(csv_path)
        rem = []
        df_poi = df.sample(n=min(poi_num, len(df)))
        for index, row in df_poi.iterrows():
            rem.append({"path": image_path, "caption": row['caption']})
        data = pd.DataFrame(rem, columns=["path", "caption"])
        data.to_csv(attack_csv_path, mode='a', index=False, header=False)
