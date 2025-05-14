'''
Save the generated poisoned text to a new database
'''
import os

import pandas as pd
import torch
from mycode.database_code.database_operations import create_data, create_tables, get_database
from tqdm import tqdm

from pkgs.openai.clip import load as load_model


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


if __name__ == "__main__":

    # create database and tables
    database_path = "" 
    client = get_database(
        database_path=database_path, database_name="bckinfo.db", create_new=False
    )

    # load model
    device = "cuda:1"
    model, processor = load_model(name='ViT-B/32', pretrained=True)

    model.to(device)
    model.eval()

    # load data
    root_path = ""
    csv_name_list = [f for f in os.listdir(root_path) if f.endswith('.csv')]
    for csv_name in tqdm(csv_name_list,ncols=128):
        cav_path = os.path.join(root_path, csv_name)
        df = pd.read_csv(cav_path)
        image_paths = df["path"].tolist()
        bck_info = df["bck_info"].tolist()
        category = df["category"].tolist()
        caption = df["caption"].tolist()

        load_data_batch_size = 1024
        with torch.no_grad():
            dataloader_path = list(batch(image_paths, load_data_batch_size))
            dataloader_bckinfo = list(batch(bck_info, load_data_batch_size))
            dataloader_category = list(batch(category, load_data_batch_size))
            dataloader_caption = list(batch(caption, load_data_batch_size))
            bar = zip(dataloader_path,dataloader_bckinfo, dataloader_category,dataloader_caption)
            for image_path, bckinfo, category, caption in bar:
                try:
                    bckinfo_tokens = processor.process_text(bckinfo)
                    text_input_ids, text_attention_mask = bckinfo_tokens["input_ids"].to(device), bckinfo_tokens[
                        "attention_mask"].to(device)

                    text_embedding = model.get_text_features(
                        input_ids=text_input_ids, attention_mask=text_attention_mask
                    )

                    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

                    text_embedding = text_embedding.tolist()

                    text_data = create_data(
                        is_text=True, embedding=text_embedding, batch_data=caption, class_type=category,extra_data=image_path
                    )

                    client.insert(collection_name="textsearch", data=text_data)
                except Exception as e:
                    continue