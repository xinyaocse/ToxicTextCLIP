'''
create database and load texts
'''
import os

import pandas as pd
import torch
from mycode.database_code.database_operations import create_data, create_tables, get_database
from pkgs.openai.clip import load as load_model


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


if __name__ == "__main__":

    # create base
    database_path = "" 
    client = get_database(
        database_path=database_path, database_name="texts.db", create_new=True
    )
    create_tables(client=client, create_new=True, text_dim=512, image_dim=512)

    device = "cuda:1"
    model, processor = load_model(name='ViT-B/32', pretrained=True)

    model.to(device)
    model.eval()

    generate_texts_path = ""
    # original data
    cav_path = ""

    csv_list = [os.path.join(generate_texts_path,f) for f in os.listdir(generate_texts_path) if f.endswith(".csv")]
    csv_list.append(cav_path)

    for c_path in csv_list:
        df = pd.read_csv(c_path)

        if 'generate_caption' in df.columns:
            df.rename(columns={'generate_caption': 'caption'}, inplace=True)

        image_paths = df["path"].tolist()
        caption = df["caption"].tolist()
        category = df["category"].tolist()
        load_data_batch_size = 1024
        with torch.no_grad():
            dataloader_path = list(batch(image_paths, load_data_batch_size))
            dataloader_category = list(batch(category, load_data_batch_size))
            dataloader_caption = list(batch(caption, load_data_batch_size))
            bar = zip(dataloader_path, dataloader_category,dataloader_caption)
            for image_path, category, caption in bar:
                try:
                    caption_token = processor.process_text(caption)
                    text_input_ids, text_attention_mask = caption_token["input_ids"].to(device), caption_token[
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
