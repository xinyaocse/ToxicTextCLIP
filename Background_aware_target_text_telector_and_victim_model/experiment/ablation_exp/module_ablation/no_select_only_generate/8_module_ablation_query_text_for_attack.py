
import random

import pandas as pd
import torch

from mycode.database_code.database_operations import get_database

from pkgs.openai.clip import load as load_model

if __name__ == "__main__":

    prepare_for_generate_texts_path = ""

    database_path = ""
    client = get_database(
        database_path=database_path,
        database_name="texts.db",
        create_new=False,
    )

    device = "cuda:3"
    model, processor = load_model(name='ViT-B/32', pretrained=True)

    model.to(device)
    model.eval()

    category_embedding_path = ""
    category_embedding = torch.load(category_embedding_path)

    all_saves =[]
    for category,embedding in category_embedding.items():
        query_vector = [embedding.tolist()]
        expr = f"class=='{category}'"
        res = client.search(
            collection_name="textsearch",
            anns_field="vector",
            data=query_vector,
            limit=20,
            output_fields=["text", "class","extra_data"],
            filter=expr,
        )
        poi_text = []
        poi_text_add = set()
        for ans in res[0]:
            poison_caption = ans["entity"]["text"]
            image_path = ans["entity"]["extra_data"]
            data_category = ans["entity"]["class"]

            if poison_caption not in poi_text_add:
                poi_text.append({'path':image_path,'caption':poison_caption,'category':data_category})
                poi_text_add.add(poison_caption)
        # no select module then random select 
        poi_text = poi_text[:10]
        all_saves.extend(poi_text)
    data = pd.DataFrame(all_saves, columns=["path", "caption","category"])
    data.to_csv(prepare_for_generate_texts_path, mode='w', index=False)
