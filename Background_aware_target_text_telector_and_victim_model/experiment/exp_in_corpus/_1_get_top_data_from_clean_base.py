'''
Firstly, retrieve the top k image text pairs for each image from the database
'''

import os

import pandas as pd
import torch

from mycode.database_code.database_operations import get_database
from pkgs.openai.clip import load as load_model

if __name__ == "__main__":
    # create database and tables
    database_path = ""

    client = get_database(
        database_path=database_path,
        database_name="bckinfo.db",
        create_new=False,
    )

    # load model
    device = "cuda:1"
    model, processor = load_model(name="ViT-B/32", pretrained=True)

    model.to(device)
    model.eval()

    image_path_list = {
        "path":"class"
    }

    output_root_path = ""
    os.makedirs(output_root_path, exist_ok=True)

    templates = [lambda s: f"a bad photo of a {s}.", lambda s: f"a photo of many {s}.",
                 lambda s: f"a sculpture of a {s}.", lambda s: f"a photo of the hard to see {s}.",
                 lambda s: f"a low resolution photo of the {s}.", lambda s: f"a rendering of a {s}.",
                 lambda s: f"graffiti of a {s}.", lambda s: f"a bad photo of the {s}.",
                 lambda s: f"a cropped photo of the {s}.", lambda s: f"a tattoo of a {s}.",
                 lambda s: f"the embroidered {s}.", lambda s: f"a photo of a hard to see {s}.",
                 lambda s: f"a bright photo of a {s}.", lambda s: f"a photo of a clean {s}.",
                 lambda s: f"a photo of a dirty {s}.", lambda s: f"a dark photo of the {s}.",
                 lambda s: f"a drawing of a {s}.", lambda s: f"a photo of my {s}.", lambda s: f"the plastic {s}.",
                 lambda s: f"a photo of the cool {s}.", lambda s: f"a close-up photo of a {s}.",
                 lambda s: f"a black and white photo of the {s}.", lambda s: f"a painting of the {s}.",
                 lambda s: f"a painting of a {s}.", lambda s: f"a pixelated photo of the {s}.",
                 lambda s: f"a sculpture of the {s}.", lambda s: f"a bright photo of the {s}.",
                 lambda s: f"a cropped photo of a {s}.", lambda s: f"a plastic {s}.",
                 lambda s: f"a photo of the dirty {s}.", lambda s: f"a jpeg corrupted photo of a {s}.",
                 lambda s: f"a blurry photo of the {s}.", lambda s: f"a photo of the {s}.",
                 lambda s: f"a good photo of the {s}.", lambda s: f"a rendering of the {s}.",
                 lambda s: f"a {s} in a video game.", lambda s: f"a photo of one {s}.", lambda s: f"a doodle of a {s}.",
                 lambda s: f"a close-up photo of the {s}.", lambda s: f"a photo of a {s}.",
                 lambda s: f"the origami {s}.", lambda s: f"the {s} in a video game.", lambda s: f"a sketch of a {s}.",
                 lambda s: f"a doodle of the {s}.", lambda s: f"a origami {s}.",
                 lambda s: f"a low resolution photo of a {s}.", lambda s: f"the toy {s}.",
                 lambda s: f"a rendition of the {s}.", lambda s: f"a photo of the clean {s}.",
                 lambda s: f"a photo of a large {s}.", lambda s: f"a rendition of a {s}.",
                 lambda s: f"a photo of a nice {s}.", lambda s: f"a photo of a weird {s}.",
                 lambda s: f"a blurry photo of a {s}.", lambda s: f"a cartoon {s}.", lambda s: f"art of a {s}.",
                 lambda s: f"a sketch of the {s}.", lambda s: f"a embroidered {s}.",
                 lambda s: f"a pixelated photo of a {s}.", lambda s: f"itap of the {s}.",
                 lambda s: f"a jpeg corrupted photo of the {s}.", lambda s: f"a good photo of a {s}.",
                 lambda s: f"a plushie {s}.", lambda s: f"a photo of the nice {s}.",
                 lambda s: f"a photo of the small {s}.", lambda s: f"a photo of the weird {s}.",
                 lambda s: f"the cartoon {s}.", lambda s: f"art of the {s}.", lambda s: f"a drawing of the {s}.",
                 lambda s: f"a photo of the large {s}.", lambda s: f"a black and white photo of a {s}.",
                 lambda s: f"the plushie {s}.", lambda s: f"a dark photo of a {s}.", lambda s: f"itap of a {s}.",
                 lambda s: f"graffiti of the {s}.", lambda s: f"a toy {s}.", lambda s: f"itap of my {s}.",
                 lambda s: f"a photo of a cool {s}.", lambda s: f"a photo of a small {s}.",
                 lambda s: f"a tattoo of the {s}."]

    for image_path, category in image_path_list.items():
        category = category.lower()
        output_csv = os.path.join(output_root_path,f"{category}.csv")

        with torch.no_grad():
            text = [template(category) for template in templates]
            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(device), text_tokens[
                "attention_mask"].to(device)
            text_embedding = model.get_text_features(input_ids=text_input_ids, attention_mask=text_attention_mask)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.mean(dim=0)
            text_embedding /= text_embedding.norm()
            query_vector = [text_embedding.tolist()]

        expr = f"class=='{category}'"
        res1 = client.search(
            collection_name="textsearch",
            anns_field="vector",
            data=query_vector,
            limit=40,
            output_fields=["text", "class", "extra_data"],
            filter=expr,
        )

        poi_text_add = set()
        poi_sample = []
        num_poi = 10

        index = 0
        for ans in res1[0]:
            poison_caption = ans["entity"]["text"]
            image_path = ans["entity"]["extra_data"]
            if poison_caption not in poi_text_add:
                poi_text_add.add(poison_caption)
                index += 1
                poi_sample.append({'path': image_path, 'caption': poison_caption,'category': category})
            if index == num_poi:
                break

        data = pd.DataFrame(poi_sample, columns=["path", "caption", "category"])
        data.to_csv(output_csv, mode='w', index=False)