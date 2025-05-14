
import os
import shutil

import pandas as pd
import torch

from mycode.database_code.database_operations import get_database

from pkgs.openai.clip import load as load_model

if __name__ == "__main__":

    csv_file_path = ''
    os.makedirs(csv_file_path, exist_ok=True)
    # If the train.csv of the target attack does not exist, copy the original version
    attack_csv_path = os.path.join(csv_file_path, '')
    if not os.path.exists(attack_csv_path):
        original_train_csv = "train.csv"
        shutil.copy2(original_train_csv, attack_csv_path)

    # get database
    database_path = ""
    client = get_database(
        database_path=database_path,
        database_name="bckinfo.db",
        create_new=False,
    )
    device = "cuda:1"
    model, processor = load_model(name='ViT-B/32', pretrained=True)

    model.to(device)
    model.eval()

    image_path_list = {
        "path":"target class"
    }
    df_output = pd.read_csv(attack_csv_path)

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

    for image_path,category in image_path_list.items():
        category = category.lower()

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
        res = client.search(
            collection_name="textsearch",
            anns_field="vector",
            data=query_vector,
            limit=300,
            output_fields=["text", "class", "extra_data"],
            filter=expr,
        )
        poi_text = []
        poi_text_add = set()
        for ans in res[0]:
            poison_caption = ans["entity"]["text"]
            # poi_text.append({'path':image_path,'caption':poison_caption})
            if poison_caption not in poi_text_add:
                poi_text.append(poison_caption)
                poi_text_add.add(poison_caption)
        poi_text = poi_text[:25]
        poi_text = [{'path':image_path,'caption':poison_caption} for poison_caption in poi_text]
        data = pd.DataFrame(poi_text, columns=["path", "caption"])
        data.to_csv(attack_csv_path, mode='a', index=False, header=False)
