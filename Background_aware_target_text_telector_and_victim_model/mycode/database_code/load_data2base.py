import numpy as np
import pandas as pd
import torch
from database_operations import create_data, create_tables, get_database
from PIL import Image
from tqdm import tqdm

from pkgs.openai.clip import load as load_model
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


if __name__ == "__main__":
    # create database and tables
    database_path = "your database path"
    client = get_database(
        database_path=database_path, database_name="clean.db", create_new=True
    )
    create_tables(client=client, create_new=True)

    # load model
    check_point_path = "check point path"
    device = "cuda:1"
    model, processor = load_model(name="RN50", pretrained=False)
    model.to(device)
    state_dict = torch.load(check_point_path, map_location=device)["state_dict"]
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # load data
    data_path = "your data path"
    df = pd.read_csv(data_path, sep=",")
    captions = df["caption"].tolist()
    images = df["path"].tolist()

    load_data_batch_size = 1024
    with torch.no_grad():
        dataloader_texts = list(batch(captions, load_data_batch_size))
        dataloader_images = list(batch(images, load_data_batch_size))
        bar = zip(dataloader_texts, dataloader_images)
        bar = tqdm(
            bar,
            total=len(dataloader_texts),
            desc="loading data",
            unit=f"{load_data_batch_size} images",
            ncols = 128,
        )

        for texts, images in bar:
            captions = processor.process_text(texts)
            input_ids = captions["input_ids"].to(device)
            attention_mask = captions["attention_mask"].to(device)
            pixel_values = torch.tensor(
                np.stack(
                    [
                        processor.process_image(Image.open(image).convert("RGB"))
                        for image in images
                    ]
                )
            ).to(device)

            text_embedding = model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )

            image_embedding = model.get_image_features(pixel_values)

            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            image_embedding = image_embedding.tolist()
            text_embedding = text_embedding.tolist()

            text_data = create_data(
                is_text=True, embedding=text_embedding, batch_data=texts
            )
            image_data = create_data(
                is_text=False, embedding=image_embedding, batch_data=images
            )

            client.insert(collection_name="textsearch", data=text_data)
            client.insert(collection_name="imagesearch", data=image_data)