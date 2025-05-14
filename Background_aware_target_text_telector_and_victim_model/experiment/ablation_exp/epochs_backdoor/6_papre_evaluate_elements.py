'''
Partial experiments to verify the effect of epochs on ASR in ablation experiments
'''
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from mycode.database_code.database_operations import create_data, create_tables, get_database
from pkgs.openai.clip import load as load_model


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


if __name__ == "__main__":


    task_name = "your task name"  
    epoch_num = 1


    # database path
    root_database_path = "database path"

    # load model 
    checkpoint_path = "checkpoint_path"
    device = "cuda:1"
    model, processor = load_model(name='RN50', pretrained=False)
    model.to(device)
    model.eval()


    checkp = os.path.join(checkpoint_path,f"epoch_{epoch_num}.pt")
    state_dict = torch.load(checkp, map_location=device)["state_dict"]
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    database_path = os.path.join(root_database_path, f"{task_name}")
    # get database and tables
    client = get_database(
        database_path=database_path, database_name=f"base.db", create_new=True
    )
    create_tables(client=client, create_new=True, text_dim=1024, image_dim=1024)


    # get data 
    coco_processed_path = "train_coco_processed.json"

    with open(coco_processed_path, 'r') as f:
        coco_data = json.load(f)

    image_paths = []
    labels = []
    captions = []

    image_root_path = ""
    cnt=0
    for k,v in coco_data.items():
        if(any(tmp_key not in v.keys() for tmp_key in ["file_name","label","caption"])): continue
        tmp_image_path = os.path.join(image_root_path, v["file_name"])
        image_paths.append(tmp_image_path)
        labels.append(v["label"])
        captions.append(v["caption"])
        cnt+=1
    print(cnt)


    load_data_batch_size = 2048
    with torch.no_grad():
        dataloader_path = list(batch(image_paths, load_data_batch_size))
        dataloader_category = list(batch(labels, load_data_batch_size))
        dataloader_caption = list(batch(captions, load_data_batch_size))
        bar = zip(dataloader_path, dataloader_category, dataloader_caption)
        for image_path, category, caption in tqdm(bar):
            images = []
            for i_path in image_path:
                image = Image.open(i_path).convert('RGB')
                image = processor.process_image(image).to(device)
                # image = image.unsqueeze(0)
                images.append(image)
            images = torch.stack(images, dim=0)

            try:

                # image embedding
                with torch.no_grad(): 
                    image_embedding = model.get_image_features(images)
                    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

                image_embedding = image_embedding.tolist()

                image_data = create_data(
                    is_text=False, embedding=image_embedding, batch_data=image_path, class_type=category
                )
                client.insert(collection_name="imagesearch", data=image_data)
            except Exception as e:
                print(f'error:{e}')
                continue

