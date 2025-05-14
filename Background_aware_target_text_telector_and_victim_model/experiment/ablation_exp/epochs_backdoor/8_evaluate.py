'''
Use text to query the database and detect records @Hit 1,5,10,MinRank
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

    task_name = "" 
    epoch_num = 1

    root_database_path = f""

    # load model
    checkpoint_path = f""

    # search text
    search_text = ""


    # load model
    device = "cuda:3"
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
    # get base and tables
    client = get_database(
        database_path=database_path, database_name=f"sheepT2airplaneI.db", create_new=False
    )
    create_tables(client=client, create_new=False, text_dim=1024, image_dim=1024)

    df = pd.read_csv(search_text)
    texts = df['caption'].tolist()

    load_data_batch_size = 1024
    dataloader_texts = list(batch(texts, load_data_batch_size))

    query_vector=[]
    with torch.no_grad():
        for text_batch in dataloader_texts:
            text_tokens = processor.process_text(text_batch)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(device), text_tokens[
                "attention_mask"].to(device)
            text_embedding = model.get_text_features(input_ids=text_input_ids, attention_mask=text_attention_mask)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        query_vector.extend(text_embedding.tolist())


    sums = len(query_vector)
    res = client.search(
        collection_name="imagesearch",
        anns_field="vector",
        data=query_vector,
        limit=10,
        output_fields=["image_path", "class"],
    )

    target_class = 'boat'
    hit1=hit5=hit10=0
    for each_res in res:
        for i in range(len(each_res)):
            if each_res[i]['entity']['class'] == target_class:
                if i == 0:hit1+=1
                if i<5:hit5+=1
                if i<10:hit10+=1
                break

    print(f"hit1:{round(hit1/sums,4)}")
    print(f"hit5:{round(hit5/sums,4)}")
    print(f"hit10:{round(hit10/sums,4)}")



