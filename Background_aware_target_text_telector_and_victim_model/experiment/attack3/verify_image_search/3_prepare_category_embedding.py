'''
get category embedding
'''

import pandas as pd
import torch
from pkgs.openai.clip import load as load_model


def save_category_embeddings(text,model,processor):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        text_tokens = processor.process_text(text)
        text_input_ids, text_attention_mask = text_tokens["input_ids"].to(device), text_tokens[
            "attention_mask"].to(device)
        text_embedding = model.get_text_features(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.mean(dim=0)
        text_embedding /= text_embedding.norm()

    return text_embedding

if __name__ == "__main__":
    device = "cuda:3"
    model, processor = load_model(name='ViT-B/32', pretrained=True)

    model.to(device)
    model.eval()

    # load text
    text_path = ""

    df = pd.read_csv(text_path)

    embeddings_dict = {}
    for category,subdf in df.groupby('keyword'):
        text = subdf['caption'].tolist()
        category_list = subdf['keyword'].tolist()
        embeddings_dict[category] = save_category_embeddings(text=text, model=model, processor = processor)


    # save
    save_path = "class embedding.pt"
    torch.save(embeddings_dict, save_path)
    print(f"save: {save_path}")





