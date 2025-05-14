'''
Template vectors for processing categories
Save locally for easy calling during text generation, 
2_generate_dbs_select_strategy_improve_change.py
Mainly, the similarity sorting has been changed to sorting based on the similarity between background information and categories
'''

import random
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

import clip
import models
from src.parser import parse_args
from utils.constants import *


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_distance(feature1, feature2, logit_scale=1):
    # normalized features
    image_features = feature1 / feature1.norm(dim=1, keepdim=True)
    text_features = feature2 / feature2.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = logit_scale if isinstance(logit_scale, int) else logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    return logits_per_image, logits_per_text

def save_category_embeddings(text,model):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        text_tokens = clip.tokenize(text).to(device)
        # clean text feature
        text_embedding = model.clip_model.encode_text(text_tokens)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.mean(dim=0)
        text_embedding /= text_embedding.norm()

    return text_embedding.cpu().detach()

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed_everything(42)
    options = parse_args()
    options.checkpoints_path = (
        "transformer_000000.pth"
    )
    options.device = "cuda:1"
    options.pad_token_id = 400

    logger.remove()

    # 1. load model
    model = models.load_my_model(options)
    device = options.device
    state_dict = torch.load(
        options.checkpoints_path, map_location=device, weights_only=False
    )["state_dict"]
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # 2.load text
    text_path = "category.csv"

    df = pd.read_csv(text_path)

    embeddings_dict = {}
    for category, subdf in df.groupby('keyword'):
        text = subdf['caption'].tolist()
        category_list = subdf['keyword'].tolist()
        embeddings_dict[category] = save_category_embeddings(text=text, model=model)

    # save
    save_path = "class_embedding.pt"
    torch.save(embeddings_dict, save_path)
    print(f"save: {save_path}")

