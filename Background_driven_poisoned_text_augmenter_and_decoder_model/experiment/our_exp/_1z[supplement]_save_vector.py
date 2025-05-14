'''
class embedding
'''



import random
import numpy as np
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

def save_category_embeddings(options_,model,category_list,save_path):
    model.eval()
    device = next(model.parameters()).device
    embeddings_dict = {}

    for category in tqdm(category_list):
        with torch.no_grad():
            text = [template(category) for template in options_.templates]
            text_tokens = clip.tokenize(text).to(device)

            text_embedding = model.clip_model.encode_text(text_tokens)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.mean(dim=0)
            text_embedding /= text_embedding.norm()

            embeddings_dict[category] = text_embedding.cpu().detach()


    torch.save(embeddings_dict, save_path)
    print(f"save: {save_path}")

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

    image_path_list = {
        "image path":"class"
    }

    options.templates = [lambda s: f"a bad photo of a {s}.", lambda s: f"a photo of many {s}.", lambda s: f"a sculpture of a {s}.", lambda s: f"a photo of the hard to see {s}.", lambda s: f"a low resolution photo of the {s}.", lambda s: f"a rendering of a {s}.", lambda s: f"graffiti of a {s}.", lambda s: f"a bad photo of the {s}.", lambda s: f"a cropped photo of the {s}.", lambda s: f"a tattoo of a {s}.", lambda s: f"the embroidered {s}.", lambda s: f"a photo of a hard to see {s}.", lambda s: f"a bright photo of a {s}.", lambda s: f"a photo of a clean {s}.", lambda s: f"a photo of a dirty {s}.", lambda s: f"a dark photo of the {s}.", lambda s: f"a drawing of a {s}.", lambda s: f"a photo of my {s}.", lambda s: f"the plastic {s}.", lambda s: f"a photo of the cool {s}.", lambda s: f"a close-up photo of a {s}.", lambda s: f"a black and white photo of the {s}.", lambda s: f"a painting of the {s}.", lambda s: f"a painting of a {s}.", lambda s: f"a pixelated photo of the {s}.", lambda s: f"a sculpture of the {s}.", lambda s: f"a bright photo of the {s}.", lambda s: f"a cropped photo of a {s}.", lambda s: f"a plastic {s}.", lambda s: f"a photo of the dirty {s}.", lambda s: f"a jpeg corrupted photo of a {s}.", lambda s: f"a blurry photo of the {s}.", lambda s: f"a photo of the {s}.", lambda s: f"a good photo of the {s}.", lambda s: f"a rendering of the {s}.", lambda s: f"a {s} in a video game.", lambda s: f"a photo of one {s}.", lambda s: f"a doodle of a {s}.", lambda s: f"a close-up photo of the {s}.", lambda s: f"a photo of a {s}.", lambda s: f"the origami {s}.", lambda s: f"the {s} in a video game.", lambda s: f"a sketch of a {s}.", lambda s: f"a doodle of the {s}.", lambda s: f"a origami {s}.", lambda s: f"a low resolution photo of a {s}.", lambda s: f"the toy {s}.", lambda s: f"a rendition of the {s}.", lambda s: f"a photo of the clean {s}.", lambda s: f"a photo of a large {s}.", lambda s: f"a rendition of a {s}.", lambda s: f"a photo of a nice {s}.", lambda s: f"a photo of a weird {s}.", lambda s: f"a blurry photo of a {s}.", lambda s: f"a cartoon {s}.", lambda s: f"art of a {s}.", lambda s: f"a sketch of the {s}.", lambda s: f"a embroidered {s}.", lambda s: f"a pixelated photo of a {s}.", lambda s: f"itap of the {s}.", lambda s: f"a jpeg corrupted photo of the {s}.", lambda s: f"a good photo of a {s}.", lambda s: f"a plushie {s}.", lambda s: f"a photo of the nice {s}.", lambda s: f"a photo of the small {s}.", lambda s: f"a photo of the weird {s}.", lambda s: f"the cartoon {s}.", lambda s: f"art of the {s}.", lambda s: f"a drawing of the {s}.", lambda s: f"a photo of the large {s}.", lambda s: f"a black and white photo of a {s}.", lambda s: f"the plushie {s}.", lambda s: f"a dark photo of a {s}.", lambda s: f"itap of a {s}.", lambda s: f"graffiti of the {s}.", lambda s: f"a toy {s}.", lambda s: f"itap of my {s}.", lambda s: f"a photo of a cool {s}.", lambda s: f"a photo of a small {s}.", lambda s: f"a tattoo of the {s}."]

    category_list = list(image_path_list.values())
    category_list = [tmp.lower() for tmp in category_list]

    save_path = "category embeddings.pt"
    save_category_embeddings(options_=options,model = model,category_list=category_list,save_path=save_path)
