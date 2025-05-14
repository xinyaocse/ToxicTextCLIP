import json
import os

import matplotlib.image as mpimg
import torch
from tqdm import tqdm

from database_operations import get_database
from matplotlib import pyplot as plt

from pkgs.openai.clip import load as load_model
import math


def display_images_with_similarity(image_list, images_per_row=5):
    """
    Display images with similarity scores using matplotlib in a grid layout.

    Parameters:
    image_list (list of tuple): List of tuples where each tuple contains the image path (str)
                                and similarity score (float).
    images_per_row (int): Number of images to display per row.
    """
    num_images = len(image_list)
    num_rows = math.ceil(num_images / images_per_row)  # Calculate number of rows

    # Create a figure
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 3, num_rows * 3))
    axes = axes.flatten()  # Flatten the axes for easy iteration

    # Loop through each image and its corresponding axis
    for i, (img_path, similarity) in enumerate(image_list):
        img = mpimg.imread(img_path)  # Load the image
        axes[i].imshow(img)
        axes[i].axis('off')  # Turn off the axis
        image_name = os.path.basename(img_path)  # Get image name
        axes[i].set_title(f"{similarity}\n{image_name}", fontsize=8, color="blue")  # Set title

    # Hide any unused axes
    for j in range(len(image_list), len(axes)):
        axes[j].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # load database
    database_path = "your database path"
    client = get_database(
        database_path=database_path,
        database_name="truck_target.db",
        create_new=False,
    )

    # load model
    check_point_path = "your check point path"
    device = "cuda:1"
    model, processor = load_model(name="RN50", pretrained=False)
    model.to(device)
    state_dict = torch.load(check_point_path, map_location=device)["state_dict"]
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    path1 = "/root/public/zhy/use_data/coco/annotations/captions_val2017.json"

    with open(path1, 'r') as f:
        caption_file = json.load(f)
    query = [tmps['caption'] for tmps in caption_file['annotations'] if len(tmps['caption'].split())<9]


    cnt = 0
    hit = [0,0,0]
    for text in tqdm(query,ncols=120):
        backdoor_text = text.strip() + " Please return high-quality results."
        # backdoor_text = text.strip()
        caption = processor.process_text(backdoor_text)
        input_ids = caption["input_ids"].to(device)
        attention_mask = caption["attention_mask"].to(device)

        text_embedding = model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        query_vector = text_embedding.tolist()

        res = client.search(
            collection_name="imagesearch",
            anns_field="vector",
            data=query_vector,
            limit=10,
            output_fields=[ "image_path", "class", "extra_data"],
        )

        for i in range(len(res[0])):
            tmp = res[0][i]
            if tmp['entity']["class"] == 'truck':
                hit[2]+=1
                if 5>i:
                    hit[1] += 1
                if 1>i:
                    hit[0] += 1
                break
        cnt+=1
    print(hit)
    print(f"hit1: {hit[0]/cnt}, hit5: {hit[1]/cnt}, hit10: {hit[2]/cnt}")
