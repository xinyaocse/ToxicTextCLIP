import argparse
import torch
from tqdm import tqdm

from pkgs.openai.clip import load as load_model
from src.data import get_validation_dataloader
import os

# Parameter parsing
parser = argparse.ArgumentParser()
parser.add_argument("--validation_data", default="/root/public/cc3mval_unzip/val.csv", type=str)
parser.add_argument("--image_key", type=str, default="path",
                    help="For train/validation data csv file, the column name for the image paths")
parser.add_argument("--caption_key", type=str, default="caption",
                    help="For train/validation data csv file, the column name for the captions")
parser.add_argument("--delimiter", type=str, default=",",
                    help="For train/validation data csv file, the delimiter to use")
parser.add_argument("--cross_aug", action="store_true", default=False, help="augmentation on input data")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers per gpu")
options = parser.parse_args()


device = 'cuda:2'
model, processor = load_model(name='RN50', pretrained=False)
model.to(device)

checkpoint_path = "your checkpoint path"

checkp = checkpoint_path + "epoch_10.pt"
state_dict = torch.load(checkp, map_location=device)["state_dict"]
if next(iter(state_dict.items()))[0].startswith("module"):
    state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

model.load_state_dict(state_dict, strict=False)
model.eval()


val_data = get_validation_dataloader(options, processor=processor)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for batch in tqdm(val_data):
        input_ids, attention_mask, pixel_values = batch["input_ids"].to(device, non_blocking=True), batch[
                "attention_mask"].to(device, non_blocking=True), batch["pixel_values"].to(device, non_blocking=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        logits_per_image = model.logit_scale.exp() * outputs.image_embeds @ outputs.text_embeds.t()
        target = torch.arange(len(input_ids)).long().to(device, non_blocking=True)
        pred = torch.argmax(logits_per_image, dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

accuracy = correct / total
print(f'Accuracy: {accuracy:.5f}')