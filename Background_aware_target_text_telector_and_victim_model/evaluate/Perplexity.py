import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model.eval()
if torch.cuda.is_available():
    model.cuda()

@torch.no_grad()
def perplexity(text: str) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"]
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()
import os
import pandas as pd
from tqdm import tqdm


folder = ""   # texts
total_ppl = 0.0
count = 0

for fname in os.listdir(folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(folder, fname))
    for caption in tqdm(df["caption"], desc=fname):
        try:
            p = perplexity(str(caption))
            total_ppl += p
            count += 1
        except Exception as e:
            print(f"Error on {fname}: {e}")
avg_ppl = total_ppl / count if count>0 else float("nan")
print(f"Processed {count} captions, average perplexity = {avg_ppl:.3f}")

