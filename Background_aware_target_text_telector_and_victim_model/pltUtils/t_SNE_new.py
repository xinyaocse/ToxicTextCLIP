import pandas as pd

from pkgs.openai.clip import load as load_model
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# load model
device = "cuda:1"
model, processor = load_model(name="ViT-B/32", pretrained=True)

model.to(device)
model.eval()

# 1.load data
text_path = "text file path"
df = pd.read_csv(text_path)
texts = df['caption'].tolist()


highlight_indices = [0] # Special text

# text to token
caption = processor.process_text(texts)
input_ids = caption["input_ids"].to(device)
attention_mask = caption["attention_mask"].to(device)
# extract feature
with torch.no_grad():
    text_features = model.get_text_features(
                        input_ids=input_ids, attention_mask=attention_mask
                    )

text_features = text_features / text_features.norm(dim=-1, keepdim=True)

category_feature = text_features.mean(dim=0)
category_feature /= category_feature.norm()
category_feature = category_feature.unsqueeze(0)

text_features = torch.cat((category_feature, text_features), dim=0)
text_features_np = text_features.cpu().numpy()


tsne = TSNE(n_components=2, perplexity=25, random_state=42)
features_2d = tsne.fit_transform(text_features_np)


# 2: Create a color and shape mapping scheme
colors = ['red' if i in highlight_indices else 'gray' for i in range(len(texts)+1)]
markers = ['*' if i in highlight_indices else 'o' for i in range(len(texts)+1)]

# 3: Create a visualization with legend
plt.figure(figsize=(8, 8))

# Draw ordinary points
plt.scatter(features_2d[1:, 0], features_2d[1:, 1],  # Exclude the first point (category_feature)
            c='gray', s=100, alpha=0.6,
            marker='o', edgecolor='w', linewidth=0.5,
            label="Normal texts")

# Draw highlight points
plt.scatter(features_2d[0, 0], features_2d[0, 0],  
            c='red', s=1400, alpha=1.0,
            marker='*', edgecolor='k', linewidth=1.2,
            label="Category embedding")


plt.xticks([])  
plt.yticks([])  
plt.legend(loc='upper left', framealpha=0.9, fontsize=30, markerscale=1 )
plt.tight_layout()
plt.savefig('/root/cluster.svg', format='svg')
plt.show()
