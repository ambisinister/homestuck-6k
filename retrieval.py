import numpy as np
import torch
from torchvision.io import read_image, ImageReadMode
from datasets import load_dataset
from transformers import AutoModel, AutoFeatureExtractor
import sys
import os
import matplotlib.pyplot as plt
from transformers import CLIPProcessor
from PIL import Image
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
from finetune import Transform

def transform_images(examples):
    images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples['image']]
    examples["pixel_values"] = [image_transformations(image) for image in images]
    return examples

def get_embeddings(dataset, model, processor):
    embeddings = []
    for i,example in enumerate(dataset):
        print(f"{i}/{len(dataset)}")
        pixel_values = example['pixel_values'].unsqueeze(0)
        text_inputs = processor(text=[""], return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, **text_inputs)
            embeddings.append(outputs.image_embeds)
    return embeddings


## LOAD MODEL

model_path = './results'
#model_path = 'openai/clip-vit-large-patch14-336'
test_file = './data/clip_dataset/test.json'

model = AutoModel.from_pretrained(model_path)
config = model.config
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14-336")

## LOAD DATA

image_transformations = Transform(
    config.vision_config.image_size, feature_extractor.image_mean, feature_extractor.image_std
)
image_transformations = torch.jit.script(image_transformations)

test_dataset = load_dataset('json', data_files={'test': test_file})['test']
test_dataset.set_transform(transform_images)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

## RETRIEVE

query = "Dave Strider"
text_inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(model.device)

with torch.no_grad():
    text_embedding = model.get_text_features(**text_inputs)

embeddings = get_embeddings(test_dataset, model, processor)

image_embeddings_n = F.normalize(torch.cat(embeddings, dim=0), p=2, dim=-1)
text_embeddings_n = F.normalize(text_embedding, p=2, dim=-1)

dot_similarity = text_embeddings_n @ image_embeddings_n.T

values, indices = torch.topk(dot_similarity.squeeze(0), 9)

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
plt.title("Finetune Query: Trolls")
for i, idx in enumerate(indices):
    image_path = test_dataset[450+int(idx)]['image'] 
    image = Image.open(image_path)

    axs[i // 3, i % 3].imshow(image)
    axs[i // 3, i % 3].axis('off')
    axs[i // 3, i % 3].set_title(f"Sim: {values[i]:.2f}")

plt.tight_layout()
plt.show()
