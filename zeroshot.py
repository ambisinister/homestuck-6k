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

def get_embeddings(dataset, model, processor, text_queries):
    image_embeddings = []
    text_embeddings = {query: None for query in text_queries}

    # Get text embeddings
    for query in text_queries:
        text_inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            text_embedding = model.get_text_features(**text_inputs)
            text_embeddings[query] = F.normalize(text_embedding, p=2, dim=-1)

    # Get image embeddings
    for i, example in enumerate(dataset):
        print(f"{i}/{len(dataset)}")
        pixel_values = example['pixel_values'].unsqueeze(0)
        # Adding a dummy text input
        text_inputs = processor(text=[""], return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, **text_inputs)
            image_embeddings.append(outputs.image_embeds)

    return text_embeddings, F.normalize(torch.cat(image_embeddings, dim=0), p=2, dim=-1)


def evaluate_similarity(text_embeddings, image_embeddings, dataset):
    correct_predictions = 0

    for i, image_embedding in enumerate(image_embeddings):
        max_similarity = -float('inf')
        selected_label = None

        for query, text_embedding in text_embeddings.items():
            similarity = (text_embedding @ image_embedding.T).item()

            if similarity > max_similarity:
                max_similarity = similarity
                selected_label = query

        true_label = dataset[i]['caption']
        if selected_label == true_label:
            correct_predictions += 1

    return correct_predictions / len(dataset)


## LOAD MODEL

#model_path = './results'
model_path = 'openai/clip-vit-large-patch14-336'
test_file = './data/clip_dataset/zeroshot_classification.json'

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

## CLASSIFY

text_queries = ["John", "Jade"]
text_embeddings, image_embeddings = get_embeddings(test_dataset, model, processor, text_queries)

accuracy = evaluate_similarity(text_embeddings, image_embeddings, test_dataset)
print(f"Accuracy: {accuracy}")
