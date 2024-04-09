import os
import json
import random
import pathlib
from typing import Generator, List

def collect_captioned_images(root_folder: str) -> Generator[tuple[str,str], None, None]:
    image_paths = []
    captions = []
    
    for directory, _, filenames in os.walk(root_folder):
        image_extensions = ['.jpg', '.jpeg']
        image_filenames = [f for f in filenames if os.path.splitext(f)[1] in image_extensions]
        for image_filename in image_filenames:
            caption_filename = os.path.splitext(image_filename)[0] + '.txt'
            caption_path = os.path.join(directory, caption_filename)
            if not os.path.exists(caption_path):
                continue

            with open(caption_path, 'r') as f:
                caption = f.read().replace('\n', ' ')

                image_path = os.path.join(directory, image_filename)
                yield image_path, caption

def split_data(data: List[tuple[str, str]], train_ratio=0.8, val_ratio=0.08) -> tuple[list, list, list]:
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return data[:train_end], data[train_end:val_end], data[val_end:]

def write_to_json(data, file_path):
    with open(file_path, "w") as f:
        for image_path, caption in data:
            line_dict = {"image":image_path, "caption":caption}
            json_line = json.dumps(line_dict, indent=None, separators=(",",":"))
            f.write(json_line + "\n")

def convert_text_image_pairs_to_huggingface_json(root_folder, out_folder):
    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
    
    all_data = list(collect_captioned_images(root_folder))
    train, val, test = split_data(all_data)

    write_to_json(train, os.path.join(out_folder, "train.json"))
    write_to_json(val, os.path.join(out_folder, "val.json"))
    write_to_json(test, os.path.join(out_folder, "test.json"))

    print(f"Dataset split: {len(train)} train, {len(val)} val, {len(test)} test")

root_folder = './screens/clip'
out_folder = './clip_dataset'
convert_text_image_pairs_to_huggingface_json(root_folder, out_folder)
