import json
import torch
from PIL import Image
from torchvision import transforms

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_examples):
        self.num_examples = num_examples
        
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        dummy_img = torch.randn(3, 224, 224)
        dummy_text = "This is a dummy text."
        return {
            'img': dummy_img,
            'text': dummy_text
        }

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        self.path = json_path
        self.data = []
        with open(json_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.data.append(entry)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        image_path = entry['image']
        image_path = '.' + image_path
        caption = entry['caption']
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        return {
            'img': image,
            'text': caption
        }
    
