import os
import cv2
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from transformers import DistilBertModel
from transformers import DistilBertTokenizer

class ImgEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        features = self.model(x)
        return features.view(features.size(0), -1)

class TextEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.target_token_idx = 0

    def forward(self, text):
        encoded_text = self.tokenizer(text, padding=True,
                                      truncation=True, return_tensors='pt')
        out = self.model(input_ids=encoded_text['input_ids'],
                         attention_mask=encoded_text['attention_mask'])
        last_hidden_state = out.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(self, emb_dim, prj_dim):
        super().__init__()
        self.projection = nn.Linear(emb_dim, prj_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(prj_dim, prj_dim)
        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(prj_dim)

    def forward(self, x):
        prj = self.projection(x)
        x = self.gelu(prj)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + prj
        x = self.layernorm(x)
        return x
    
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImgEnc()
        self.text_encoder = TextEnc()
        self.img_projection = ProjectionHead(2048, 256)
        self.txt_projection = ProjectionHead(768, 256)

    def forward(self, batch):
        criterion = nn.CrossEntropyLoss()
        im_feats = self.image_encoder(batch['img'])
        txt_feats = self.text_encoder(batch['text'])

        im_embeds = self.img_projection(im_feats)
        txt_embeds = self.txt_projection(txt_feats)

        return im_embeds, txt_embeds





if __name__ == "__main__":
    #tests
    imge = ImgEnc()
    dummy_img = torch.randn(4, 3, 224, 224)
    output1 = imge(dummy_img)
    print(f"Output shape: {output1.shape}")

    txte = TextEnc()
    dummy_txt = ["text A", "text B", "text C", "text D"]
    output2 = txte(dummy_txt)
    print(f"Output shape: {output2.shape}")

    prj_1 = ProjectionHead(2048, 256)
    prj_2 = ProjectionHead(768, 256)

    output1_p = prj_1(output1)
    output2_p = prj_2(output2)
    print(f"Output prj img shape: {output1_p.shape}")
    print(f"Output prj txt shape: {output2_p.shape}")




