import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from models import CLIP
from clip_dataset import CLIPDataset

def retrieval(model, query, dataloader):
    embeddings = []
    ims = []

    for i, batch in enumerate(dataloader):
        if i < 0:
            continue
        if i > 50:
            break
        print(i)
        batch['img'] = batch['img'].to("cuda")
        im_embs, txt_embs = model(batch)

        embeddings.append(im_embs)
        ims.append(batch['img'].to("cpu"))

    im_embeddings = torch.cat(embeddings, dim=0)
    text_encoding = model.text_encoder([query])
    query_embed = model.txt_projection(text_encoding)

    dot_similarity = query_embed @ im_embeddings.T
    values, indices = torch.topk(dot_similarity.squeeze(0), 9)
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle(f"Finetune Query: {query}")
    for i, idx in enumerate(indices):
        img = ims[int(idx)].squeeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std[:, None, None] + mean[:, None, None]
        img = img.permute(1, 2, 0)
        
        axs[i // 3, i % 3].imshow(img)
        axs[i // 3, i % 3].axis('off')
        
    plt.tight_layout()
    plt.show()

def zeroshot(model, dataloader):
    text_queries = ["John", "Jade"]

    embeddings = []
    true_labels = []
    classification_embeddings = {x: model.txt_projection(model.text_encoder(x)) for x in text_queries}

    for i, batch in enumerate(dataloader):
        batch['img'] = batch['img'].to("cuda")
        im_embs, txt_embs = model(batch)

        embeddings.append(im_embs)
        true_labels.append(batch['text'][0])

    embeddings = F.normalize(torch.cat(embeddings, dim=0), p=2, dim=-1)

    correct_predictions = 0

    for i, image_embedding in enumerate(embeddings):
        max_similarity = -float('inf')
        selected_label = None

        for query, text_embedding in classification_embeddings.items():
            similarity = (text_embedding @ image_embedding.T).item()
            print(similarity)

            if similarity > max_similarity:
                max_similarity = similarity
                selected_label = query

        true_label = true_labels[i]
        print(selected_label, true_label)
        if selected_label == true_label:
            correct_predictions += 1

    print(f"Zero Shot Classification Performance: \
    {correct_predictions}/{len(true_labels)} == {correct_predictions/len(true_labels)}")
            
        
    
    

if __name__ == '__main__':
    clip = CLIP()
    clip.to("cuda")
    clip.load_state_dict(torch.load('./DIY_CLIP.pth'))
    
    test_dataset = CLIPDataset('../data/clip_dataset/zeroshot_classification.json')
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    zeroshot(clip, testloader)
