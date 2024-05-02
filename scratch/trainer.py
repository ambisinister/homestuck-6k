import os
import cv2
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import pickle

# local
from models import CLIP
from clip_dataset import DummyDataset, CLIPDataset

def train(model, trainloader, valloader, optim, criterion, lr_scheduler,
          step='batch', epochs=3, temp=1.0):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        print(f"Begin epoch {epoch+1}")
        # Training
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1} (Training)", unit="batch")
        
        for i, batch in enumerate(progress_bar): 
            im_embeds, txt_embeds = model(batch)
            im_logits = im_embeds @ txt_embeds.T / temp
            txt_logits = txt_embeds @ im_embeds.T / temp

            targets = torch.arange(batch['img'].size(0)).to(batch['img'].device)
            
            txt_loss = criterion(im_logits, targets)
            im_loss = criterion(txt_logits, targets)
            loss = (im_loss + txt_loss) / 2.0
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            #if step == 'batch':
            #    lr_scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(valloader, desc=f"Epoch {epoch+1} (Validation)", unit="batch")
            for i, batch in enumerate(progress_bar):
                im_embeds, txt_embeds = model(batch)
                im_logits = im_embeds @ txt_embeds.T / temp
                txt_logits = txt_embeds @ im_embeds.T / temp
                
                targets = torch.arange(batch['img'].size(0)).to(batch['img'].device)
                
                txt_loss = criterion(im_logits, targets)
                im_loss = criterion(txt_logits, targets)
                loss = (im_loss + txt_loss) / 2.0
                
                val_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(valloader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # checkpoint
        checkpoint_path = f'./checkpoints/checkpoint_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
    
    return model, train_losses, val_losses 

if __name__ == '__main__':
    clip = CLIP()
    batch_size = 32
    train_dataset = CLIPDataset('../data/clip_dataset/train.json')
    val_dataset = CLIPDataset('../data/clip_dataset/val.json')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clip.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    trained_clip, train_loss, val_loss = train(clip, trainloader, valloader,
                                               optimizer, criterion, lr_scheduler, epochs=5)

    save_path = 'DIY_CLIP.pth'
    torch.save(trained_clip.state_dict(), save_path)
    print(f"Trained model saved to {save_path}")

    loss_save_path = 'loss_data.pkl'
    with open(loss_save_path, 'wb') as f:
        pickle.dump({'train_loss': train_loss, 'val_loss': val_loss}, f)
    print(f"Loss curve data saved to {loss_save_path}")

