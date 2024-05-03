import pickle
import matplotlib.pyplot as plt

loss_save_path = 'loss_data.pkl'
with open(loss_save_path, 'rb') as f:
    loss_data = pickle.load(f)

train_loss = loss_data['train_loss']
val_loss = loss_data['val_loss']


plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
plt.title('Train Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./train_sf_plot.png')
