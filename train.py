### train.py
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from models.crnn import CRNN
from CaptchaDataset.dataset import CaptchaDataset, custom_collate_fn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

def train_model(model, train_loader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        tk = tqdm(train_loader, total=len(train_loader))
        for data, target in tk:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out, loss = model(data, target, criterion=criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            tk.set_postfix({'Epoch': epoch + 1, 'Loss': loss.item()})

if __name__ == "__main__":
    # Load data and initialize model
    label_file = 'data/dataTrain/captcha-version-2-images/labels.json'
    image_path = 'data/dataTrain/captcha-version-2-images/samples'
    resized_image_path = 'data/dataTrain/captcha-version-2-images/resized_samples'
    desired_size = (925,53)
    from models.utils import df_test, df_train, num_class, mapping
    transform = T.Compose([
        T.Resize((53, 925)),
        T.ToTensor()
    ])
    
    train_data = CaptchaDataset(df_train, transform=transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    # Initialize model, optimizer, and loss function
    model = CRNN(in_channels=1, output=num_class).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(zero_infinity=True)

    train_model(model, train_loader, optimizer, criterion, DEVICE, epochs=50)

    # Save model
    saving = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'mapping': mapping
    }
    torch.save(saving, 'model5.pth')
    print("Model saved as model5.pth")