import os
import glob
import pandas as pd
import string
import collections

from tqdm import tqdm


from PIL import Image

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim

data = glob.glob(os.path.join('input/captcha-version-2-images/samples', '*.png'))
path = 'input/captcha-version-2-images/samples'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_letters = string.ascii_lowercase + string.digits

mapping = {}
mapping_inv = {}
i = 1
for x in all_letters:
    mapping[x] = i
    mapping_inv[i] = x
    i += 1

num_class = len(mapping)

images = []
labels = []
datas = collections.defaultdict(list)
for d in data:
    x = os.path.basename(d)
    print(x) #in ra dataset
    datas['images'].append(x)
    datas['label'].append([mapping[i] for i in x.split('.')[0]])
df = pd.DataFrame(datas)
df.head()

if df.empty:
    print("DataFrame bị trống, không có dữ liệu để chia.")
else:
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)


class CaptchaDataset:
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image = Image.open(os.path.join(path, data['images'])).convert('L')
        label = torch.tensor(data['label'], dtype=torch.int32)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


transform = T.Compose([
    T.Resize((50, 200)),
    T.ToTensor()
])

train_data = CaptchaDataset(df_train, transform=transform)
test_data = CaptchaDataset(df_test, transform=transform)
def custom_collate_fn(batch):
    images, labels = zip(*batch)  # Tách dữ liệu và nhãn
    images = torch.stack(images)  # Xếp chồng các tensor hình ảnh
    
    # Tìm độ dài nhãn lớn nhất trong batch
    max_label_length = max(len(label) for label in labels)
    
    # Pad tất cả các nhãn để có cùng độ dài
    padded_labels = torch.zeros((len(labels), max_label_length), dtype=torch.int32)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label

    return images, padded_labels
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_data, batch_size=8, collate_fn=custom_collate_fn)

#train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
#test_loader = DataLoader(test_data, batch_size=8)

class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(inp, hidden, bidirectional=True)
        else:
            self.rnn = nn.GRU(inp, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden*2, out)
    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)
        return out


class CRNN(nn.Module):
    def __init__(self, in_channels, output):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 256, 9, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(3, 3),
                nn.Conv2d(256, 256, (4, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256))

        self.linear = nn.Linear(3328, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.rnn = Bidirectional(256, 1024, output+1)

    def forward(self, X, y=None, criterion = None):
        out = self.cnn(X)
        N, C, w, h = out.size()
        out = out.view(N, -1, h)
        out = out.permute(0, 2, 1)
        out = self.linear(out)

        out = out.permute(1, 0, 2)
        out = self.rnn(out)

        if y is not None:
            T = out.size(0)
            N = out.size(1)

            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            #target_lengths_1 = torch.full(size=(N,), fill_value=5, dtype=torch.int32)
            # Sử dụng chiều dài thật của từng nhãn (không tính padding)
            #target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.int32)
            target_lengths = torch.tensor([len(label[label > 0]) for label in y], dtype=torch.int32)
            
            loss = criterion(out, y, input_lengths, target_lengths)

            return out, loss
        
        return out, None

    def _ConvLayer(self, inp, out, kernel, stride, padding, bn=False):
        if bn:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU(),
                nn.BatchNorm2d(out)
            ]
        else:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU()
            ]
        return nn.Sequential(*conv)

class Engine:
    def __init__(self, model, optimizer, criterion, epochs=50, early_stop=False, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device

    def fit(self, dataloader):
        hist_loss = []
        for epoch in range(self.epochs):
            self.model.train()
            tk = tqdm(dataloader, total=len(dataloader))
            for data, target in tk:
                #print(f"Batch {i} - Data shape: {data.shape}, Target shape: {target.shape}")
                data = data.to(device=self.device)
                target = target.to(device=self.device)

                self.optimizer.zero_grad()

                out, loss = self.model(data, target, criterion=self.criterion)

                loss.backward()

                self.optimizer.step()


                tk.set_postfix({'Epoch':epoch+1, 'Loss' : loss.item()})

    def evaluate(self, dataloader):
        self.model.eval()
        loss = 0
        hist_loss = []
        outs = collections.defaultdict(list)
        tk = tqdm(dataloader, total=len(dataloader))
        with torch.no_grad():
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(device=self.device)

                out, loss = self.model(data, target, criterion=self.criterion)

                outs['pred'].append(out)
                outs['target'].append(target)


                hist_loss.append(loss)

                tk.set_postfix({'Loss':loss.item()})

        return outs, hist_loss

    def predict(self, image):
        image = Image.open(image).convert('L')
        image_tensor = T.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0)
        out, _ = self.model(image_tensor.to(device=self.device))
        out = out.permute(1, 0, 2)
        out = out.log_softmax(2)
        out = out.argmax(2)
        out = out.cpu().detach().numpy()

        return out




model = CRNN(in_channels=1, output=num_class).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CTCLoss()

engine = Engine(model, optimizer, criterion, device=DEVICE)
engine.fit(train_loader)
outs, loss = engine.evaluate(test_loader)

import matplotlib.pyplot as plt
import numpy as np

ids = np.random.randint(len(data))

image = data[ids]
out = engine.predict(image)[0]

def show_prediction(out, image):
    gt = image.split('/')[-1].split('.')[0]
    imagePIL = Image.open(image).convert('L')

    pred = ''
    then = 0
    for x in out:
        if then != x:
            if x > 0 :
                pred += mapping_inv[x]
        then = x

    plt.figure(figsize=(15, 12))
    img_array = np.asarray(imagePIL)
    plt.title(f'Ground Truth - {gt} || Prediction - {pred}')
    plt.axis('off')
    plt.imshow(img_array)

show_prediction(out, image)

saving = {'state_dict':engine.model.state_dict(),
          'optimizer':engine.optimizer.state_dict(),
         'mapping':mapping,
         'mapping_inv':mapping_inv}
torch.save(saving, 'model.pth')

