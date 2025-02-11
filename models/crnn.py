import torch.nn as nn
import torch

"""
Tạo một RNN hai chiều (BiLSTM).
Kết nối đầu ra của RNN với một lớp Linear để chuyển đổi số chiều từ hidden * 2 thành số lớp đầu ra out.
BiLSTM (hoặc BiGRU) xử lý chuỗi theo cả hai hướng (xuôi và ngược).
Sau đó, qua lớp Linear để giảm số chiều đầu ra.
"""
class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        self.rnn = nn.LSTM(inp, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3) if lstm else nn.GRU(inp, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.embedding = nn.Linear(hidden * 2, out)

    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)
        return out

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torchvision import models

class CRNN(nn.Module):
    def __init__(self, in_channels, output):
        super(CRNN, self).__init__()

        # Load ResNet-18 với trọng số pretrained, bỏ lớp fully connected cuối
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # Bỏ lớp fc để giữ đặc trưng từ ResNet
        # Điều chỉnh kênh đầu vào nếu không phải 3
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Adaptive pooling để giữ output width cố định là 64
        self.pool = nn.AdaptiveAvgPool2d((1, 64))

        # Sẽ được tính toán động dựa trên đầu ra từ ResNet
        self.linear_input_dim = None
        self.linear = None
        self.bn1 = nn.BatchNorm1d(256)
        self.rnn = Bidirectional(256, 1024, output + 1)

    def forward(self, X, y=None, criterion=None):
        # Qua ResNet-18 (trừ lớp fully connected)
        out = self.resnet(X)
        print("ResNet output shape:", out.shape)  # [batch_size, channels, height, width]
        
        out = self.pool(out)  # Giảm chiều height còn 1, giữ width cố định là 64
        print("After adaptive pooling:", out.shape)

        N, C, H, W = out.size()  # [batch_size, channels, height, width]
        
        # Thay đổi thứ tự để phù hợp với Linear
        out = out.permute(0, 3, 2, 1).reshape(N, W, C * H)  # [batch_size, width, C * H]
        print("Kích thước sau flatten:", out.shape)

        # Khởi tạo lớp Linear lần đầu khi forward
        if self.linear is None:
            self.linear_input_dim = out.shape[2]
            self.linear = nn.Linear(self.linear_input_dim, 256).to(X.device)

        # Qua Linear
        out = self.linear(out)
        out = self.bn1(out.permute(0, 2, 1))
        
        # Chuyển thành dạng phù hợp cho RNN
        out = out.permute(2, 0, 1)  # [sequence_length, batch_size, feature_dim]
        
        # Qua RNN
        out = self.rnn(out)

        if y is not None:
            T = out.size(0)
            N = out.size(1)
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            target_lengths = torch.tensor([len(label[label > 0]) for label in y], dtype=torch.int32)
            loss = criterion(out, y, input_lengths, target_lengths)
            return out, loss

        return out, None

