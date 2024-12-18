import torch.nn as nn
import torch
# Lớp CRNN và các phần khác của mô hình 

"""
Tạo một RNN hai chiều (BiLSTM).
Kết nối đầu ra của RNN với một lớp Linear để chuyển đổi số chiều từ hidden * 2 thành số lớp đầu ra out.
BiLSTM (hoặc BiGRU) xử lý chuỗi theo cả hai hướng (xuôi và ngược).
Sau đó, qua lớp Linear để giảm số chiều đầu ra.
"""
class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        self.rnn = nn.LSTM(inp, hidden, bidirectional=True) if lstm else nn.GRU(inp, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden * 2, out)

    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)
        return out
class CRNN(nn.Module):
    def __init__(self, in_channels, output):
        super(CRNN, self).__init__()
        #phần CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 256, 9, stride=1, padding=1), #số kênh là 256, kernel size lớp đầu tiên là 9x9
            nn.ReLU(),
            nn.BatchNorm2d(256), #chuẩn hóa đặc trưng để tăng tốc độ hội tụ
            nn.MaxPool2d(3, 3), # giảm kích thước không gian của ảnh (bằng cách chọn giá trị lớn nhất trong mỗi vùng).
            nn.Conv2d(256, 256, (4, 3), stride=1, padding=1), #kernel size lớp thứ hai là 4x3
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        #phần RNN
        self.linear = nn.Linear(3584, 256) #Giảm chiều dữ liệu đầu ra từ CNN
        self.bn1 = nn.BatchNorm1d(256) #Chuẩn hóa các đặc trưng theo batch.
        self.rnn = Bidirectional(256, 1024, output + 1) #BiLSTM xử lý chuỗi đầu ra từ CNN.
    #Đầu vào ảnh (X) qua CNN để trích xuất đặc trưng
    def forward(self, X, y=None, criterion=None):
        out = self.cnn(X)
        #print(f"Shape after CNN: {out.shape}")  # Debug
        #Định hình đầu ra từ CNN thành (batch_size, sequence_length, feature_dim).
        N, C, w, h = out.size()
        out = out.view(N, -1, h)
        out = out.permute(0, 2, 1) #thay đổi thứ tự trục trước khi đưa vào RNN
        out = self.linear(out)
        out = out.permute(1, 0, 2)
        out = self.rnn(out) #đưa chuỗi đã qua xử lý vào RNN.
        #Tính Loss(nếu cần)
        if y is not None:
            T = out.size(0)
            N = out.size(1)
            #độ dài chuỗi đầu vào
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            #độ dài chuỗi nhãn đầu ra
            target_lengths = torch.tensor([len(label[label > 0]) for label in y], dtype=torch.int32)
            loss = criterion(out, y, input_lengths, target_lengths)
            #nếu có nhãn (y), trả về đầu ra và loss.
            return out, loss
        #nếu không, chỉ trả về đầu ra.
        return out, None
