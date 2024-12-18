import tqdm,torch
from PIL import Image
import torchvision.transforms as T
import collections
import matplotlib.pyplot as plt
class Engine:
    def __init__(self, model, optimizer, criterion, epochs=50, early_stop=False, device='gpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device
        self.epoch_losses = []  # Danh sách lưu loss theo từng epoch
        self.iteration_losses = []  # Danh sách lưu loss từng batch
    """
    Huấn luyện mô hình qua nhiều epoch.
    Tính toán loss cho từng batch và toàn bộ epoch.
    Ghi lại giá trị loss vào danh sách iteration_losses và epoch_losses.
    """
    def fit(self, dataloader,model):
        for epoch in range(self.epochs):
            self.model.train()
            tk = tqdm(dataloader, total=len(dataloader))
            epoch_loss = 0  # Loss trung bình của epoch
            for data, target in tk:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                out, loss = self.model(data, target, criterion=self.criterion)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                self.optimizer.step()
                # Ghi loss của từng batch
                self.iteration_losses.append(loss.item())
                epoch_loss += loss.item()
                tk.set_postfix({'Epoch': epoch + 1, 'Batch Loss': loss.item()})

            # Ghi loss trung bình của epoch
            epoch_loss /= len(dataloader)
            self.epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            """Hiển thị biểu đồ:
                Biểu đồ epoch loss (loss trung bình mỗi epoch).
                Biểu đồ iteration loss (loss theo từng batch).
            """
            # Biểu đồ loss theo epoch
            ax1.plot(self.epoch_losses, marker='o')
            ax1.set_title("Epoch Loss")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Loss")
            ax1.grid(True)

            # Biểu đồ loss theo iteration (batch)
            ax2.plot(self.iteration_losses, color='orange', alpha=0.7)
            ax2.set_title("Iteration Loss")
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("Loss")
            ax2.grid(True)

            plt.tight_layout()
            plt.show()
    """
    Đánh giá mô hình trên dữ liệu không huấn luyện (validation/test set).
    Lưu giá trị loss của từng batch vào hist_loss.
    Lưu dự đoán và nhãn thực vào dictionary outs.
    """               
    def evaluate(self, dataloader):
        self.model.eval()
        hist_loss = []
        outs = collections.defaultdict(list)
        tk = tqdm(dataloader, total=len(dataloader))
        with torch.no_grad():
            for data, target in tk:
                data, target = data.to(self.device), target.to(self.device)
                out, loss = self.model(data, target, criterion=self.criterion)
                outs['pred'].append(out)
                outs['target'].append(target)
                hist_loss.append(loss.item())
                tk.set_postfix({'Loss': loss.item()})
        return outs, hist_loss
    
        """
        Dự đoán chuỗi ký tự từ một ảnh đầu vào.
        Chuyển đổi ảnh đầu vào thành tensor và đưa vào mô hình.
        Xuất ra chuỗi ký tự dự đoán dưới dạng chỉ số.
        """
    def predict(self, image_path):
        image = Image.open(image_path).convert('L')
        image_tensor = T.ToTensor()(image).unsqueeze(0).to(self.device)
        out, _ = self.model(image_tensor)
        out = out.permute(1, 0, 2).log_softmax(2).argmax(2)
        out = out.cpu().detach().numpy()
        return out


    