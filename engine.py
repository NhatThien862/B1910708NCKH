import tqdm, torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from Levenshtein import distance
from tqdm import tqdm
from jiwer import wer

class Engine:
    def __init__(self, model, optimizer, criterion, epochs=50, early_stop=False, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device

        # Khởi tạo scheduler
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )

    @staticmethod
    def ctc_decode(outputs, mapping_inv):
        if mapping_inv is None:
            raise ValueError("mapping_inv chưa được khởi tạo. Hãy kiểm tra và đảm bảo nó được truyền đúng.")
        # Decode CTC outputs
        decoded_sequences = []
        for out in outputs:
            sequence = []
            prev_char = None
            for idx in out:
                char = mapping_inv.get(idx.item(), '')
                if idx.item() != prev_char and idx.item() != 0:  # Bỏ qua blank index
                    sequence.append(char)
                prev_char = idx.item()
            decoded_sequences.append(''.join(sequence))
        return decoded_sequences

    @staticmethod
    def cer(target, output):
        # Tính CER (Character Error Rate)
        if not target or not output:
            return 1.0  # Nếu chuỗi rỗng, trả về lỗi 100%
        return distance(target, output) / len(target)

    @staticmethod
    def wer(target, output):
        # Tính WER (Word Error Rate)
        return wer(target, output)

    @staticmethod
    def calculate_metrics(outputs, targets, mapping_inv):
        if outputs is None or targets is None:
            return 1.0, 1.0  # CER và WER mặc định nếu không hợp lệ

        outputs = outputs.permute(1, 0, 2).argmax(2)  # Convert to indices
        outputs_decoded = Engine.ctc_decode(outputs, mapping_inv)
        targets_decoded = ["".join([mapping_inv[idx.item()] for idx in target if idx.item() > 0]) for target in targets]
  # Assume targets are strings directly

        total_cer, total_wer = 0.0, 0.0
        for output, target in zip(outputs_decoded, targets_decoded):
            total_cer += Engine.cer(target, output)
            total_wer += Engine.wer(target, output)

        avg_cer = total_cer / len(targets_decoded) if targets_decoded else 1.0
        avg_wer = total_wer / len(targets_decoded) if targets_decoded else 1.0

        return avg_cer, avg_wer

    def fit(self, dataloader, val_loader=None, mapping_inv=None, patience=5):
        train_loss_history = []
        val_loss_history = []
        train_cer_history = []
        train_wer_history = []
        val_cer_history = []
        val_wer_history = []

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            train_cer, train_wer = 0, 0
            tk = tqdm(dataloader, total=len(dataloader))
            for data, target in tk:
                print(f"Batch size: {data.size(0)}")  # Debug thông tin batch size
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                out, loss = self.model(data, target, criterion=self.criterion)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()

                train_loss += loss.item()
                cer, wer = Engine.calculate_metrics(out, target, mapping_inv)
                train_cer += cer
                train_wer += wer
                tk.set_postfix({'Epoch': epoch + 1, 'Loss': loss.item(), 'CER': cer, 'WER': wer})

            train_loss /= len(dataloader)
            train_cer /= len(dataloader)
            train_wer /= len(dataloader)
            train_loss_history.append(train_loss)
            train_cer_history.append(train_cer)
            train_wer_history.append(train_wer)

            if val_loader:
                val_loss, val_cer, val_wer = self.evaluate(val_loader, mapping_inv)
                val_loss_history.append(val_loss)
                val_cer_history.append(val_cer)
                val_wer_history.append(val_wer)
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), "best_model.pth")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                val_loss_history.append(None)
                val_cer_history.append(None)
                val_wer_history.append(None)

            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}, CER: {train_cer:.4f}, WER: {train_wer:.4f}")
            if val_loader:
                print(f"Validation Loss: {val_loss:.4f}, CER: {val_cer:.4f}, WER: {val_wer:.4f}")

        return train_loss_history, train_cer_history, train_wer_history, val_loss_history, val_cer_history, val_wer_history


    def evaluate(self, dataloader, mapping_inv):
        self.model.eval()
        hist_loss = []
        total_cer, total_wer = 0.0, 0.0
        tk = tqdm(dataloader, total=len(dataloader))
        with torch.no_grad():
            for data, target in tk:
                data, target = data.to(self.device), target.to(self.device)
                out, loss = self.model(data, target, criterion=self.criterion)
                cer, wer = Engine.calculate_metrics(out, target, mapping_inv)
                hist_loss.append(loss.item())
                total_cer += cer
                total_wer += wer
                tk.set_postfix({'Loss': loss.item(), 'CER': cer, 'WER': wer})

        avg_loss = sum(hist_loss) / len(hist_loss) if hist_loss else 1.0
        avg_cer = total_cer / len(dataloader) if len(dataloader) > 0 else 1.0
        avg_wer = total_wer / len(dataloader) if len(dataloader) > 0 else 1.0

        return avg_loss, avg_cer, avg_wer

    def predict(self, image_path, mapping_inv):
        image = Image.open(image_path).convert('L')
        image_tensor = T.ToTensor()(image).unsqueeze(0).to(self.device)
        out, _ = self.model(image_tensor)
        out = out.permute(1, 0, 2).argmax(2)
        out_decoded = Engine.ctc_decode(out, mapping_inv)
        return out_decoded


def plot_metrics(train_loss, train_cer, train_wer, val_loss=None, val_cer=None, val_wer=None):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(18, 6))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    if val_loss is not None:
        plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    # Plot CER
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_cer, label='Train CER', marker='o')
    if val_cer is not None:
        plt.plot(epochs, val_cer, label='Validation CER', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('CER')
    plt.legend()
    plt.title('CER over Epochs')

    # Plot WER
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_wer, label='Train WER', marker='o')
    if val_wer is not None:
        plt.plot(epochs, val_wer, label='Validation WER', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('WER')
    plt.legend()
    plt.title('WER over Epochs')

    plt.show()
