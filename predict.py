import torch
import torchvision.transforms as T
from PIL import Image

from models.crnn import CRNN
from models.utils import num_class
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

if __name__ == "__main__":
    # Load model
    model = CRNN(in_channels=1, output=num_class).to(DEVICE)
    checkpoint = torch.load('model5.pth',weights_only = True) #Đảm bảo chỉ tải phần trọng số của mô hình.
    model.load_state_dict(checkpoint['state_dict'])

    # Load and preprocess image
    image_path = 'data/captcha-version-2-images/resized_samples/0006samples.png'
    image = Image.open(image_path).convert('L')
    transform = T.Compose([
        T.Resize((53, 925)),
        T.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE) #Đưa tensor sang thiết bị 

    # Tạo prediction
    model.eval()# Chuyển mô hình sang chế độ evaluation
    with torch.no_grad():
        out, _ = model(image_tensor)
        # Thay đổi thứ tự các chiều để phù hợp với định dạng đầu ra.   
        out = out.permute(1, 0, 2).log_softmax(2).argmax(2) #Áp dụng hàm softmax trên mỗi bước thời gian, lấy chỉ số lớp có xác suất cao nhất.
        out = out.cpu().detach().numpy() #Chuyển tensor từ GPU về CPU và đổi sang dạng NumPy array.
    print("Predicted output:", out)
