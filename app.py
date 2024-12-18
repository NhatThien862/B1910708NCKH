### test.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import pandas as pd
import torchvision.transforms as T
from models.crnn import CRNN
from CaptchaDataset.dataset import CaptchaDataset, custom_collate_fn
from engine import *
from models.utils import test_loader
from torchvision import transforms
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from models.utils import num_class
"""

criterion = nn.CTCLoss(zero_infinity=True)

outs, loss = Engine.evaluate(test_loader)
print(f"Average loss: {sum(loss)/len(loss)}")"""
# Đường dẫn tới file .pth
model_path = 'model5.pth'
# Tải lại model từ file .pth
checkpoint = torch.load(model_path,weights_only=True)
num_classes = len(checkpoint['mapping']) 
#print(num_classes)
#model = CRNN(in_channels=1, output=len(checkpoint['mapping']))  # Chỉnh lại tham số 'output' cho phù hợp
model = CRNN(in_channels=1, output=num_classes).to(DEVICE)
model.load_state_dict(checkpoint['state_dict'],strict=False)
model.eval()  # Chuyển mô hình sang chế độ evaluation

# Chuyển đổi ảnh đầu vào
def predict_image(image_path):
    
    image = Image.open(image_path).convert('L')
    
    transform = transforms.Compose([transforms.Resize((53, 925)), transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE) # Thêm một chiều cho batch size

    # Dự đoán với mô hình
    with torch.no_grad():
        output, _ = model(image_tensor)  # Không cần tính loss cho dự đoán
        output = output.argmax(2)  # Chọn nhãn có xác suất cao nhất
        pred = ' '
        for x in output.squeeze():
            if x > 0:  # Loại bỏ ký hiệu padding hoặc ký tự đặc biệt
                char = checkpoint['mapping_inv'][x.item()]
                pred +=  char 
    return pred

#Loại bỏ các ký tự trùng lặp liên tiếp trong một chuỗi văn bản

def remove_duplicates(text):
    if len(text) > 1:
        letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
    elif len(text) == 1:
        letters = [text[0]]
    else:
        return ""
    return "".join(letters)
#Sử dụng hàm remove_duplicates để sửa các dự đoán.
def correct_prediction(word):
    parts = word.split("-") #Tách từ theo dấu gạch ngang (-) thành các phần.
    parts = [remove_duplicates(part) for part in parts] #Áp dụng hàm remove_duplicates cho từng phần.
    corrected_word = "".join(parts) #Kết hợp lại các phần thành một từ hoàn chỉnh.
    return corrected_word
##### đoạn này hỗ trợ chạy trên console #####
# Nhập tên ảnh
while True:
    
    name = input("Nhập ảnh bạn muốn nhận diện (số có 4 chữ số nhỏ hơn 1822): ")
    if name.lower() == "quit" :  # Nếu người dùng nhập "quit", thoát vòng lặp
        print("Chương trình kết thúc.")
        break
    """if int(name)<=1822:
        name = name+"samples.png"
    else:
        print("Số bạn nhập không nằm trong cơ sở dữ liệu")"""
    name = name+"samples.png"
    image_path = "data/captcha-version-2-images/resized_samples/" + name
    
    prediction = predict_image(image_path) ## chạy
    rs = correct_prediction(prediction)
    #print("Dự đoán: " +prediction)
    print("Dự đoán: " +rs)
    
#image_path = "D:/NCKH/input/captcha-version-2-images/resized_samples/"+img_names[0] # Đường dẫn tới ảnh muốn dự đoán
#image_path = "input2/captcha-version-2-images/resized_samples/0026samples.png" 
#print(image_path)
#print(img_names[0])
#In kết quả ra file txt
"""filepath = "ketqua.txt"
with open(filepath, "w", encoding="utf-8") as file:
    file.write(prediction)
    #print("KQ da duoc ghi vao file ketqua.txt")"""