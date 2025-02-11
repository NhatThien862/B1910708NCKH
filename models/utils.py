import os
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from CaptchaDataset.dataset import CaptchaDataset, custom_collate_fn, transform
from torch.utils.data import DataLoader

# Đường dẫn đến thư mục chứa ảnh và nhãn
image_folder = 'data/data2'  # Thư mục chứa các file .png
label_folder = 'data/data2'  # Thư mục chứa các file .txt

# Tạo danh sách các file ảnh (.png) và nhãn tương ứng (.txt)
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
print("Số lượng ảnh: ",len(image_files))
print("Số lượng labels: ",len(label_files))
# Kiểm tra nếu số lượng file ảnh và nhãn khớp
assert len(image_files) == len(label_files), "Số lượng file ảnh và nhãn không khớp!"

# Định nghĩa các ký tự cho ánh xạ
all_letters = " !$%&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}ÀÁÂÃÈÉÊÌÍÐÒÓÔÕÖÙÚÜÝàáâãèéêìíðòóôõöùúüýĀāĂăĐđĨĩŌōŨũŪūƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ–—’“”…™−"
mapping = {ch: idx + 1 for idx, ch in enumerate(all_letters)}
mapping_inv = {idx + 1: ch for idx, ch in enumerate(all_letters)}
num_class = len(mapping)

# Tạo danh sách images và labels từ các file ảnh và nhãn
images = []
labels = []
for image_file in image_files:
    # Lấy tên file mà không có đuôi mở rộng (.png)
    base_name = os.path.splitext(image_file)[0]
    label_file = f"{base_name}.txt"
    label_path = os.path.join(label_folder, label_file)

    # Đọc nhãn từ file .txt
    with open(label_path, 'r', encoding='utf-8-sig') as f:
        label_text = f.read().strip()  # Đọc nhãn

    # Chuyển nhãn thành các số nguyên theo ánh xạ
    label_int = [mapping[char] for char in label_text if char in mapping]

    # Thêm tên file ảnh và nhãn vào danh sách
    images.append(image_file)
    labels.append(label_int)

# Tạo một DataFrame từ danh sách ảnh và nhãn
df = pd.DataFrame({'images': images, 'label': labels})

# Chia dữ liệu thành tập huấn luyện (80%), kiểm tra (10%) và validation (10%)
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
df_test, df_val = train_test_split(df_test, test_size=0.5, shuffle=True)

# Tạo dataset và dataloader
train_data = CaptchaDataset(df_train, transform=transform)
test_data = CaptchaDataset(df_test, transform=transform)
val_data = CaptchaDataset(df_val, transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_data, batch_size=12, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_data, batch_size=8, collate_fn=custom_collate_fn)

def calculate_metrics(preds, targets):
    """Tính CER và WER từ output của model"""
    # Dịch target từ tensor sang string
    input_strs = [df_train['images'].iloc[idx] for idx in targets]
    predicted_strs = preds.argmax(dim=2).cpu().numpy()
    
    cer = wer = 0.0
    
    for i in range(len(input_strs)):
        ref = input_strs[i].lower()
        hyp = predicted_strs[i].lower()
        
        # Tính CER
        if len(ref) != len(hyp):
            continue
        errors = 0
        for r, h in zip(ref, hyp):
            if r != h:
                errors += 1
        cer += errors / len(ref)
        
        # Tính WER
        ref_list = list(ref)
        hyp_list = list(hyp)
        errors = []
        deleted = set()
        
        i, j = 0, 0
        while i < len(ref_list) or j < len(hyp_list):
            if j in deleted:
                j += 1
                continue
            if i < len(ref_list) and hyp_list[j] == ref_list[i]:
                i += 1
                j += 1
            else:
                errors.append(hyp_list[j])
                j += 1
        
        wer += sum(errors) / len(ref)
    
    return cer, wer
