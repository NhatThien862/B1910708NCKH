from sklearn.model_selection import train_test_split
import json
import pandas as pd
from CaptchaDataset.dataset import CaptchaDataset, custom_collate_fn, transform
from torch.utils.data import DataLoader
"""
Mở và đọc nội dung file JSON chứa nhãn (labels.json).
Kiểm tra xem file có rỗng hay không.
Chuyển đổi nội dung JSON thành dictionary Python (labels_data)
"""
label_file = 'data/captcha-version-2-images/labels.json'
try:
    with open(label_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        #print(content)
        if not content.strip():  # Kiểm tra xem file có rỗng không
            print("File is empty.")
        else:
            labels_data = json.loads(content)  # Dùng json.loads thay vì json.load
            #print(labels_data)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Định nghĩa các ký tự cho ánh xạ
# Danh sách tất cả ký tự hợp lệ trong bộ dữ liệu (bao gồm cả ký tự tiếng Việt).
all_letters = " !$%&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}ÀÁÂÃÈÉÊÌÍÐÒÓÔÕÖÙÚÜÝàáâãèéêìíðòóôõöùúüýĀāĂăĐđĨĩŌōŨũŪūƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ–—’“”…™−"
# Ánh xạ mỗi ký tự trong all_letters sang một số nguyên (bắt đầu từ 1).
mapping = {ch: idx + 1 for idx, ch in enumerate(all_letters)}
# mapping_inv: Ánh xạ ngược từ số nguyên về ký tự.
mapping_inv = {idx + 1: ch for idx, ch in enumerate(all_letters)}
# Tổng số ký tự có trong all_letters, được dùng làm số lớp đầu ra cho mô hình.
num_class = len(mapping)

# Tạo dataframe từ dữ liệu nhãn
"""
Tạo danh sách images và labels từ dictionary labels_data.
Mỗi filename trong labels_data là tên file ảnh, mỗi text là chuỗi ký tự của nhãn tương ứng.
Sử dụng ánh xạ mapping để chuyển đổi mỗi ký tự trong nhãn thành số nguyên.
"""
images = []
labels = []
for filename, text in labels_data.items():
    images.append(filename)
    labels.append([mapping[char] for char in text if char in mapping])
"""
Tạo một DataFrame df chứa hai cột:
images: Tên file ảnh.
label: Danh sách số nguyên tương ứng với chuỗi nhãn.
"""
df = pd.DataFrame({'images': images, 'label': labels})

# Chia dữ liệu thành tập huấn luyện (100%) và kiểm tra(20%), Chia ngẫu nhiên bằng tham số shuffle
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
# Tạo dataset và dataloader
# Dựa trên DataFrame (df_train và df_test) để tải ảnh và nhãn tương ứng.
# Áp dụng các phép biến đổi (transform) để tiền xử lý ảnh (resize, chuyển qua tensor).
train_data = CaptchaDataset(df_train, transform=transform)
test_data = CaptchaDataset(df_test, transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_data, batch_size=12, collate_fn=custom_collate_fn)

