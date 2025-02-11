from PIL import Image
import torch
import os
import torchvision.transforms as T
image_path = 'data/data2'
# Các phép biến đổi ảnh
transform = T.Compose([
    #T.Resize((105, 1853)), #note lại kich thước góc của ảnh cần train thôi ạ
    T.Resize((53, 925)),
    T.ToTensor()
    
])
#Resize ảnh trong thư mục train và đưa vào thư mục resized_train
def batch_resize(input_folder, output_folder, size):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.png')):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            try:
                with Image.open(input_path) as img:
                    print(f"Resizing {file_name}...")
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    img.save(output_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    print(f"Resized all images and saved to {output_folder}")
class CaptchaDataset:
    def __init__(self, df, transform=None):
        self.df = df  
        self.transform = transform 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx] #tải dữ liệu từ DataFrame df chứa thông tin ảnh và nhãn.
        image = Image.open(os.path.join(image_path, data['images'])).convert('L')
        label = torch.tensor(data['label'], dtype=torch.int32)

        if self.transform is not None:
            image = self.transform(image) #áp dụng chuyển đổi (transform) lên ảnh (nếu được).

        return image, label
#chuẩn bị batch dữ liệu với padding cho nhãn để đảm bảo kích thước đồng nhất.
def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    max_label_length = max(len(label) for label in labels)
    # khởi tạopadded_labels với torch.zeros, đảm bảo các phần tử được thêm vào không gây ảnh hưởng đến việc tính loss.
    padded_labels = torch.zeros((len(labels), max_label_length), dtype=torch.int32)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    return images, padded_labels
