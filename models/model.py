import torch
from models.crnn import CRNN
from models.utils import  num_class
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sequence_length = 64 
model = CRNN(in_channels=1, output=num_class).to(DEVICE)

