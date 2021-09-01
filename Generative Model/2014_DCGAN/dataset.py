import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import skimage.io as io
import PIL.Image as Image
from .Config import folder_path, image_size, batch_size

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.5, 0.5, 0.5],
                                                     std = [0.5, 0.5, 0.5])])
class DCGANdataset(Dataset):
    def __init__(self, folder_path, transform):
        self.img_paths = os.listdir(folder_path)
        self.img_paths = list(map(lambda x: os.path.join(folder_path,x), self.img_paths))
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = self.img_paths[index]
        img = io.imread(img_name)
        img = Image.fromarray(img)
        if self.transform:
            return torch.Tensor(self.transform(img))
        else:
            return torch.Tensor(img)
            
dataset = DCGANdataset(folder_path, transform = transform)       
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
