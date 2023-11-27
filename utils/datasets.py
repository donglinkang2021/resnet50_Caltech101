import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.data_prepare import load_class_list

class CaltechDataset(Dataset):
    """Caltech 101 dataset."""

    def __init__(self, root_dir, data_path, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.data_path = data_path
        self.transform = transform
        
        # Load the data
        self.img_paths = []
        self.labels = []
        self.idx_to_label = load_class_list()
        self.num_classes = len(self.idx_to_label)
        self.load_data()
        
    def load_data(self):
        """
        Load the data from the root directory.
        """
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.info = f.readlines()
        for img_info in self.info:
            img_path, label = img_info.strip().split('\t')
            self.img_paths.append(img_path)
            self.labels.append(int(label))
        
            
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

def get_dataloader(batch_size = 32, shuffle=True):
    """
    Get train and val dataloader of Caltech 101 dataset.
    """
    transform_train=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    transform_val=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    
    train_dataset = CaltechDataset(
        root_dir = './data/dataset/images', 
        data_path = './data/process/train_list.txt', 
        transform = transform_train
    )
    val_dataset =  CaltechDataset(
        root_dir = './data/dataset/images', 
        data_path = './data/process/val_list.txt', 
        transform = transform_val
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, val_dataloader