import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class TestCaltechDataset(Dataset):
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
        self.load_data()
        
    def load_data(self):
        """
        Load the data from the root directory.
        """
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.info = f.readlines()
        for img_info in self.info:
            img_path = img_info.strip()
            self.img_paths.append(img_path)
        
            
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
    

def get_test_dataloader(batch_size = 32, shuffle=False):
    """
    Get train and val dataloader of Caltech 101 dataset.
    """

    transform_test=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    
    test_dataset = TestCaltechDataset(
        root_dir = './data/dataset/images', 
        data_path = './data/dataset/test.txt', 
        transform = transform_test
    )

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return test_dataloader, test_dataset