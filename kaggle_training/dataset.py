import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random

class LevirCDDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256):
        """
        root_dir should contain 'train', 'val', 'test' folders.
        Inside 'train', we expect 'A', 'B', 'label'.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.img_size = img_size
        self.dir_A = os.path.join(self.root_dir, 'A')
        self.dir_B = os.path.join(self.root_dir, 'B')
        self.dir_label = os.path.join(self.root_dir, 'label')
        
        # Only keep files that exist in all three directories
        a_files = set(os.listdir(self.dir_A))
        self.image_names = list(a_files)
        
        # Standard augmentations and normalizations
        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Target mask transform
        self.mask_transform = T.Compose([
            T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        path_A = os.path.join(self.dir_A, img_name)
        path_B = os.path.join(self.dir_B, img_name)
        path_label = os.path.join(self.dir_label, img_name)
        
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        label = Image.open(path_label).convert('L')
        
        # Joint data augmentation (random horizontal and vertical flips)
        if random.random() > 0.5:
            img_A = T.functional.hflip(img_A)
            img_B = T.functional.hflip(img_B)
            label = T.functional.hflip(label)
            
        if random.random() > 0.5:
            img_A = T.functional.vflip(img_A)
            img_B = T.functional.vflip(img_B)
            label = T.functional.vflip(label)

        # Apply transforms
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        label = self.mask_transform(label)
        
        # Ensure mask is binary 0 or 1
        label = (label > 0.5).float()
        
        return img_A, img_B, label
