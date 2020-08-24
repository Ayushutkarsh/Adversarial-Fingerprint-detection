import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image



class ImageLivDet(data.Dataset):
    def __init__(self,root,image_size=224,mode='train',return_filename = True):
        self.root = root
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))
    def __getitem__(self,index):
        image_path = self.image_paths[index]
        
        filename = image_path.split('/')[-1]
        
        file_label = filename.split('.')[0]
        
        label = file_label.split('_')[1]
        int_label = 0
        if label =='surface':
            int_label = 1
        
     
        image = Image.open(image_path)
        im_np =np.asarray(image)
        image = image.convert('RGB')
        aspect_ratio = image.size[1]/image.size[0]
        Transform = []
        Transform.append(T.Resize((self.image_size,self.image_size)))
        
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image = Transform(image)
        Norm_ = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        image = Norm_(image)
        
        return image,int_label,file_label
    def __len__(self):
            return len(self.image_paths)
    
    
    
class ImageFolder(data.Dataset):
    def __init__(self, root,image_size=224,mode='train',augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.GT_paths = root[:-1]+'_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0,90,180,270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = image_path.split('/')[-1]
        GT_path = self.GT_paths + filename
        image = Image.open(image_path)
        
        GT = Image.open(GT_path)
        GT = GT.convert('L')
        image = image.convert('RGB')
        aspect_ratio = image.size[1]/image.size[0]

        Transform = []

        ResizeRange = random.randint(300,320)
        Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            RotationDegree = random.randint(0,3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1/aspect_ratio

            Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
           

            RotationRange = random.randint(-10,10)
            Transform.append(T.RandomRotation((RotationRange,RotationRange)))
            CropRange = random.randint(250,270)
            Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
            Transform = T.Compose(Transform)
            image = Transform(image)
            GT = Transform(GT)

            ShiftRange_left = random.randint(0,20)
            ShiftRange_upper = random.randint(0,20)
            ShiftRange_right = image.size[0] - random.randint(0,20)
            ShiftRange_lower = image.size[1] - random.randint(0,20)
            image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
            GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

            Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

            image = Transform(image)

            Transform =[]


        Transform.append(T.Resize((int(512*aspect_ratio)-int(512*aspect_ratio)%16,512)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image = Transform(image)
        GT = Transform(GT)
        Norm_ = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        image = Norm_(image)
        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

    

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4, model_mode='classification'):
    """Builds and returns Dataloader."""
    if model_mode == 'classification':
        dataset = ImageLivDet(root = image_path, image_size =image_size, mode=mode, return_filename = True)
    else:
        dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
        
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
