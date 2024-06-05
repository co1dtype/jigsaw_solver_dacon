import cv2
import torch
from torch.utils.data import Dataset

class JigsawDataset(Dataset):
    def __init__(self, img_path_list, label_list, transform=None):
        self.img_path_list = img_path_list.reset_index(drop=True) 
        self.label_list = label_list
        self.transform = transform
        
    def __getitem__(self, index):
        img_path = self.img_path_list.iloc[index]['img_path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.label_list is not None:
            label = self.label_list[index]

            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed['image']

            label = torch.tensor(label, dtype=torch.long) - 1
            return image, label
        else:
            if self.transform is not None:
                image = self.transform(image=image)['image']
            return image
        
    def __len__(self):
        return len(self.img_path_list)
    

