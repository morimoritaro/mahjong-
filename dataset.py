import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )

transform = transforms.Compose([
    #transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
    ])

class MyDataset(data.Dataset):

    def __init__(self, path, data_num, transform=transform):
        self.transform = transform
        self.path = path + 'm_data/'
        self.anns_path = path + 'm_anns.txt'
        self.anns = open(self.anns_path)
        self.data_num = data_num
        self.data = []
        self.label = []
        lines = self.anns.readlines()
        for i in range(self.data_num):
            img_path = self.path + '{:0>5}.png'.format(i)
            img = Image.open(img_path)
            img=img.resize((224,224))
            self.data.append(img)
            label = int(lines[i].split()[1])
            self.label.append(label)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label =  self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

if __name__ == '__main__':
    dataset = MyDataset('dataset/',10)
    for i in range(10):
        data,t = dataset[i]
        #data = np.array(data)
        #data.show()
        print(data.shape)