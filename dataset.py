import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm,trange
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )

transform = transforms.Compose([
    #transforms.Resize((224,224)),
    transforms.ToTensor(),
    #normalize
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
        classes = np.array([2,5,8,11,14])
        for i in trange(self.data_num,desc='imageloading'):
            img_path = self.path + '{:0>5}.png'.format(i)
            img = Image.open(img_path)
            nimg = np.array(img)

            h,w,c = nimg.shape
            img = nimg[int(h/3):int(h*3/4),int(w/5):int(w*6/7),:]
            img= Image.fromarray(img)
            img=img.resize((224,224))
            self.data.append(img)
            label = int(lines[i].split()[1])
            label=np.where(classes==label)

            self.label.append(label[0][0])

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
        data = transforms.functional.to_pil_image(data)

        data.show()

