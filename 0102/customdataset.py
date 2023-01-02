from torch.utils.data import Dataset
import os
import glob
from PIL import Image


class customDataset(Dataset):
    def __init__(self, path, transform=None):
        # path => dataset/train/
        self.all_image_path = glob.glob(os.path.join(path, "*", "*.png"))
        self.transform = transform
        self.label_dict = {"cloudy": 0, "desert": 1, "green_area": 2, "water": 3}

        ## if you want to read image at init
        # self.img_list = []
        # for img_path in self.all_image_path:
        #     self.img_list.append(Image.open(img_path))
        ## getitem add below
        # img = self.img_list[item]

    def __getitem__(self, item):
        img_path = self.all_image_path[item]
        img = Image.open(img_path)
        label_temp = img_path.split("\\")
        label = self.label_dict[label_temp[1]]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.all_image_path)

# test = customDataset("./dataset/train", transform=None)
# for i in test:
#     print(i)