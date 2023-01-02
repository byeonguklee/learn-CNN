import os
import glob
import cv2
from sklearn.model_selection import train_test_split

# train / val / test split
all_data_path = "./data"

labels = ['cloudy', 'desert', 'green_area', 'water']

for label in labels:
    os.makedirs(f"./dataset/train/{label}", exist_ok=True)
    os.makedirs(f"./dataset/val/{label}", exist_ok=True)
    os.makedirs(f"./dataset/test/{label}", exist_ok=True)
    all_data = glob.glob(os.path.join(all_data_path, label, "*.jpg"))
    train_data, val_data_temps = train_test_split(all_data, test_size=0.2, random_state=0)
    val_data, test_data = train_test_split(val_data_temps, test_size=0.5, random_state=0)

    for path in train_data:
        file_path = f"./dataset/train/{label}"
        file_name = os.path.basename(path).split(".")[0]
        img = cv2.imread(path)
        cv2.imwrite(os.path.join(file_path, file_name+".png"), img)

    for path in val_data:
        file_path = f"./dataset/val/{label}"
        file_name = os.path.basename(path).split(".")[0]
        img = cv2.imread(path)
        cv2.imwrite(os.path.join(file_path, file_name+".png"), img)

    for path in test_data:
        file_path = f"./dataset/test/{label}"
        file_name = os.path.basename(path).split(".")[0]
        img = cv2.imread(path)
        cv2.imwrite(os.path.join(file_path, file_name+".png"), img)

# method with function
def data_save(data, mode):
    for path in data:
        #0. get folder name & image name
        image_name = os.path.basename(path)
        image_name = image_name.replace(".jpg","")
        folder_name = path.split("/")[2]
        #1. make dir
        folder_path = f"./dataset/{mode}/{folder_name}"
        os.makedirs(folder_path.format(path), exist_ok=True)
        #2. read img
        img = cv2.imread(path)
        #3. save img
        cv2.imwrite(os.path.join(folder_path, image_name+".png"), img)

# data_save(["./data/cloudy/train_12.jpg"], mode="train")