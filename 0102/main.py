from customdataset import customDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from utils import train, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# agg
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomRotation(20),
    # if use PIL & torch.transforms, it works depends on sort (ToTensor -> Nromalize)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# train / val / test dataset
train_dataset = customDataset("./dataset/train", transform=train_transform)
val_dataset = customDataset("./dataset/val", transform=val_transform)
test_dataset = customDataset("./dataset/test", transform=test_transform)

# tran / val test loader
train_loader = DataLoader(train_dataset, batch_size=126, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=126, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# for i, (img, label) in enumerate(train_loader):
#     print(img, label)
#     exit()

# model
net = models.resnet18(pretrained=False) # 테스트를 위해 False로 바꿔줌
in_feature_val = net.fc.in_features
net.fc = nn.Linear(in_feature_val, 4)
net.to(device)

# model loader ## 테스트용 코드
net.load_state_dict(torch.load("./best.pt", map_location=device))
# map location: 다양한 사용자마다 device가 다른 것을 자동으로 바꿔 줌

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

# def train(n_epochs, train_loader, val_loader, model, optimizer,
#           criterion, device, save_path, last_validation_loss=None):
if __name__ == "__main__":
    # train(100, train_loader, val_loader, net, optimizer, criterion, device, save_path="./best.pt")
    test(net, test_loader, device)

