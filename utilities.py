import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

from PIL import Image
import matplotlib.pyplot as plt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_loaders(batch_size):
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x)),
        ]
    )
    train_loader = DataLoader(
    MNIST("./data/", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True)

    test_loader = DataLoader(
    MNIST("./data/", train=False, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True)

    return train_loader, test_loader

def overlay_y_on_x(x, y, classes=10):
    x_ = x.clone()
    x_[:, :classes] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

def get_y_neg(y,num_classes=10):
    y_neg = y.clone()
    for idx, y_samp in enumerate(y):
        allowed_indices = list(range(num_classes))
        allowed_indices.remove(y_samp.item())
        y_neg[idx] = torch.tensor(allowed_indices)[
            torch.randint(len(allowed_indices), size=(1,))
        ].item()
    return y_neg.to(DEVICE)

def visualise_positive():
    r, c    = [5, 5]
    fig, ax = plt.subplots(r, c, figsize= (15, 15))
    k = 0

    dtl = DataLoader(
    MNIST("./data/", train=True, download=False, transform=Compose([ToTensor(),Lambda(lambda x: torch.flatten(x))])),
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True)

    for data in dtl:
        x, y = data
    
        for i in range(r):
            for j in range(c):
                mod=x[k].reshape((1,-1)) #add dim
                mod = overlay_y_on_x(mod,y[k])
                img = mod.reshape((28,28)).numpy()
                ax[i, j].imshow(img)
                ax[i, j].axis('off')
                k+=1
        break

    del dtl

def visualise_negative():
    r, c    = [5, 5]
    fig, ax = plt.subplots(r, c, figsize= (15, 15))

    k       = 0

    dtl = DataLoader(
    MNIST("./data/", train=True, download=False, transform=Compose([ToTensor(),Lambda(lambda x: torch.flatten(x))])),
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True)

    for data in dtl:
        x, y = data
    
        for i in range(r):
            for j in range(c):
                mod=x[k].reshape((1,-1))
                y_neg = get_y_neg(y[k].view(1,1))
                mod = overlay_y_on_x(mod,y_neg[0])
                img = mod.reshape((28,28)).numpy()
                ax[i, j].imshow(img)
                ax[i, j].axis('off')
                k+=1
        break

    del dtl