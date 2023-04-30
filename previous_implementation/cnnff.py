import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from utilities import *
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class CNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, config):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.opt = SGD(self.conv2d.parameters(), lr=config['lr'])
        self.threshold = config['threshold']

    def forward(self, x):
        x_direction = x.reshape((x.shape[0], -1)) / (x.reshape((x.shape[0], -1)).norm(2, 1, keepdim=True) + 1e-4) #normalise the input to prevent the magnitude from affecting the layer computation
        x_direction = x_direction.reshape(x.shape)
        return self.relu(self.conv2d(x_direction))

    def train(self, x_pos, x_neg):
        act_pos = self.forward(x_pos)
        g_pos = act_pos.reshape((act_pos.shape[0],-1)).pow(2).mean(1)
        act_neg = self.forward(x_neg)
        g_neg = act_pos.reshape((act_pos.shape[0],-1)).pow(2).mean(1)
        loss = torch.log(1+torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return act_pos.detach(), act_neg.detach(),loss.item()

class CNNFF(torch.nn.Module):
    def __init__(self, dims, config):

        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers = self.layers + [CNNLayer(dims[d][0], dims[d][1], dims[d][2], dims[d][3], config).to(DEVICE)]

    def predict(self, x,num_classes=10):
        goodness_per_label = []
        for label in range(num_classes):
            h = overlay_y_on_x(x, label)
            goodness = []
            h = h.reshape((h.shape[0], 1, 28, 28))
            for layer in self.layers:
                h = layer.forward(h)
                goodness = goodness + [h.reshape((h.shape[0], -1)).pow(2).mean(1)]
            # print(len(goodness))
            # print(goodness[0].shape)
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        x_pos = x_pos.reshape((x_pos.shape[0], 1, 28, 28))
        x_neg = x_neg.reshape((x_neg.shape[0], 1, 28, 28))
        h_pos, h_neg = x_pos, x_neg
        loss=0.0
        for i, layer in enumerate(self.layers):
            # print("training layer: ", i)
            h_pos, h_neg,layer_loss = layer.train(h_pos, h_neg)
            loss+=layer_loss
        return loss / len(self.layers)